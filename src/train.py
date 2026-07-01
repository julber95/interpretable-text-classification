"""
How to use:

    uv run python -m src.train dataset=ag_news
    uv run python -m src.train dataset=ag_news model.embedding_dim=128 model.n_layers=2
    uv run python -m src.train dataset=ag_news tokenizer=wordpiece training.lr=0.001

Override any config value from the CLI. Config groups:
    dataset:   ag_news | sst2 | imdb | amazon | 20newsgroups | subj | clinc150
    tokenizer: ngram | wordpiece
    model:     default (then override model.embedding_dim, model.n_layers, etc.)
    training:  default (then override training.lr, training.batch_size, etc.)
"""

import logging
import os
import time
import warnings
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer
from torchTextClassifiers.value_encoder import DictEncoder, ValueEncoder

# Suppress noisy logs
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data Loading ──────────────────────────────────────────────────────────────

def _make_X(texts, titles, cat_arrays):
    """Build the X array: text only (1D) or text + categorical features (2D)."""
    if titles is not None:
        texts = [(t + " " + b) if t is not None else b for t, b in zip(titles, texts)]  # prepend title to text
    if cat_arrays:
        # Stack text + categorical columns: col 0 = text, cols 1+ = raw categorical values
        return np.column_stack([np.array(texts)] + [np.array(c) for c in cat_arrays])
    return np.array(texts)


def _make_y(labels, label_offset, oos_label=None):
    return np.array([oos_label if v is None else int(v) + label_offset for v in labels])


def _resolve_n_train(n_train, total, subtract, fraction):
    if n_train is None:
        n_train = total - subtract
    if fraction < 1.0:
        n_train = max(1, int(n_train * fraction))
    return n_train


def load_data(dataset_cfg: dict, seed: int):
    text_col      = dataset_cfg["text_col"]
    title_col     = dataset_cfg.get("title_col")
    label_col     = dataset_cfg["label_col"]
    label_offset  = dataset_cfg.get("label_offset", 0)
    oos_label     = dataset_cfg.get("oos_label")
    cat_cols_cfg  = dataset_cfg.get("categorical_cols", {})
    cat_cols      = list(cat_cols_cfg.keys())
    n_train       = dataset_cfg.get("train_size")
    n_val         = dataset_cfg.get("val_size")
    val_split     = dataset_cfg.get("val_split")
    val_from_test = dataset_cfg.get("val_from_test", False)
    hf_config     = dataset_cfg.get("hf_config")
    data_files    = dataset_cfg.get("data_files")
    fraction      = dataset_cfg.get("train_fraction", 1.0)

    if data_files:
        from datasets import Dataset
        import pandas as pd
        path = data_files["train"]
        opts = None if path.startswith("http") else dataset_cfg.get("storage_options")
        all_data = pd.read_parquet(path, storage_options=opts)
        all_data = all_data.dropna(subset=[label_col, text_col]).reset_index(drop=True)
        label_enc = SKLabelEncoder().fit(all_data[label_col].astype(str))
        all_data[label_col] = label_enc.transform(all_data[label_col].astype(str))
        all_data = all_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        ds_full = Dataset.from_pandas(all_data, preserve_index=False)
        n_test  = dataset_cfg.get("test_size", max(1000, int(0.1 * len(ds_full))))
        n_val   = n_val or max(1000, int(0.1 * len(ds_full)))
        n       = len(ds_full)
        test_data  = ds_full.select(range(n - n_test, n))
        remainder  = ds_full.select(range(n - n_test))
        r          = len(remainder)
        val_data_p = remainder.select(range(r - n_val, r))
        train_data = remainder.select(range(r - n_val))
        if n_train is not None:
            train_data = train_data.select(range(min(n_train, len(train_data))))
        if fraction < 1.0:
            train_data = train_data.select(range(max(1, int(len(train_data) * fraction))))

        def from_parquet(data):
            return _make_X(data[text_col], data[title_col] if title_col else None,
                           [data[c] for c in cat_cols]), \
                   _make_y(data[label_col], label_offset, oos_label)

        X_train, y_train = from_parquet(train_data)
        X_val,   y_val   = from_parquet(val_data_p)
        X_test,  y_test  = from_parquet(test_data)
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test, \
               ValueEncoder(label_encoder=label_enc, categorical_encoders=None)

    ds = load_dataset(dataset_cfg["hf_path"], hf_config) if hf_config else load_dataset(dataset_cfg["hf_path"])
    train_split = dataset_cfg.get("train_split", "train")
    test_split  = dataset_cfg.get("test_split", "test")
    train_data  = ds[train_split].shuffle(seed=seed)
    test_data   = ds[test_split].shuffle(seed=seed)

    if dataset_cfg.get("filter_oos", False):
        train_data = train_data.filter(lambda x: x[label_col] is not None and str(x[label_col]) != "nan")
        test_data  = test_data.filter(lambda x: x[label_col] is not None and str(x[label_col]) != "nan")

    def from_split(data, start, end):
        return _make_X(data[text_col][start:end],
                       data[title_col][start:end] if title_col else None,
                       [data[c][start:end] for c in cat_cols]), \
               _make_y(data[label_col][start:end], label_offset, oos_label)

    n_test = dataset_cfg.get("test_size", len(test_data))

    if val_from_test:
        n_train = _resolve_n_train(n_train, len(train_data), 0, fraction)
        n_val   = n_val or len(test_data) // 2
        X_train, y_train = from_split(train_data, 0, n_train)
        X_val,   y_val   = from_split(test_data, 0, n_val)
        X_test,  y_test  = from_split(test_data, n_val, None)
    elif val_split:
        n_train  = _resolve_n_train(n_train, len(train_data), 0, fraction)
        val_data = ds[val_split].shuffle(seed=seed)
        if dataset_cfg.get("filter_oos", False):
            val_data = val_data.filter(lambda x: x[label_col] is not None and str(x[label_col]) != "nan")
        n_val    = n_val or len(val_data)
        X_train, y_train = from_split(train_data, 0, n_train)
        X_val,   y_val   = from_split(val_data, 0, n_val)
        X_test,  y_test  = from_split(test_data, 0, n_test)
    else:
        n_train = _resolve_n_train(n_train, len(train_data), n_val, fraction)
        X_train, y_train = from_split(train_data, 0, n_train)
        X_val,   y_val   = from_split(train_data, n_train, n_train + n_val)
        X_test,  y_test  = from_split(test_data, 0, n_test)

    value_encoder = None
    if cat_cols:
        cat_train = X_train[:, 1:] if X_train.ndim > 1 else None
        categorical_encoders = {
            str(i): DictEncoder({v: idx for idx, v in enumerate(sorted(set(cat_train[:, i].tolist())))})
            for i, col in enumerate(cat_cols) if cat_cols_cfg[col] is None
        }
        value_encoder = ValueEncoder(
            label_encoder=SKLabelEncoder().fit(y_train),
            categorical_encoders=categorical_encoders or None,
        )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, value_encoder


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(tok_cfg: dict, X_train: np.ndarray):
    texts = X_train[:, 0].tolist() if X_train.ndim > 1 else X_train.tolist()

    if tok_cfg["type"] == "ngram":
        tokenizer = NGramTokenizer(
            min_count=tok_cfg.get("min_count", 1),
            min_n=tok_cfg.get("min_n", 3),
            max_n=tok_cfg.get("max_n", 6),
            num_tokens=tok_cfg.get("num_tokens", 100000),
            len_word_ngrams=tok_cfg.get("len_word_ngrams", 1),
        )
        tokenizer.train(texts)
    elif tok_cfg["type"] == "wordpiece":
        from torchTextClassifiers.tokenizers import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(
            vocab_size=tok_cfg.get("vocab_size", 10000),
            output_dim=tok_cfg.get("output_dim", 128),
        )
        tokenizer.train(texts)
    else:
        raise ValueError(f"Unknown tokenizer: {tok_cfg['type']}")
    return tokenizer


# ── Run ───────────────────────────────────────────────────────────────────────

def _resolve_accelerator() -> str:
    """Test actual CUDA init — is_available() can return True even when init fails."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    accelerator = _resolve_accelerator()

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tok_cfg     = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    m, t        = cfg.model, cfg.training

    dataset_name = dataset_cfg["name"]

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} | Tokenizer: {tok_cfg['type']}")
    print(f"emb_dim={m.embedding_dim} | n_layers={m.n_layers} | label_attn={m.n_heads_label_attention}")
    print(f"lr={t.lr} | batch_size={t.batch_size}")
    print(f"{'='*50}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # 1 MLflow experiment per dataset — all hyperparameter runs for a dataset are grouped together
    experiment_name = cfg.get("experiment_name") or dataset_name
    mlflow.set_experiment(experiment_name)

    train_fraction = cfg.get("train_fraction", 1.0)
    dataset_cfg["train_fraction"] = train_fraction
    X_train, y_train, X_val, y_val, X_test, y_test, value_encoder = load_data(dataset_cfg, cfg.seed)
    num_classes = value_encoder.num_classes if value_encoder is not None else len(np.unique(y_train))

    tokenizer = build_tokenizer(tok_cfg, X_train)

    attention_config = {"n_layers": m.n_layers, "n_head": m.n_head, "n_kv_head": m.n_kv_head,
                        "positional_encoding": m.positional_encoding, "sequence_len": m.sequence_len,
                        } if m.n_layers > 0 else None

    model_config = ModelConfig(
        embedding_dim=m.embedding_dim,
        num_classes=num_classes,
        aggregation_method=m.aggregation_method,
        categorical_vocabulary_sizes=value_encoder.vocabulary_sizes if value_encoder else None,
        attention_config=attention_config,
        n_heads_label_attention=m.n_heads_label_attention,
    )
    clf = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config, value_encoder=value_encoder)

    save_path = str(RESULTS_DIR / "models" / dataset_name)

    training_config = TrainingConfig(
        num_epochs=t.num_epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        patience_early_stopping=t.patience_early_stopping,
        num_workers=t.num_workers,
        accelerator=accelerator,
        raw_labels=False,
        raw_categorical_inputs=value_encoder is not None,
        save_path=save_path,
        optimizer_params={"weight_decay": 1e-4},
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={"mode": "min", "factor": 0.5, "patience": 2},
    )

    with mlflow.start_run():
        vocab_size = getattr(tokenizer, "vocab_size", getattr(tokenizer, "num_tokens", None))
        mlflow.log_params({
            "dataset": dataset_name,
            "tokenizer": tok_cfg["type"],
            "emb_dim": m.embedding_dim,
            "n_layers": m.n_layers,
            "n_head": m.n_head if m.n_layers > 0 else None,
            "n_heads_label_attention": m.n_heads_label_attention,
            "agg": m.aggregation_method,
            "lr": t.lr,
            "batch_size": t.batch_size,
            "epochs": t.num_epochs,
            "n_classes": num_classes,
            "vocab_size": vocab_size,
            "train_fraction": train_fraction,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        mlf_logger = MLFlowLogger(
            experiment_name=dataset_name,
            tracking_uri=tracking_uri or "mlruns",
            run_id=mlflow.active_run().info.run_id,
        )
        steps_per_epoch = max(1, len(X_train) // t.batch_size)
        training_config.trainer_params = {"logger": mlf_logger, "log_every_n_steps": steps_per_epoch}

        t0 = time.time()
        clf.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)
        train_time = round(time.time() - t0, 1)

        # clf.train() leaves cached/fragmented CUDA memory from the optimizer and
        # scheduler state reloaded with the checkpoint; release it before predicting.
        if accelerator == "cuda":
            torch.cuda.empty_cache()

        num_params = sum(p.numel() for p in clf.lightning_module.model.parameters())

        predict_batch_size = dataset_cfg.get("predict_batch_size", 512)

        def batch_predict(X):
            all_preds = []
            for i in range(0, len(X), predict_batch_size):
                batch = X[i:i + predict_batch_size]
                with torch.no_grad():
                    result = clf.predict(batch, raw_categorical_inputs=value_encoder is not None, device=clf.device)
                pred = result["prediction"]
                pred = pred.squeeze(dim=-1).cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze(axis=-1)
                all_preds.append(pred)
            return np.concatenate(all_preds)

        preds = batch_predict(X_test)

        # For parquet datasets, y_test is integer-encoded but clf.predict() decodes preds
        # back to original string labels via value_encoder. Re-encode preds to int to match.
        y_test_eval = y_test
        preds_eval  = preds
        if (value_encoder is not None
                and np.issubdtype(np.array(y_test).dtype, np.integer)
                and len(preds) > 0
                and isinstance(preds[0], str)):
            preds_eval = value_encoder.label_encoder.transform(np.array(preds).astype(str))

        preds_path = RESULTS_DIR / f"predictions_{mlflow.active_run().info.run_id}.npz"
        np.savez(preds_path, X_test=X_test, y_test=y_test_eval, y_pred=preds_eval)
        mlflow.log_artifact(str(preds_path), artifact_path="predictions")
        preds_path.unlink()

        test_accuracy = round(accuracy_score(y_test_eval, preds_eval), 4)
        test_f1_macro = round(f1_score(y_test_eval, preds_eval, average="macro", zero_division=0), 4)

        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "train_time_s": train_time,
            "num_params": num_params,
        })


        mlflow.log_artifacts(save_path, artifact_path="model")

        print(f"Test Accuracy : {test_accuracy:.4f}")
        print(f"Test F1 macro : {test_f1_macro:.4f}")
        print(f"Train time    : {train_time}s")


if __name__ == "__main__":
    main()
