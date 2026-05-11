"""
How to use:

    uv run run.py dataset=ag_news
    uv run run.py dataset=ag_news model.embedding_dim=128 model.n_layers=2
    uv run run.py dataset=ag_news tokenizer=wordpiece training.lr=0.001

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
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer
from torchTextClassifiers.value_encoder import DictEncoder, ValueEncoder

# Suppress noisy logs
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"


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


def load_data(dataset_cfg: dict, seed: int):
    text_col     = dataset_cfg["text_col"]
    title_col    = dataset_cfg.get("title_col")
    label_col    = dataset_cfg["label_col"]
    label_offset = dataset_cfg.get("label_offset", 0)
    oos_label    = dataset_cfg.get("oos_label")
    cat_cols_cfg = dataset_cfg.get("categorical_cols", {})
    cat_cols     = list(cat_cols_cfg.keys())
    n_train      = dataset_cfg.get("train_size")  # None = take all available data minus val
    n_val        = dataset_cfg["val_size"]
    hf_config    = dataset_cfg.get("hf_config")

    ds = load_dataset(dataset_cfg["hf_path"], hf_config) if hf_config else load_dataset(dataset_cfg["hf_path"])

    train_split = dataset_cfg.get("train_split", "train")
    test_split  = dataset_cfg.get("test_split", "test")
    train_data  = ds[train_split].shuffle(seed=seed)

    if n_train is None:
        n_train = len(train_data) - n_val

    def from_split(data, start, end):
        """Extract a slice [start:end] from a HuggingFace Dataset split and return (X, y)."""
        texts  = data[text_col][start:end]
        titles = data[title_col][start:end] if title_col else None
        labels = _make_y(data[label_col][start:end], label_offset, oos_label)
        cats   = [data[col][start:end] for col in cat_cols]
        return _make_X(texts, titles, cats), labels

    X_train, y_train = from_split(train_data, 0, n_train)
    X_val, y_val     = from_split(train_data, n_train, n_train + n_val)

    test_data = ds[test_split].shuffle(seed=seed)
    n_test    = dataset_cfg.get("test_size", len(test_data))
    X_test, y_test = from_split(test_data, 0, n_test)

    value_encoder = None
    if cat_cols:
        cat_train = X_train[:, 1:] if X_train.ndim > 1 else None
        categorical_encoders = {}
        for i, col in enumerate(cat_cols):
            if cat_cols_cfg[col] is None:
                unique_vals = sorted(set(cat_train[:, i].tolist()))
                categorical_encoders[str(i)] = DictEncoder({v: idx for idx, v in enumerate(unique_vals)})
        label_enc = SKLabelEncoder().fit(y_train)
        value_encoder = ValueEncoder(
            label_encoder=label_enc,
            categorical_encoders=categorical_encoders if categorical_encoders else None,
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

# Hydra assembles the config from conf/config.yaml (which pulls in conf/dataset/, conf/model/, etc.)
# and passes it as `cfg`. CLI overrides like `model.embedding_dim=128` are applied on top.
# Example: uv run run.py dataset=sst2 model.embedding_dim=128 model.n_layers=2
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # cfg.dataset and cfg.tokenizer are DictConfig objects — convert to plain dicts
    # so they work with load_data() and build_tokenizer() which expect dict.get(), etc.
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tok_cfg     = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    # cfg.model and cfg.training stay as DictConfig — accessed via dot notation (m.embedding_dim)
    m           = cfg.model
    t           = cfg.training

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
    mlflow.set_experiment(dataset_name)

    X_train, y_train, X_val, y_val, X_test, y_test, value_encoder = load_data(dataset_cfg, cfg.seed)
    num_classes = len(np.unique(y_train))

    tokenizer = build_tokenizer(tok_cfg, X_train)

    # attention_config=None means no transformer block (fasttext-style mean pooling)
    # n_layers > 0 enables the transformer encoder on top of the embeddings
    attention_config = None
    if m.n_layers > 0:
        attention_config = {
            "n_layers": m.n_layers,
            "n_head": m.n_head,
            "n_kv_head": m.n_kv_head,
            "positional_encoding": m.positional_encoding,
            "sequence_len": m.sequence_len,
        }

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
        raw_labels=False,
        raw_categorical_inputs=value_encoder is not None,
        save_path=save_path,
    )

    with mlflow.start_run():
        vocab_size = getattr(tokenizer, "vocab_size", getattr(tokenizer, "num_tokens", None))
        mlflow.log_params({
            "dataset": dataset_name,
            "tokenizer": tok_cfg["type"],
            "emb_dim": m.embedding_dim,
            "n_layers": m.n_layers,
            "n_heads_label_attention": m.n_heads_label_attention,
            "agg": m.aggregation_method,
            "lr": t.lr,
            "batch_size": t.batch_size,
            "epochs": t.num_epochs,
            "n_classes": num_classes,
            "vocab_size": vocab_size,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        # Pass the active MLflow run_id to Lightning's MLFlowLogger so training metrics
        # (loss, val_loss per epoch) are logged into the same run as our params/metrics
        mlf_logger = MLFlowLogger(
            experiment_name=dataset_name,
            tracking_uri=tracking_uri or "mlruns",
            run_id=mlflow.active_run().info.run_id,
        )
        # log_every_n_steps=steps_per_epoch → log once per epoch, not every step (avoids slow HTTP calls)
        steps_per_epoch = max(1, len(X_train) // t.batch_size)
        training_config.trainer_params = {"logger": mlf_logger, "log_every_n_steps": steps_per_epoch}

        t0 = time.time()
        clf.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)
        train_time = round(time.time() - t0, 1)

        num_params = sum(p.numel() for p in clf.lightning_module.model.parameters())

        predict_batch_size = dataset_cfg.get("predict_batch_size", 512)

        def batch_predict(X):
            all_preds = []
            for i in range(0, len(X), predict_batch_size):
                batch = X[i:i + predict_batch_size]
                result = clf.predict(batch, raw_categorical_inputs=value_encoder is not None)
                pred = result["prediction"]
                pred = pred.squeeze(dim=-1).numpy() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze(axis=-1)
                all_preds.append(pred)
            return np.concatenate(all_preds)

        preds = batch_predict(X_test)

        test_accuracy = round(accuracy_score(y_test, preds), 4)
        test_f1_macro = round(f1_score(y_test, preds, average="macro"), 4)

        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "train_time_s": train_time,
            "num_params": num_params,
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        print(f"Test Accuracy : {test_accuracy:.4f}")
        print(f"Test F1 macro : {test_f1_macro:.4f}")
        print(f"Train time    : {train_time}s")


if __name__ == "__main__":
    main()
