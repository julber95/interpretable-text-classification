"""
How to use:

    python run.py configs/<config_name.yaml> --dataset <dataset_name>

Arguments:
    config          Path to a YAML model config file (e.g. configs/fasttext.yaml)
    --dataset       Name of the dataset to run (must be defined in the config)

Examples:
    python run.py configs/fasttext.yaml --dataset sst2
    python run.py configs/fasttext.yaml --dataset ag_news

Results are saved as JSON files under benchmark/results/<model_name>/.
"""

import argparse
import json
import logging
import time

import warnings
from datetime import datetime
from pathlib import Path

import torch

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
    n_train      = dataset_cfg["train_size"]
    n_val        = dataset_cfg["val_size"]
    hf_config    = dataset_cfg.get("hf_config")

    ds = load_dataset(dataset_cfg["hf_path"], hf_config) if hf_config else load_dataset(dataset_cfg["hf_path"]) #Dataset Dict 

    train_split = dataset_cfg.get("train_split", "train")
    test_split  = dataset_cfg.get("test_split", "test")
    train_data  = ds[train_split].shuffle(seed=seed)

    def from_split(data, start, end):
        """Extract a slice [start:end] from a HuggingFace Dataset split and return (X, y)."""
        texts  = data[text_col][start:end]
        titles = data[title_col][start:end] if title_col else None
        labels = _make_y(data[label_col][start:end], label_offset, oos_label)
        cats   = [data[col][start:end] for col in cat_cols]  # raw values, encoding handled by ValueEncoder
        return _make_X(texts, titles, cats), labels

    X_train, y_train = from_split(train_data, 0, n_train)
    X_val, y_val     = from_split(train_data, n_train, n_train + n_val)

    test_data = ds[test_split].shuffle(seed=seed)
    n_test    = dataset_cfg.get("test_size", len(test_data))
    X_test, y_test = from_split(test_data, 0, n_test)

    # Build ValueEncoder for categorical columns with unknown vocab (string cols).
    # Categorical cols with a known vocab size in the config are already integers — no encoder needed.
    value_encoder = None
    if cat_cols:
        cat_train = X_train[:, 1:] if X_train.ndim > 1 else None
        categorical_encoders = {}
        for i, col in enumerate(cat_cols):
            if cat_cols_cfg[col] is None:  # string col: build DictEncoder from training data
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
    # Text is always column 0 (or the full array if 1D)
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


# ── Save Results ──────────────────────────────────────────────────────────────

def save_results(model_name: str, dataset_name: str, metrics: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dir = RESULTS_DIR / model_name
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{dataset_name}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {json_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

def run(config_path: str, dataset_name: str):
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    seed = cfg.get("seed", 42)
    all_datasets = cfg["datasets"]

    if dataset_name not in all_datasets:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(all_datasets)}")

    dataset_cfg = all_datasets[dataset_name]
    dataset_cfg["name"] = dataset_name

    print(f"\n{'='*50}")
    print(f"Model : {model_name} | Dataset : {dataset_name}")
    print(f"{'='*50}")

    X_train, y_train, X_val, y_val, X_test, y_test, value_encoder = load_data(dataset_cfg, seed)
    num_classes = len(np.unique(y_train))

    tokenizer = build_tokenizer(cfg["tokenizer"], X_train)

    cat_cols_cfg = dataset_cfg.get("categorical_cols", {})

    m = cfg["model"]
    model_config = ModelConfig(
        embedding_dim=m["embedding_dim"],
        num_classes=num_classes,
        aggregation_method=m.get("aggregation_method", "mean"),
        # categorical_vocabulary_sizes is derived from value_encoder when provided
        categorical_vocabulary_sizes=value_encoder.vocabulary_sizes if value_encoder else None,
        attention_config=m.get("attention_config"),
        n_heads_label_attention=m.get("n_heads_label_attention"),
    )
    clf = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config, value_encoder=value_encoder)

    t = cfg["training"]
    training_config = TrainingConfig(
        num_epochs=t["num_epochs"],
        batch_size=t["batch_size"],
        lr=t["lr"],
        patience_early_stopping=t.get("patience_early_stopping", 3),
        num_workers=t.get("num_workers", 0),
        raw_labels=False,                                    # labels are already integer-encoded by _make_y
        raw_categorical_inputs=value_encoder is not None,   # True only if there are categorical cols to encode
        save_path=str(RESULTS_DIR / "models" / model_name / dataset_name),
    )

    t0 = time.time()
    clf.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)
    train_time = round(time.time() - t0, 1)

    # Predict in batches to avoid OOM (predicting all test examples at once can exceed GPU memory).
    # predict_batch_size can be reduced per dataset in the config (e.g. for long texts like 20newsgroups).
    predict_batch_size = dataset_cfg.get("predict_batch_size", 512)
    all_preds = []
    for i in range(0, len(X_test), predict_batch_size):
        batch = X_test[i:i + predict_batch_size]
        result = clf.predict(batch, raw_categorical_inputs=value_encoder is not None)
        pred = result["prediction"]
        pred = pred.squeeze(dim=-1).numpy() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze(axis=-1)
        all_preds.append(pred)
    preds = np.concatenate(all_preds)

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tokenizer": cfg["tokenizer"]["type"],
        "embedding_dim": m["embedding_dim"],
        "aggregation_method": m.get("aggregation_method", "mean"),
        "attention_config": m.get("attention_config"),
        "categorical_cols": list(cat_cols_cfg.keys()) if cat_cols_cfg else [],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "train_time_s": train_time,
        "timestamp": datetime.now().isoformat(),
        "classification_report": classification_report(
            y_test, preds,
            target_names=dataset_cfg.get("class_names"),
            output_dict=True,
        ),
    }

    print(f"Test Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Test F1 macro : {metrics['test_f1_macro']:.4f}")
    print(f"Train time    : {train_time}s")

    save_results(model_name, dataset_name, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Chemin vers le fichier YAML du modèle")
    parser.add_argument("--dataset", required=True, help="Nom du dataset à utiliser")
    args = parser.parse_args()
    run(args.config, args.dataset)
