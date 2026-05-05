"""
Randomized hyperparameter search for fasttext on AG News.

How to use:
    uv run grid_search.py configs/fasttext.yaml

Results are saved as JSON under benchmark/results/<model_name>/grid_search.json.
"""

import argparse
import copy
import json
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

from run import load_data, build_tokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"

# ── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "embedding_dim": [64, 128, 256, 512],
    "lr":            [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "batch_size":    [32, 64, 128],
    "num_tokens":    [50000, 100000, 200000],
    "ngram_range":   [(2, 4), (3, 6), (2, 6)],  # (min_n, max_n)
}

N_RUNS     = 10
DATASET    = "ag_news"  # representative dataset for tuning
NUM_EPOCHS = 20


# ── Main ──────────────────────────────────────────────────────────────────────

def random_search(config_path: str, seed: int = 42):
    random.seed(seed)

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name  = cfg["model_name"]
    data_seed   = cfg.get("seed", 42)
    dataset_cfg = copy.deepcopy(cfg["datasets"][DATASET])

    print(f"\nRandom search — model: {model_name} | dataset: {DATASET} | runs: {N_RUNS}\n")

    X_train, y_train, X_val, y_val, X_test, y_test, value_encoder = load_data(dataset_cfg, data_seed)
    num_classes = len(np.unique(y_train))

    results = []
    best_f1, best_params = -1, None

    for i in range(N_RUNS):
        params = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        min_n, max_n = params.pop("ngram_range")
        print(f"[{i+1}/{N_RUNS}] embedding_dim={params['embedding_dim']} lr={params['lr']} "
              f"batch_size={params['batch_size']} num_tokens={params['num_tokens']} "
              f"ngram=({min_n},{max_n})", end=" ... ", flush=True)

        tok_cfg = copy.deepcopy(cfg["tokenizer"])
        tok_cfg["min_n"]      = min_n
        tok_cfg["max_n"]      = max_n
        tok_cfg["num_tokens"] = params["num_tokens"]
        tokenizer = build_tokenizer(tok_cfg, X_train)

        m = cfg["model"]
        model_config = ModelConfig(
            embedding_dim=params["embedding_dim"],
            num_classes=num_classes,
            aggregation_method=m.get("aggregation_method", "mean"),
            categorical_vocabulary_sizes=value_encoder.vocabulary_sizes if value_encoder else None,
            attention_config=m.get("attention_config"),
            n_heads_label_attention=m.get("n_heads_label_attention"),
        )
        clf = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config, value_encoder=value_encoder)

        training_config = TrainingConfig(
            num_epochs=NUM_EPOCHS,
            batch_size=params["batch_size"],
            lr=params["lr"],
            patience_early_stopping=3,
            num_workers=0,
            raw_labels=False,
            raw_categorical_inputs=value_encoder is not None,
            save_path=str(RESULTS_DIR / "models" / model_name / f"grid_{i}"),
        )

        clf.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)

        all_preds = []
        for j in range(0, len(X_val), 512):
            batch = X_val[j:j + 512]
            result = clf.predict(batch, raw_categorical_inputs=value_encoder is not None)
            pred = result["prediction"]
            pred = pred.squeeze(dim=-1).numpy() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze(axis=-1)
            all_preds.append(pred)
        preds = np.concatenate(all_preds)

        f1 = round(f1_score(y_val, preds, average="macro"), 4)
        print(f"val_f1_macro={f1}")

        run_params = {**params, "min_n": min_n, "max_n": max_n}
        results.append({"params": run_params, "val_f1_macro": f1})

        if f1 > best_f1:
            best_f1     = f1
            best_params = run_params

    print(f"\nBest params: {best_params} → val_f1_macro={best_f1}")

    out = {"best_params": best_params, "best_val_f1_macro": best_f1, "all_results": results}
    out_path = RESULTS_DIR / model_name / "grid_search.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the model YAML config file")
    args = parser.parse_args()
    random_search(args.config)
