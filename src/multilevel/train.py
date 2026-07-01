"""
Multi-level NACE classification for NAF dataset.

Trains a shared token embedder with one classification head per NAF hierarchy
level using MultiLevelTextClassificationModel and MultiLevelCrossEntropyLoss.

Usage:
    uv run python -m src.multilevel.train
    uv run python -m src.multilevel.train model.embedding_dim=256 model.n_heads_label_attention=4
"""

import logging
import os
import time
import warnings
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from torchTextClassifiers import TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import WordPieceTokenizer

from src.multilevel import MultiLevelCrossEntropyLoss
from src.multilevel.naf_data import NACE_LEVELS, load_naf
from src.multilevel.naf_model import build_model

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _resolve_accelerator() -> str:
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


@hydra.main(config_path="../../conf", config_name="train_multilevel", version_base=None)
def main(cfg: DictConfig):
    accelerator = _resolve_accelerator()

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    tok_cfg     = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    m, t        = cfg.model, cfg.training

    emb_dim            = m.embedding_dim
    n_heads_label_attn = m.n_heads_label_attention
    vocab_size         = tok_cfg.get("vocab_size", 10000)

    print(f"\n{'='*50}")
    print("Dataset: NAF multi-level | Tokenizer: wordpiece")
    print(f"emb_dim={emb_dim} | label_attn={n_heads_label_attn} | vocab_size={vocab_size}")
    print(f"lr={t.lr} | batch_size={t.batch_size}")
    print(f"{'='*50}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get("experiment_name", "naf_multilevel"))

    train_fraction = cfg.get("train_fraction", 1.0)
    dataset_cfg["train_fraction"] = train_fraction

    X_train, y_train, X_val, y_val, X_test, y_test, num_classes_per_level = load_naf(
        dataset_cfg, cfg.seed
    )

    tokenizer = WordPieceTokenizer(vocab_size=vocab_size, output_dim=emb_dim)
    tokenizer.train(X_train)

    model = build_model(tokenizer, num_classes_per_level, emb_dim, n_heads_label_attn)
    clf   = torchTextClassifiers.from_model(tokenizer=tokenizer, pytorch_model=model)

    training_config = TrainingConfig(
        num_epochs=t.num_epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        patience_early_stopping=t.patience_early_stopping,
        num_workers=t.num_workers,
        accelerator=accelerator,
        raw_labels=False,
        loss=MultiLevelCrossEntropyLoss(num_classes=num_classes_per_level),
        save_path=str(RESULTS_DIR / "models" / "naf_multilevel"),
    )

    t0 = time.time()
    clf.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
              training_config=training_config)
    train_time = time.time() - t0

    model.eval()
    device = torch.device("cuda" if accelerator == "cuda" else "cpu")
    model.to(device)

    def batch_predict_multilevel(X, batch_size=512):
        all_preds = [[] for _ in NACE_LEVELS]
        for i in range(0, len(X), batch_size):
            batch_texts = X[i:i + batch_size].tolist()
            enc = tokenizer.tokenize(batch_texts)
            with torch.no_grad():
                logits_list = model(input_ids=enc.input_ids.to(device),
                                    attention_mask=enc.attention_mask.to(device))
            for lvl, logits in enumerate(logits_list):
                all_preds[lvl].append(logits.argmax(dim=-1).cpu().numpy())
        return [np.concatenate(p) for p in all_preds]

    preds_per_level = batch_predict_multilevel(X_test)

    with mlflow.start_run():
        mlflow.log_params({
            "embedding_dim":           emb_dim,
            "n_layers":                0,
            "vocab_size":              vocab_size,
            "n_heads_label_attention": n_heads_label_attn,
            "lr":                      t.lr,
            "batch_size":              t.batch_size,
            "num_epochs":              t.num_epochs,
            "patience_early_stopping": t.patience_early_stopping,
            "train_fraction":          train_fraction,
            "n_train":                 len(X_train),
            "n_val":                   len(X_val),
            "n_test":                  len(X_test),
        })

        for (name, _), n_cls, preds in zip(NACE_LEVELS, num_classes_per_level, preds_per_level):
            y_true = y_test[:, NACE_LEVELS.index((name, _))]
            acc    = round(accuracy_score(y_true, preds), 4)
            f1     = round(f1_score(y_true, preds, average="macro", zero_division=0), 4)
            mlflow.log_metrics({
                f"test_accuracy_{name}": acc,
                f"test_f1_macro_{name}": f1,
            })
            print(f"Level '{name}' ({n_cls} classes): acc={acc:.4f}  f1={f1:.4f}")

        mlflow.log_metric("train_time_s", round(train_time, 1))
        mlflow.log_metric("num_params",
                          sum(p.numel() for p in model.parameters() if p.requires_grad))

        preds_path = RESULTS_DIR / f"predictions_multilevel_{mlflow.active_run().info.run_id}.npz"
        np.savez(preds_path, y_test=y_test,
                 **{f"y_pred_{name}": p for (name, _), p in zip(NACE_LEVELS, preds_per_level)})
        mlflow.log_artifact(str(preds_path), artifact_path="predictions")
        preds_path.unlink()

    print(f"\nDone in {train_time:.0f}s")


if __name__ == "__main__":
    main()
