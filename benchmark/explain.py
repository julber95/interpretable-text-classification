"""
Generate Captum Integrated Gradients explainability artifacts for two models:
  - Label Attention  (run_id_labatt):  best Transformer + Label Attention
  - Mean Pooling     (run_id_pooling): same backbone, standard mean pooling

Both models are evaluated on the same test examples (same seed → same indices).

Artifacts logged to each MLflow run under explainability/:
  - captum.npz   word-level Layer Integrated Gradients scores, signed.
                 Keys: texts, words, word_attn, class_order, y_true, y_pred.
                 word_attn[i]: (n_classes, n_words) in confidence order.
                 class_order[i]: (n_classes,) mapping row k → actual class.

Usage:
    uv run python -m benchmark.explain \\
        run_id_labatt=<LABATT_ID> run_id_pooling=<POOLING_ID>
    uv run python -m benchmark.explain \\
        run_id_labatt=<LABATT_ID> run_id_pooling=<POOLING_ID> \\
        dataset=amazon n_captum=200
"""

import logging
import os
import tempfile
from pathlib import Path

import hydra
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from torchTextClassifiers import torchTextClassifiers
from benchmark.train import load_data

log = logging.getLogger(__name__)


def _load_model(run_id: str, tmp_dir: str) -> torchTextClassifiers:
    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "model", dst_path=tmp_dir)
    return torchTextClassifiers.load(local_path)


def _ragged(lst: list) -> np.ndarray:
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst):
        arr[i] = v
    return arr


def _captum_to_word(attributions: np.ndarray, word_ids: list) -> np.ndarray:
    """
    Aggregate token-level Captum attributions to word level by summing sub-tokens.
    Signed values are preserved: positive = supports class, negative = opposes it.

    Args:
        attributions: (n_classes, seq_len) float array — rows in confidence order
        word_ids:     list[int|None], one entry per token (None = special token)

    Returns:
        (n_classes, n_words) array — rows still in confidence order
    """
    ids     = np.array([x if x is not None else -1 for x in word_ids], dtype=int)
    valid   = ids >= 0
    attr_v  = attributions[:, valid]
    ids_v   = ids[valid]
    unique  = np.unique(ids_v)
    result  = np.zeros((attributions.shape[0], len(unique)), dtype=np.float32)
    for j, wid in enumerate(unique):
        result[:, j] = attr_v[:, ids_v == wid].sum(axis=1)
    return result


def _run_captum(
    clf: torchTextClassifiers,
    texts: list,
    y_sample: np.ndarray,
    n_captum: int,
    captum_batch_sz: int,
    label: str,
) -> dict:
    """
    Run Captum Layer Integrated Gradients on texts[:n_captum].

    Returns a dict with arrays ready to be saved via np.savez:
        texts, words, word_attn, class_order, y_true, y_pred
    """
    n_captum  = min(n_captum, len(texts))
    n_classes = clf.pytorch_model.num_classes

    all_words     = []
    all_word_attn = []
    all_class_ord = []
    all_preds     = []

    n_batches = (n_captum + captum_batch_sz - 1) // captum_batch_sz
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold yellow]Captum IG  [{label}]  "),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("", total=n_batches)
        for start in range(0, n_captum, captum_batch_sz):
            batch_texts = texts[start : start + captum_batch_sz]

            result = clf.predict(
                np.array(batch_texts),
                explain_with_captum=True,
                top_k=n_classes,
            )

            captum_attn  = result["captum_attributions"]   # (B, n_classes, seq_len) — signed
            word_ids_all = result["word_ids"]
            class_order  = result["prediction"]            # (B, n_classes) — confidence order
            offsets      = result["offset_mapping"]

            if isinstance(captum_attn, torch.Tensor):
                captum_attn = captum_attn.detach().cpu().numpy()
            if isinstance(class_order, torch.Tensor):
                class_order = class_order.cpu().numpy()

            for b, text in enumerate(batch_texts):
                # Reconstruct word strings from character offsets
                ids       = np.array([x if x is not None else -1 for x in word_ids_all[b]], dtype=int)
                valid_pos = np.where(ids >= 0)[0]
                word_strs: dict[int, str] = {}
                for pos in valid_pos:
                    wid = int(ids[pos])
                    if wid not in word_strs:
                        s, e = offsets[b][pos]
                        word_strs[wid] = text[s:e]

                words = [word_strs[wid] for wid in sorted(word_strs)]
                all_words.append(words)
                all_word_attn.append(_captum_to_word(captum_attn[b], word_ids_all[b]))
                all_class_ord.append(class_order[b])

            # Top-1 predicted class per example
            pred = np.array(class_order)
            all_preds.append(pred[:, 0])   # (B,) — most confident class
            progress.advance(task)

    return {
        "texts":       np.array(texts[:n_captum], dtype=object),
        "words":       _ragged(all_words),
        "word_attn":   _ragged(all_word_attn),
        "class_order": np.array(all_class_ord),
        "y_true":      y_sample[:n_captum],
        "y_pred":      np.concatenate(all_preds),
    }


@hydra.main(config_path="conf", config_name="explain", version_base=None)
def main(cfg: DictConfig) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    n_captum        = cfg.get("n_captum", 200)
    captum_batch_sz = cfg.get("captum_batch_size", 4)
    seed            = cfg.get("seed", 42)
    dataset_cfg     = OmegaConf.to_container(cfg.dataset, resolve=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    _, _, _, _, X_test, y_test, _ = load_data(dataset_cfg, seed)
    rng      = np.random.default_rng(seed)
    n_ex     = min(n_captum, len(X_test))
    idx      = rng.choice(len(X_test), size=n_ex, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    texts    = X_sample.tolist() if X_sample.ndim == 1 else X_sample[:, 0].tolist()

    # ── Run Captum IG for each model and log to its MLflow run ─────────────────
    models = {
        "labatt":  cfg.run_id_labatt,
        "pooling": cfg.run_id_pooling,
    }

    for label, run_id in models.items():
        log.info(f"Loading {label} model from MLflow run {run_id}")
        with tempfile.TemporaryDirectory() as tmp_model:
            clf = _load_model(run_id, tmp_model)

        data = _run_captum(clf, texts, y_sample, n_captum, captum_batch_sz, label)

        with mlflow.start_run(run_id=run_id):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "captum.npz"
                np.savez(str(p), **data)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info(f"Logged captum.npz to run {run_id}  ({n_captum} examples)")

    log.info("Done. Artifacts logged under explainability/captum.npz in each run.")


if __name__ == "__main__":
    main()
