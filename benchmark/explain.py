"""
Generate explainability artifacts for two models trained on the same data:
  - Label Attention  (run_id_labatt):  best Transformer + Label Attention
  - Mean Pooling     (run_id_pooling): same backbone, standard mean pooling

Both models are evaluated on the same test examples (same seed → same indices),
which makes every artifact below directly comparable across architectures.

Artifacts logged to each MLflow run under explainability/:
  - captum.npz         word-level Layer Integrated Gradients scores, signed (both models).
                       Keys: texts, words, word_attn, class_order, y_true, y_pred.
                       word_attn[i]: (n_classes, n_words) in confidence order.
                       class_order[i]: (n_classes,) mapping row k → actual class.
  - label_attn.npz     word-level label-attention weights (Label Attention model only —
                       Mean Pooling has no such mechanism).
                       Keys: texts, words, word_attn, head_attn, y_true.
                       word_attn[i]: (n_classes, n_words), averaged over heads, softmax-
                       normalised over words. head_attn[i]: (n_heads, n_classes, seq_len), raw.
  - class_vectors.npz  per-class direction vectors used at the classification step —
                       comparable across architectures despite different mechanisms:
                         * Label Attention → label_embeds: learned query vectors of the
                           cross-attention (n_classes, emb_dim)
                         * Mean Pooling    → linear_weight: classification-head weight
                           rows, each row = direction of a class in embedding space
                           (n_classes, emb_dim)
                       Saved under a common key `class_vectors`; `kind` documents which.
  - self_attn.npz      transformer self-attention matrices for a subsample, averaged over
                       heads, per layer (both models — they share the same backbone).
                       Keys: self_attn (ragged, (n_layers, L, L) per example), texts,
                       y_true, y_pred.
  - faithfulness.npz   comprehensiveness-style deletion test: probability mass retained
                       by the originally predicted class as the most-attributed words are
                       progressively removed (guided), vs. removing the same number of
                       random words (control), averaged over several random draws.
                       Keys: fractions, guided_probs, random_probs, y_true, y_pred.
                       guided_probs / random_probs: (n_faithfulness, n_fractions).

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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from torchTextClassifiers import torchTextClassifiers
from torchTextClassifiers.model.components.attention import apply_rotary_emb
from torchTextClassifiers.utilities.plot_explainability import map_attributions_to_word
from benchmark.train import load_data

log = logging.getLogger(__name__)


def _progress(description: str, color: str, unit: str = "batches") -> Progress:
    """Build a rich Progress bar with the styling shared by every explainability pass."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold {color}]{description}  "),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} " + unit),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


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
    with _progress(f"Captum IG  [{label}]", "yellow") as progress:
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


def _run_label_attention(
    clf: torchTextClassifiers,
    texts: list,
    y_sample: np.ndarray,
    n_ex: int,
    batch_sz: int,
    label: str,
) -> dict:
    """
    Extract word-level label-attention weights: one cross-attention query per class,
    so each class gets its own view of "which words mattered for this rating".

    Returns a dict with arrays ready to be saved via np.savez:
        texts, words, word_attn, head_attn, y_true
    word_attn[i]: (n_classes, n_words) — averaged over heads, softmax-normalised over words.
    head_attn[i]: (n_heads, n_classes, seq_len) — raw, per head, token level (head specialisation).
    """
    n_ex = min(n_ex, len(texts))

    all_words     = []
    all_word_attn = []
    all_head_attn = []

    n_batches = (n_ex + batch_sz - 1) // batch_sz
    with _progress(f"Label attention  [{label}]", "green") as progress:
        task = progress.add_task("", total=n_batches)
        for start in range(0, n_ex, batch_sz):
            batch_texts = texts[start : start + batch_sz]

            result = clf.predict(np.array(batch_texts), explain_with_label_attention=True)

            attn_matrix  = result["label_attention_attributions"]   # (B, n_heads, n_classes, seq_len)
            offsets      = result["offset_mapping"]
            word_ids_all = result["word_ids"]

            if isinstance(attn_matrix, torch.Tensor):
                attn_matrix = attn_matrix.detach().cpu().numpy()

            for b, text in enumerate(batch_texts):
                attn_b    = attn_matrix[b]          # (n_heads, n_classes, seq_len)
                attn_mean = attn_b.mean(axis=0)     # (n_classes, seq_len) — averaged over heads

                # Token → word mapping: sub-tokens belonging to the same word are aggregated
                words_b, word_attn_b = map_attributions_to_word(
                    attributions=torch.tensor(attn_mean),
                    text=text,
                    word_ids=word_ids_all[b],
                    offsets=offsets[b],
                )
                all_words.append(list(words_b.values()))
                all_word_attn.append(word_attn_b)   # (n_classes, n_words)
                all_head_attn.append(attn_b)        # (n_heads, n_classes, seq_len)

            progress.advance(task)

    return {
        "texts":     np.array(texts[:n_ex], dtype=object),
        "words":     _ragged(all_words),
        "word_attn": _ragged(all_word_attn),
        "head_attn": _ragged(all_head_attn),
        "y_true":    y_sample[:n_ex],
    }


def _extract_self_attn(clf: torchTextClassifiers, texts: list) -> list:
    """
    Extract transformer self-attention matrices for a batch of texts.

    Uses forward hooks on c_q / c_k of each SelfAttentionLayer to capture Q and K,
    then recomputes softmax(QK^T / sqrt(d)) with RoPE and QK-norm applied — the same
    recipe the model itself uses, so the recovered weights match the forward pass exactly.

    Returns a list of (n_layers, seq_len_i, seq_len_i) arrays — one per example,
    cropped to the real (non-padded) sequence length, averaged over heads.
    """
    model          = clf.pytorch_model
    token_embedder = model.token_embedder
    config         = token_embedder.attention_config
    n_layers       = len(token_embedder.transformer.h)
    n_head         = config.n_head
    n_kv_head      = config.n_kv_head
    head_dim       = config.n_embd // n_head

    # ── Tokenize ──────────────────────────────────────────────────────────────
    device       = clf.device
    tokenize_out = clf.tokenizer.tokenize(texts)
    input_ids    = tokenize_out.input_ids.to(device)
    attn_mask    = tokenize_out.attention_mask.to(device)
    B, T         = input_ids.shape

    # ── Register hooks on c_q / c_k of every transformer block ───────────────
    q_cache: dict[int, torch.Tensor] = {}
    k_cache: dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(token_embedder.transformer.h):
        hooks.append(block.attn.c_q.register_forward_hook(
            lambda m, inp, out, idx=i: q_cache.update({idx: out.detach().float()})
        ))
        hooks.append(block.attn.c_k.register_forward_hook(
            lambda m, inp, out, idx=i: k_cache.update({idx: out.detach().float()})
        ))

    with torch.no_grad():
        cat_vars = torch.empty((B, 0), dtype=torch.float32, device=device)
        model(input_ids, attn_mask, cat_vars)

    for h in hooks:
        h.remove()

    # ── Recompute attention scores from Q and K ───────────────────────────────
    layer_attn = []
    for i in range(n_layers):
        q = q_cache[i].view(B, T, n_head,    head_dim)  # (B, T, H, D) — before transpose
        k = k_cache[i].view(B, T, n_kv_head, head_dim)  # (B, T, Hkv, D)

        if config.positional_encoding:
            cos = token_embedder.cos[:, :T].float()
            sin = token_embedder.sin[:, :T].float()
            q   = apply_rotary_emb(q, cos, sin)   # expects (B, T, H, D), cos is (1, T, 1, D/2)
            k   = apply_rotary_emb(k, cos, sin)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, Hkv, T, D)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        if n_head != n_kv_head:
            k = k.repeat_interleave(n_head // n_kv_head, dim=1)  # GQA expand

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        scores   = scores.masked_fill(pad_mask, float("-inf"))
        attn     = torch.softmax(scores, dim=-1).mean(dim=1)   # (B, T, T) avg over heads

        layer_attn.append(attn.cpu().numpy())

    # ── Crop to real seq length per example ───────────────────────────────────
    real_lengths = attn_mask.sum(dim=1).tolist()
    result = []
    for b in range(B):
        L = int(real_lengths[b])
        result.append(np.stack([layer_attn[i][b, :L, :L] for i in range(n_layers)]))
        # shape per example: (n_layers, L, L)

    return result


def _run_self_attn(
    clf: torchTextClassifiers,
    texts: list,
    y_sample: np.ndarray,
    y_pred: np.ndarray,
    n_self_attn: int,
    batch_sz: int,
    label: str,
) -> dict:
    """Run _extract_self_attn over a subsample; returns a dict ready for np.savez."""
    n_self_attn = min(n_self_attn, len(texts))

    matrices  = []
    n_batches = (n_self_attn + batch_sz - 1) // batch_sz
    with _progress(f"Self-attention  [{label}]", "magenta") as progress:
        task = progress.add_task("", total=n_batches)
        for start in range(0, n_self_attn, batch_sz):
            matrices.extend(_extract_self_attn(clf, texts[start : min(start + batch_sz, n_self_attn)]))
            progress.advance(task)

    return {
        "self_attn": _ragged(matrices),
        "texts":     np.array(texts[:n_self_attn], dtype=object),
        "y_true":    y_sample[:n_self_attn],
        "y_pred":    y_pred[:n_self_attn],
    }


def _predict_target_probs(
    clf: torchTextClassifiers,
    texts: list,
    target_class: np.ndarray,
    batch_sz: int,
) -> np.ndarray:
    """
    Plain forward pass (no Captum, no gradients): for each text, return the softmax
    probability assigned to target_class[i]. Bypasses clf.predict's confidence rounding
    (2 decimals) — too coarse to track the small probability shifts a single masked
    word can cause.
    """
    model  = clf.pytorch_model
    device = clf.device
    probs_out = np.zeros(len(texts), dtype=np.float32)

    for start in range(0, len(texts), batch_sz):
        batch = texts[start : start + batch_sz]
        tok   = clf.tokenizer.tokenize(batch)
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.to(device)
        cat_vars  = torch.empty((input_ids.shape[0], 0), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(input_ids, attn_mask, cat_vars)
            probs  = logits.softmax(dim=-1).cpu().numpy()

        for b in range(len(batch)):
            probs_out[start + b] = probs[b, target_class[start + b]]

    return probs_out


def _mask_words(words: list, drop_idx: set) -> str:
    """
    Rebuild a text from the words that survive removal, joined by spaces.
    This loses original punctuation spacing, but is the standard ERASER-style proxy
    for word deletion — and both models see the exact same reconstruction artefacts,
    so the LabAtt vs Pooling comparison stays fair.
    """
    kept = [w for j, w in enumerate(words) if j not in drop_idx]
    return " ".join(kept) if kept else " "


def _run_faithfulness(
    clf: torchTextClassifiers,
    captum_data: dict,
    n_faithfulness: int,
    fractions: list,
    n_random: int,
    batch_sz: int,
    seed: int,
    label: str,
) -> dict:
    """
    Comprehensiveness-style faithfulness test, reusing the Captum results: progressively
    remove the words with the highest |IG score| for the predicted class — word_attn row 0,
    since class_order (and therefore word_attn's rows) is sorted by confidence — and track
    how much probability mass the originally predicted class retains.

    A random-removal control of the same size, averaged over n_random draws, isolates the
    effect of *which* words are removed from the effect of removing *any* words: faithful
    explanations should make the guided curve drop faster than the random one.

    Returns a dict with arrays ready to be saved via np.savez:
        fractions, guided_probs, random_probs, y_true, y_pred
    guided_probs / random_probs: (n_faithfulness, n_fractions)
    """
    n_faithfulness = min(n_faithfulness, len(captum_data["texts"]))
    rng = np.random.default_rng(seed)

    words_list   = [list(w) for w in captum_data["words"][:n_faithfulness]]
    guided_order = [np.argsort(-np.abs(captum_data["word_attn"][i][0])) for i in range(n_faithfulness)]
    target_class = captum_data["class_order"][:n_faithfulness, 0].astype(int)

    n_levels     = len(fractions)
    guided_probs = np.zeros((n_faithfulness, n_levels), dtype=np.float32)
    random_probs = np.zeros((n_faithfulness, n_levels), dtype=np.float32)

    n_passes = n_levels * (1 + n_random)
    with _progress(f"Faithfulness  [{label}]", "cyan", unit="passes") as progress:
        task = progress.add_task("", total=n_passes)

        for li, frac in enumerate(fractions):
            # Guided removal: drop the top `frac` fraction of words by |attribution|
            guided_texts = []
            for i in range(n_faithfulness):
                n_drop   = int(round(frac * len(words_list[i])))
                drop_idx = set(guided_order[i][:n_drop].tolist())
                guided_texts.append(_mask_words(words_list[i], drop_idx))
            guided_probs[:, li] = _predict_target_probs(clf, guided_texts, target_class, batch_sz)
            progress.advance(task)

            # Random-removal control: same number of words dropped, averaged over n_random draws
            acc = np.zeros(n_faithfulness, dtype=np.float32)
            for _ in range(n_random):
                random_texts = []
                for i in range(n_faithfulness):
                    n_words  = len(words_list[i])
                    n_drop   = int(round(frac * n_words))
                    drop_idx = set(rng.choice(n_words, size=n_drop, replace=False).tolist()) if n_drop else set()
                    random_texts.append(_mask_words(words_list[i], drop_idx))
                acc += _predict_target_probs(clf, random_texts, target_class, batch_sz)
                progress.advance(task)
            random_probs[:, li] = acc / n_random

    return {
        "fractions":    np.array(fractions, dtype=np.float32),
        "guided_probs": guided_probs,
        "random_probs": random_probs,
        "y_true":       captum_data["y_true"][:n_faithfulness],
        "y_pred":       captum_data["y_pred"][:n_faithfulness],
    }


@hydra.main(config_path="conf", config_name="explain", version_base=None)
def main(cfg: DictConfig) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    n_captum            = cfg.get("n_captum", 200)
    captum_batch_sz     = cfg.get("captum_batch_size", 4)
    batch_sz            = cfg.get("batch_size", 32)
    n_self_attn         = cfg.get("n_self_attn", 50)
    n_faithfulness      = cfg.get("n_faithfulness", 50)
    faithfulness_fracs  = list(cfg.get("faithfulness_fractions", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]))
    faithfulness_random = cfg.get("faithfulness_n_random", 3)
    seed                = cfg.get("seed", 42)
    dataset_cfg         = OmegaConf.to_container(cfg.dataset, resolve=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    _, _, _, _, X_test, y_test, _ = load_data(dataset_cfg, seed)
    rng      = np.random.default_rng(seed)
    n_ex     = min(n_captum, len(X_test))
    idx      = rng.choice(len(X_test), size=n_ex, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    texts    = X_sample.tolist() if X_sample.ndim == 1 else X_sample[:, 0].tolist()

    # ── Run explainability passes for each model and log to its MLflow run ─────
    models = {
        "labatt":  cfg.run_id_labatt,
        "pooling": cfg.run_id_pooling,
    }

    for label, run_id in models.items():
        log.info(f"Loading {label} model from MLflow run {run_id}")
        with tempfile.TemporaryDirectory() as tmp_model:
            clf = _load_model(run_id, tmp_model)
        model = clf.pytorch_model

        has_label_attn = (
            hasattr(model, "sentence_embedder")
            and hasattr(model.sentence_embedder, "label_attention_module")
            and model.sentence_embedder.label_attention_module is not None
        )
        has_transformer = (
            model.token_embedder is not None
            and hasattr(model.token_embedder, "transformer")
        )
        log.info(f"[{label}] label_attention={has_label_attn}  transformer={has_transformer}")

        artifacts: dict[str, dict] = {}

        # 1. Captum Layer Integrated Gradients — signed, both models, directly comparable
        captum_data = _run_captum(clf, texts, y_sample, n_captum, captum_batch_sz, label)
        artifacts["captum.npz"] = captum_data

        # 2. Label-attention internals — only the Label Attention model has this mechanism
        if has_label_attn:
            artifacts["label_attn.npz"] = _run_label_attention(
                clf, texts, y_sample, n_captum, batch_sz, label
            )

        # 3. Per-class direction vectors, saved under a common key so the report can
        #    compare them architecture-agnostically (see module docstring for the mapping)
        if has_label_attn:
            vectors = model.sentence_embedder.label_attention_module.label_embeds.weight
            kind    = "label_embeds"
        else:
            vectors = model.classification_head.net.weight
            kind    = "linear_weight"
        artifacts["class_vectors.npz"] = {
            "class_vectors": vectors.detach().cpu().numpy(),
            "kind":          np.array(kind),
        }

        # 4. Transformer self-attention evolution — both models share the same backbone
        if has_transformer:
            artifacts["self_attn.npz"] = _run_self_attn(
                clf, texts, y_sample, captum_data["y_pred"], n_self_attn, batch_sz, label
            )

        # 5. Faithfulness: guided vs. random word-removal curves (both models)
        artifacts["faithfulness.npz"] = _run_faithfulness(
            clf, captum_data, n_faithfulness, faithfulness_fracs,
            faithfulness_random, batch_sz, seed, label,
        )

        # ── Log everything to this model's MLflow run ───────────────────────────
        with mlflow.start_run(run_id=run_id):
            with tempfile.TemporaryDirectory() as tmp:
                for fname, data in artifacts.items():
                    p = Path(tmp) / fname
                    np.savez(str(p), **data)
                    mlflow.log_artifact(str(p), artifact_path="explainability")
                    log.info(f"Logged {fname} to run {run_id}")

    log.info("Done. Artifacts logged under explainability/ in each run.")


if __name__ == "__main__":
    main()
