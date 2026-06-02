"""
Generate explainability artifacts from a trained model.

Artifacts logged to MLflow run under explainability/:
  - word_attn.npz              per-example word-level attention averaged over heads
  - head_attn.npz              per-example raw per-head label attention
  - corpus_word_importance.npz top-K words per class aggregated over the corpus
  - label_embeddings.npz       label_embeds weight matrix + linear head weight
  - self_attn.npz              transformer self-attention matrices (seq_len × seq_len)
                               for a small subsample, averaged over heads, per layer
  - captum_attn.npz            Layer Integrated Gradients attributions per token/word
                               for all models; signed (positive = supports class,
                               negative = opposes class)

Usage:
    uv run python -m benchmark.explain run_id=<RUN_ID>
    uv run python -m benchmark.explain run_id=<RUN_ID> dataset=amazon n_examples=200
"""

import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import hydra
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
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


def _load_model(run_id: str, tmp_dir: str) -> torchTextClassifiers:
    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "model", dst_path=tmp_dir)
    return torchTextClassifiers.load(local_path)


def _ragged(lst: list) -> np.ndarray:
    """Build a 1-D object array from a list of arrays with potentially different shapes."""
    arr = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst):
        arr[i] = v
    return arr


def _captum_to_word(attributions: np.ndarray, word_ids: list) -> np.ndarray:
    """
    Aggregate token-level Captum attributions to word level by summing sub-tokens.

    Unlike map_attributions_to_word, NO softmax is applied — signed values are preserved
    so that positive = supports the class, negative = opposes it.

    Args:
        attributions: (n_classes, seq_len) float array
        word_ids:     list[int|None], one entry per token (None = special token)

    Returns:
        (n_classes, n_words) array — columns ordered by word_id (i.e. word order)
    """
    ids = np.array([x if x is not None else -1 for x in word_ids], dtype=int)
    valid   = ids >= 0
    attr_v  = attributions[:, valid]   # (n_classes, n_real_tokens)
    ids_v   = ids[valid]
    unique  = np.unique(ids_v)
    result  = np.zeros((attributions.shape[0], len(unique)), dtype=np.float32)
    for j, wid in enumerate(unique):
        result[:, j] = attr_v[:, ids_v == wid].sum(axis=1)
    return result                      # (n_classes, n_words)


def _extract_self_attn(clf, texts: list) -> list:
    """
    Extract transformer self-attention matrices for a batch of texts.

    Uses forward hooks on c_q / c_k of each SelfAttentionLayer to capture Q and K,
    then recomputes softmax(QK^T / sqrt(d)) with RoPE and QK-norm applied.

    Returns a list of (n_layers, seq_len_i, seq_len_i) arrays — one per example,
    cropped to the real (non-padded) sequence length, averaged over heads.
    """
    model         = clf.pytorch_model
    token_embedder = model.token_embedder
    config        = token_embedder.attention_config
    n_layers      = len(token_embedder.transformer.h)
    n_head        = config.n_head
    n_kv_head     = config.n_kv_head
    head_dim      = config.n_embd // n_head

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenize_out = clf.tokenizer.tokenize(texts)
    input_ids    = tokenize_out.input_ids
    attn_mask    = tokenize_out.attention_mask
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
        cat_vars = torch.empty((B, 0), dtype=torch.float32)
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


@hydra.main(config_path="conf", config_name="explain", version_base=None)
def main(cfg: DictConfig) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id   = cfg.run_id
    n_ex        = cfg.get("n_examples", 500)
    batch_sz    = cfg.get("batch_size", 32)
    top_k_w     = cfg.get("top_k_words", 50)
    n_self_attn = cfg.get("n_self_attn", 50)   # examples for seq_len×seq_len matrices
    n_captum    = cfg.get("n_captum", 200)     # examples for Captum IG (slow: ~N forward passes)
    seed        = cfg.get("seed", 42)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading model from MLflow run {run_id}")
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
    log.info(f"label_attention={has_label_attn}  transformer={has_transformer}")

    # ── Load test data ────────────────────────────────────────────────────────
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    _, _, _, _, X_test, y_test, _ = load_data(dataset_cfg, seed)

    n_ex = min(n_ex, len(X_test)) if n_ex is not None else len(X_test)
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(X_test), size=n_ex, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    texts = X_sample.tolist() if X_sample.ndim == 1 else X_sample[:, 0].tolist()

    # ── Pass 1: predictions + label attention (if available) ────────────────
    # For each example: prediction, confidence, and if label attention:
    #   - word_attn  (n_classes, n_words): attention weight per word per class,
    #                                      averaged over heads → "which word for which rating?"
    #   - head_attn  (n_heads, n_classes, seq_len): same but per head, not aggregated
    #                                      → "do heads specialise?"
    all_words     = []
    all_word_attn = []
    all_head_attn = []
    all_preds     = []
    all_conf      = []

    n_batches = (n_ex + batch_sz - 1) // batch_sz
    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total} batches"),
                  TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        task = progress.add_task(f"Predictions ({n_ex} examples)", total=n_batches)
        for start in range(0, n_ex, batch_sz):
            batch_texts = texts[start : start + batch_sz]

            result = clf.predict(
                np.array(batch_texts),
                explain_with_label_attention=has_label_attn,
            )

            if has_label_attn:
                # attn_matrix : (B, n_heads, n_classes, seq_len)
                attn_matrix  = result["label_attention_attributions"]
                offset_maps  = result["offset_mapping"]
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
                        offsets=offset_maps[b],
                    )
                    all_words.append(list(words_b.values()))
                    all_word_attn.append(word_attn_b)   # (n_classes, n_words)
                    all_head_attn.append(attn_b)        # (n_heads, n_classes, seq_len)

            pred = result["prediction"]
            conf = result["confidence"]
            pred = pred.squeeze(-1).numpy() if isinstance(pred, torch.Tensor) else np.array(pred).squeeze(-1)
            conf = conf.squeeze(-1).numpy() if isinstance(conf, torch.Tensor) else np.array(conf).squeeze(-1)
            all_preds.append(pred)
            all_conf.append(conf)
            progress.advance(task)

    y_pred     = np.concatenate(all_preds)
    confidence = np.concatenate(all_conf)

    # ── Corpus word importance (label attention only) ────────────────────────
    # For each class, accumulate attention weights of every word across all examples,
    # then keep the top-K. Result: words the model systematically associates with each rating.
    # Minimum 5 occurrences filter to exclude rare, non-representative words.
    if has_label_attn:
        n_classes = all_word_attn[0].shape[0]
        word_acc  = [defaultdict(list) for _ in range(n_classes)]
        for words_ex, word_attn_ex in zip(all_words, all_word_attn):
            for c in range(n_classes):
                for w_idx, word in enumerate(words_ex):
                    word_acc[c][word.lower()].append(float(word_attn_ex[c, w_idx]))
        corpus = {}
        for c in range(n_classes):
            word_means = {w: np.mean(v) for w, v in word_acc[c].items() if len(v) >= 5}
            top = sorted(word_means.items(), key=lambda x: -x[1])[:top_k_w]
            corpus[f"class_{c}_words"]  = np.array([w for w, _ in top], dtype=object)
            corpus[f"class_{c}_scores"] = np.array([s for _, s in top])

    # ── Label embeddings (label attention only) ──────────────────────────────
    # label_embeds  (n_classes, emb_dim): learned vector per class, used as queries
    #   in the cross-attention. PCA on these vectors reveals whether classes form
    #   an ordinal gradient in the embedding space.
    # linear_weight (1, emb_dim): shared projection from embedding to logit (less informative).
    if has_label_attn:
        label_embeds  = model.sentence_embedder.label_attention_module \
                             .label_embeds.weight.detach().cpu().numpy()
        linear_weight = model.classification_head.net.weight.detach().cpu().numpy()

    # ── Self-attention seq_len × seq_len (transformer only) ─────────────────
    # For n_self_attn examples: matrix (n_layers, L, L) where [layer, i, j] =
    # weight that token i assigns to token j in the given layer.
    # Shows how information flows between tokens and evolves across layers:
    # early layers tend to be local (neighbouring tokens), later layers capture long-range dependencies.
    if has_transformer:
        n_self_attn = min(n_self_attn, n_ex)
        log.info(f"Extracting self-attention matrices for {n_self_attn} examples")
        self_attn_matrices = []
        for start in range(0, n_self_attn, batch_sz):
            batch = texts[start : start + batch_sz]
            self_attn_matrices.extend(_extract_self_attn(clf, batch))

    # ── Pass 2: Captum Layer Integrated Gradients (all models) ──────────────
    # Post-hoc method: integrates the gradient of the prediction with respect to
    # the embedding along a path from the zero vector to the actual embedding.
    # Result: SIGNED score per word — positive = pushes toward this class,
    # negative = pushes against it. Works on FastText / Transformer / Label Attention
    # → enables cross-architecture comparison on the same examples.
    # Slower than label attention: requires ~50 forward/backward passes per example.
    n_captum    = min(n_captum, n_ex)
    n_classes   = clf.pytorch_model.num_classes
    all_captum_words     = []
    all_captum_word_attn = []
    all_captum_class_ord = []

    n_captum_batches = (n_captum + batch_sz - 1) // batch_sz
    with Progress(SpinnerColumn(), TextColumn("[bold yellow]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total} batches"),
                  TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        task = progress.add_task(f"Captum IG  ({n_captum} examples, top_k={n_classes})", total=n_captum_batches)
        for start in range(0, n_captum, batch_sz):
            batch_texts = texts[start : start + batch_sz]
            result = clf.predict(
                np.array(batch_texts),
                explain_with_captum=True,
                top_k=n_classes,   # attributions for all classes, not just the top-1
            )
            captum_attn  = result["captum_attributions"]   # (B, n_classes, seq_len) — signed
            word_ids_all = result["word_ids"]
            class_order  = result["prediction"]            # (B, n_classes) — classes sorted by confidence

            if isinstance(captum_attn, torch.Tensor):
                captum_attn = captum_attn.detach().cpu().numpy()
            if isinstance(class_order, torch.Tensor):
                class_order = class_order.numpy()

            offsets = result["offset_mapping"]
            for b, text in enumerate(batch_texts):
                # Reconstruct word strings from offset_mapping
                ids  = np.array([x if x is not None else -1 for x in word_ids_all[b]], dtype=int)
                valid_pos = np.where(ids >= 0)[0]
                word_strs: dict[int, str] = {}
                for pos in valid_pos:
                    wid = int(ids[pos])
                    if wid not in word_strs:
                        s, e = offsets[b][pos]
                        word_strs[wid] = text[s:e]

                all_captum_words.append([word_strs[wid] for wid in sorted(word_strs)])
                # Aggregate tokens → words by summing (no softmax: signed values are preserved)
                all_captum_word_attn.append(_captum_to_word(captum_attn[b], word_ids_all[b]))
                all_captum_class_ord.append(class_order[b])
            progress.advance(task)

    # ── Save artifacts to MLflow (resume the original training run) ─────────
    # Resume the existing run to keep all artifacts in one place.
    # Files are written to a temporary directory and uploaded; nothing is stored locally.
    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            if has_label_attn:
                # word_attn: attention weight per word per class, averaged over heads.
                # Ragged array: each example has a different number of words.
                # → "Which word mattered for which rating?" across n_ex examples.
                p = tmp / "word_attn.npz"
                np.savez(p, texts=np.array(texts, dtype=object),
                         words=_ragged(all_words),
                         word_attn=_ragged(all_word_attn),
                         y_true=y_sample, y_pred=y_pred, confidence=confidence)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info("Logged word_attn.npz")

                # head_attn: same as word_attn but per head, not aggregated.
                # → "Do the attention heads look at the same words or do they specialise?"
                p = tmp / "head_attn.npz"
                np.savez(p, head_attn=_ragged(all_head_attn),
                         texts=np.array(texts, dtype=object),
                         y_true=y_sample, y_pred=y_pred)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info("Logged head_attn.npz")

                # corpus_word_importance: top-K words per class aggregated over the full corpus.
                # → "Which words does the model systematically associate with each rating?"
                p = tmp / "corpus_word_importance.npz"
                np.savez(p, **corpus)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info(f"Logged corpus_word_importance.npz (top {top_k_w} words/class)")

                # label_embeddings: learned vectors per class (queries in the cross-attention).
                # PCA on label_embeds reveals whether classes form an ordinal gradient.
                # linear_weight (1, emb_dim): shared projection to logit.
                p = tmp / "label_embeddings.npz"
                np.savez(p, label_embeds=label_embeds, linear_weight=linear_weight)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info(f"Logged label_embeddings.npz — shape: {label_embeds.shape}")

            else:
                # linear_weights: classification head weights (n_classes, emb_dim).
                # Each row = direction of the class in the embedding space.
                # Transposed → comparable to label_embeds from the label attention model:
                # label attention is expected to produce more structured class directions.
                p = tmp / "linear_weights.npz"
                linear_weight = model.classification_head.net.weight.detach().cpu().numpy()
                np.savez(p, linear_weight=linear_weight)
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info(f"Logged linear_weights.npz — shape: {linear_weight.shape}")

            # captum_attn: signed attributions per word per class (all models).
            # Positive value = word pushes toward this class, negative = pushes against it.
            # class_order: classes sorted by descending confidence (not by class index).
            # → Cross-architecture comparison on the same examples.
            p = tmp / "captum_attn.npz"
            np.savez(p,
                     texts=np.array(texts[:n_captum], dtype=object),
                     words=_ragged(all_captum_words),
                     word_attn=_ragged(all_captum_word_attn),
                     class_order=np.array(all_captum_class_ord),
                     y_true=y_sample[:n_captum],
                     y_pred=y_pred[:n_captum])
            mlflow.log_artifact(str(p), artifact_path="explainability")
            log.info(f"Logged captum_attn.npz ({n_captum} examples)")

            if has_transformer:
                # self_attn: matrix (n_layers, L, L) per example.
                # [layer, i, j] = weight that token i assigns to token j in the given layer.
                # → Evolution of attention across transformer layers.
                p = tmp / "self_attn.npz"
                np.savez(p,
                         self_attn=np.array(self_attn_matrices, dtype=object),
                         texts=np.array(texts[:n_self_attn], dtype=object),
                         y_true=y_sample[:n_self_attn],
                         y_pred=y_pred[:n_self_attn])
                mlflow.log_artifact(str(p), artifact_path="explainability")
                log.info(f"Logged self_attn.npz ({n_self_attn} examples, (n_layers, L, L) per example)")

    log.info(f"Done. All artifacts logged to MLflow run {run_id} under explainability/")


if __name__ == "__main__":
    main()
