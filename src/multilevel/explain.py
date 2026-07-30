"""
Generate explainability artifacts for the multi-level NAF model: one shared
token encoder feeding five *independent* classification heads (section,
division, group, class, sub-class — see src/multilevel/naf_model.py).

Why this can't just reuse src/explain.py
-----------------------------------------
torchTextClassifiers.predict(explain_with_captum=..., explain_with_label_attention=...)
only supports models whose forward() returns a single (batch, num_classes)
tensor. Multi-level models return a *list* of five tensors instead, and
predict() special-cases that: see the `if isinstance(model_output, list):`
branch in torchTextClassifiers.torchTextClassifiers.predict() — it computes
predictions/confidence and returns immediately, *before* the Captum /
label-attention code further down even runs. So passing
explain_with_captum=True to clf.predict() on this model silently does nothing
(no error, no attributions — the flag is just never read for list outputs).

This script therefore talks to the shared token encoder and each level's own
sentence embedder / classification head directly, one level at a time, instead
of going through clf.predict()'s explain path. Concretely, per level it:
  - wraps `model(...)` in a small nn.Module that returns only that level's
    logits, so Captum's LayerIntegratedGradients sees a single tensor to
    differentiate — mirroring exactly what torchTextClassifiers does
    internally for single-head models (LayerIntegratedGradients(pytorch_model, ...)),
    just aimed at one head instead of the whole model;
  - calls that level's `SentenceEmbedder(..., return_label_attention_matrix=True)`
    directly to recover its raw cross-attention weights, since
    NAFMultiLevelModel.forward() never requests or returns that matrix itself
    (unlike the single-head model, whose forward() has a dedicated
    return_label_attention_matrix kwarg it forwards through).

Everything that only ever touched model.token_embedder — self-attention hook
extraction, word masking, batching/progress helpers — has no dependency on
the single-vs-list forward() distinction and is imported unchanged from
src.explain. Self-attention itself is still gated on the shared encoder
actually having a transformer (build_model always sets attention_config=None
today — multi-level training is FastText-only, no FATE variant yet — so
self_attn.npz simply won't be produced now, the same way it wouldn't for a
FastText-only single-head run; the code is kept for when that changes).

Artifacts logged to the run under explainability/, one file per artifact
type, each holding one block of keys per level (suffixes _sec/_div/_grp/_cls/_sub):
  - captum.npz         per-level word-level Layer Integrated Gradients, signed.
                       Keys: texts (shared), words_<lvl>, word_attn_<lvl>,
                       class_order_<lvl>, y_true_<lvl>, y_pred_<lvl>.
  - label_attn.npz     per-level label-attention weights — only written if the
                       model was actually trained with n_heads_label_attention
                       set (mean-pooling runs have no such mechanism, at any level).
                       Keys per level: words_<lvl>, word_attn_<lvl>, head_attn_<lvl>,
                       head_word_attn_<lvl>, y_true_<lvl>, and (only when
                       label_attn_top_k is set) class_order_<lvl> — capping matters
                       here since classes sum across all 5 levels (~1700 for NAF),
                       which can produce an upload too large for MLflow (see
                       scripts/resume_multilevel_explain.py for the incident this fixes).
  - class_vectors.npz  per-level direction vectors (label_embeds or linear_weight,
                       whichever that level actually has). Keys: class_vectors_<lvl>,
                       kind_<lvl>.
  - self_attn.npz      only written if the shared token_embedder has a transformer
                       (currently never, see above).
  - faithfulness.npz   per-level guided-vs-random word-deletion test. Keys:
                       fractions_<lvl>, guided_probs_<lvl>, random_probs_<lvl>,
                       y_true_<lvl>, y_pred_<lvl>.

Usage:
    uv run python -m src.multilevel.explain run_id=<RUN_ID>
    uv run python -m src.multilevel.explain run_id=<RUN_ID> n_captum=200
"""

import logging
import os
import tempfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from torchTextClassifiers import torchTextClassifiers

from src.explain import (
    _progress,
    _ragged,
    _captum_to_word,
    _mask_words,
    _run_self_attn,
    _log_artifact_safely,
)
from src.multilevel.naf_data import NACE_LEVELS, load_naf

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

log = logging.getLogger(__name__)

LEVEL_NAMES = [name for name, _ in NACE_LEVELS]  # ["sec", "div", "grp", "cls", "sub"]


def _load_model(run_id: str, tmp_dir: str) -> torchTextClassifiers:
    """Identical to src.explain._load_model. Loading itself isn't the problem —
    torchTextClassifiers.from_model()/.load() already round-trip this custom
    architecture fine (it's pickled as a plain nn.Module); only predict()'s
    explain path is single-head-only."""
    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "model", dst_path=tmp_dir)
    return torchTextClassifiers.load(local_path)


class _LevelForward(nn.Module):
    """Makes the shared model look like a single-head model to Captum: forward()
    returns level_idx's (batch, num_classes) logits instead of the list of five."""

    def __init__(self, model: nn.Module, level_idx: int):
        super().__init__()
        self.model = model
        self.level_idx = level_idx

    def forward(self, input_ids, attention_mask, categorical_vars):
        return self.model(input_ids, attention_mask, categorical_vars)[self.level_idx]


def _run_captum_level(
    clf: torchTextClassifiers,
    level_idx: int,
    level_name: str,
    texts: list,
    y_level: np.ndarray,
    n_captum: int,
    captum_batch_sz: int,
    label: str,
    top_k: int | None = None,
) -> dict:
    """
    Same recipe as src.explain._run_captum (LayerIntegratedGradients over
    token_embedder.embedding_layer, one pass per top-k class, aggregated to
    word level), but driven by hand instead of clf.predict(explain_with_captum=True)
    since that path doesn't run at all for this model (see module docstring).

    Returns a dict of level-suffixed arrays (no "texts" key — that's identical
    across levels and saved once by the caller).
    """
    if not HAS_CAPTUM:
        raise ImportError("Captum is not installed — run 'uv add captum'.")

    device = clf.device
    model = clf.pytorch_model
    n_captum = min(n_captum, len(texts))
    n_classes = top_k or model.num_classes[level_idx]

    level_wrapper = _LevelForward(model, level_idx).to(device).eval()
    lig = LayerIntegratedGradients(level_wrapper, model.token_embedder.embedding_layer)

    all_words, all_word_attn, all_class_ord, all_preds = [], [], [], []

    n_batches = (n_captum + captum_batch_sz - 1) // captum_batch_sz
    with _progress(f"Captum IG [{label}/{level_name}]", "yellow") as progress:
        task = progress.add_task("", total=n_batches)
        for start in range(0, n_captum, captum_batch_sz):
            batch_texts = texts[start:start + captum_batch_sz]
            tok = clf.tokenizer.tokenize(
                batch_texts, return_offsets_mapping=True, return_word_ids=True
            )
            input_ids = tok.input_ids.to(device)
            attn_mask = tok.attention_mask.to(device)
            cat_vars  = torch.empty((input_ids.shape[0], 0), dtype=torch.float32, device=device)

            with torch.no_grad():
                scores = level_wrapper(input_ids, attn_mask, cat_vars).softmax(dim=-1)
            class_order = torch.topk(scores, k=n_classes, dim=-1).indices  # (B, n_classes)

            captum_attn = []
            for k in range(n_classes):
                attributions = lig.attribute(
                    (input_ids, attn_mask, cat_vars), target=class_order[:, k].to(device),
                )
                captum_attn.append(attributions.sum(dim=-1).detach().cpu())
            captum_attn = torch.stack(captum_attn, dim=1).numpy()  # (B, n_classes, seq_len)
            class_order_np = class_order.cpu().numpy()

            for b, text in enumerate(batch_texts):
                # Word spans grow to cover every sub-token sharing a word_id, not
                # just the first (see src.explain._run_captum for why).
                ids = np.array([x if x is not None else -1 for x in tok.word_ids[b]], dtype=int)
                valid_pos = np.where(ids >= 0)[0]
                word_spans: dict[int, list[int]] = {}
                for pos in valid_pos:
                    wid = int(ids[pos])
                    s, e = tok.offset_mapping[b][pos]
                    if wid not in word_spans:
                        word_spans[wid] = [s, e]
                    else:
                        word_spans[wid][1] = e
                word_strs = {wid: text[s:e] for wid, (s, e) in word_spans.items()}

                words = [word_strs[wid] for wid in sorted(word_strs)]
                all_words.append(words)
                all_word_attn.append(_captum_to_word(captum_attn[b], tok.word_ids[b]))
                all_class_ord.append(class_order_np[b])

            all_preds.append(class_order_np[:, 0])
            progress.advance(task)

    return {
        f"words_{level_name}":       _ragged(all_words),
        f"word_attn_{level_name}":   _ragged(all_word_attn),
        f"class_order_{level_name}": np.array(all_class_ord),
        f"y_true_{level_name}":      y_level[:n_captum],
        f"y_pred_{level_name}":      np.concatenate(all_preds),
    }


def _run_label_attention_level(
    clf: torchTextClassifiers,
    level_idx: int,
    level_name: str,
    texts: list,
    y_level: np.ndarray,
    n_ex: int,
    batch_sz: int,
    label: str,
    top_k: int | None = None,
) -> dict:
    """
    Same recipe as src.explain._run_label_attention, but calls this level's
    SentenceEmbedder directly with return_label_attention_matrix=True instead
    of clf.predict(explain_with_label_attention=True) — NAFMultiLevelModel.forward()
    never requests or surfaces that matrix on its own (see module docstring).

    top_k caps how many classes' attention rows get kept per example, ranked by
    this level's own predicted confidence (an extra classification-head forward
    pass just for the ranking — the attention forward pass itself is unaffected).
    None (default) keeps every class, in raw class-id order. Capping matters at
    this scale: label_attn.npz sums cumulative classes across all 5 levels
    (~1700 for NAF), which can produce an upload too large for MLflow — see
    scripts/resume_multilevel_explain.py for the incident this fixes.
    """
    model = clf.pytorch_model
    device = clf.device
    sentence_embedder = model.sentence_embedders[level_idx]
    if sentence_embedder.label_attention_config is None:
        raise RuntimeError(f"Level '{level_name}' was not trained with label attention.")
    classification_head = model.classification_heads[level_idx]

    n_ex = min(n_ex, len(texts))
    all_words, all_word_attn, all_head_attn, all_head_word_attn = [], [], [], []
    all_class_ord = []

    n_batches = (n_ex + batch_sz - 1) // batch_sz
    with _progress(f"Label attention [{label}/{level_name}]", "green") as progress:
        task = progress.add_task("", total=n_batches)
        for start in range(0, n_ex, batch_sz):
            batch_texts = texts[start:start + batch_sz]
            tok = clf.tokenizer.tokenize(
                batch_texts, return_offsets_mapping=True, return_word_ids=True
            )
            input_ids = tok.input_ids.to(device)
            attn_mask = tok.attention_mask.to(device)

            with torch.no_grad():
                x_token = model.token_embedder(input_ids, attn_mask)["token_embeddings"]
                sent_out = sentence_embedder(
                    token_embeddings=x_token, attention_mask=attn_mask,
                    return_label_attention_matrix=True,
                )
                attn_matrix = sent_out["label_attention_matrix"]  # (B, n_head, n_classes, seq_len)
                if top_k is not None:
                    logits = classification_head(sent_out["sentence_embedding"]).squeeze(-1)
                    k = min(top_k, logits.shape[-1])
                    class_order = torch.topk(logits.softmax(dim=-1), k=k, dim=-1).indices  # (B, k)
            attn_matrix = attn_matrix.detach().cpu().numpy()
            if top_k is not None:
                class_order_np = class_order.cpu().numpy()

            for b, text in enumerate(batch_texts):
                attn_b = attn_matrix[b]   # (n_head, n_classes, seq_len) — full
                if top_k is not None:
                    sel    = class_order_np[b]
                    attn_b = attn_b[:, sel, :]   # (n_head, k, seq_len) — capped
                    all_class_ord.append(sel)
                attn_mean = attn_b.mean(axis=0)

                ids      = np.array([x if x is not None else -1 for x in tok.word_ids[b]], dtype=int)
                valid    = ids >= 0
                ids_v    = ids[valid]
                attn_v   = attn_mean[:, valid]
                attn_bv  = attn_b[:, :, valid]
                unique_w = np.unique(ids_v)
                word_attn_b      = np.zeros((attn_mean.shape[0], len(unique_w)), dtype=np.float32)
                head_word_attn_b = np.zeros((attn_b.shape[0], attn_mean.shape[0], len(unique_w)), dtype=np.float32)
                for j, wid in enumerate(unique_w):
                    word_attn_b[:, j] = attn_v[:, ids_v == wid].sum(axis=1)
                    head_word_attn_b[:, :, j] = attn_bv[:, :, ids_v == wid].sum(axis=2)

                word_spans: dict[int, list[int]] = {}
                for pos, wid in enumerate(ids):
                    if wid < 0:
                        continue
                    s, e = tok.offset_mapping[b][pos]
                    if wid not in word_spans:
                        word_spans[wid] = [s, e]
                    else:
                        word_spans[wid][1] = e
                word_strs = {wid: text[s:e] for wid, (s, e) in word_spans.items()}

                all_words.append([word_strs[wid] for wid in unique_w])
                all_word_attn.append(word_attn_b)
                all_head_attn.append(attn_b)
                all_head_word_attn.append(head_word_attn_b)

            progress.advance(task)

    out = {
        f"words_{level_name}":          _ragged(all_words),
        f"word_attn_{level_name}":      _ragged(all_word_attn),
        f"head_attn_{level_name}":      _ragged(all_head_attn),
        f"head_word_attn_{level_name}": _ragged(all_head_word_attn),
        f"y_true_{level_name}":         y_level[:n_ex],
    }
    if top_k is not None:
        out[f"class_order_{level_name}"] = np.array(all_class_ord)
    return out


def _predict_target_probs_level(
    clf: torchTextClassifiers,
    level_idx: int,
    texts: list,
    target_class: np.ndarray,
    batch_sz: int,
) -> np.ndarray:
    """Like src.explain._predict_target_probs, but reads outputs[level_idx]
    from the list the shared model returns instead of a single logits tensor."""
    model  = clf.pytorch_model
    device = clf.device
    probs_out = np.zeros(len(texts), dtype=np.float32)

    for start in range(0, len(texts), batch_sz):
        batch = texts[start:start + batch_sz]
        tok   = clf.tokenizer.tokenize(batch)
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.to(device)
        cat_vars  = torch.empty((input_ids.shape[0], 0), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(input_ids, attn_mask, cat_vars)[level_idx]
            probs  = logits.softmax(dim=-1).cpu().numpy()

        for b in range(len(batch)):
            probs_out[start + b] = probs[b, target_class[start + b]]

    return probs_out


def _run_faithfulness_level(
    clf: torchTextClassifiers,
    level_idx: int,
    level_name: str,
    captum_level_data: dict,
    n_faithfulness: int,
    fractions: list,
    n_random: int,
    batch_sz: int,
    seed: int,
    label: str,
) -> dict:
    """Same recipe as src.explain._run_faithfulness (guided-by-Captum vs. random
    word deletion), rewritten against this level's own word_attn/class_order
    and _predict_target_probs_level instead of the single-head clf.predict()."""
    words_list   = [list(w) for w in captum_level_data[f"words_{level_name}"][:n_faithfulness]]
    class_order  = captum_level_data[f"class_order_{level_name}"][:n_faithfulness]
    target_class = class_order[:, 0].astype(int)
    n_faithfulness = len(words_list)

    guided_order = [
        np.argsort(-np.abs(captum_level_data[f"word_attn_{level_name}"][i][0]))
        for i in range(n_faithfulness)
    ]

    rng = np.random.default_rng(seed)
    n_levels     = len(fractions)
    guided_probs = np.zeros((n_faithfulness, n_levels), dtype=np.float32)
    random_probs = np.zeros((n_faithfulness, n_levels), dtype=np.float32)

    n_passes = n_levels * (1 + n_random)
    with _progress(f"Faithfulness [{label}/{level_name}]", "cyan", unit="passes") as progress:
        task = progress.add_task("", total=n_passes)

        for li, frac in enumerate(fractions):
            guided_texts = []
            for i in range(n_faithfulness):
                n_drop   = int(round(frac * len(words_list[i])))
                drop_idx = set(guided_order[i][:n_drop].tolist())
                guided_texts.append(_mask_words(words_list[i], drop_idx))
            guided_probs[:, li] = _predict_target_probs_level(
                clf, level_idx, guided_texts, target_class, batch_sz
            )
            progress.advance(task)

            acc = np.zeros(n_faithfulness, dtype=np.float32)
            for _ in range(n_random):
                random_texts = []
                for i in range(n_faithfulness):
                    n_words  = len(words_list[i])
                    n_drop   = int(round(frac * n_words))
                    drop_idx = set(rng.choice(n_words, size=n_drop, replace=False).tolist()) if n_drop else set()
                    random_texts.append(_mask_words(words_list[i], drop_idx))
                acc += _predict_target_probs_level(clf, level_idx, random_texts, target_class, batch_sz)
                progress.advance(task)
            random_probs[:, li] = acc / n_random

    return {
        f"fractions_{level_name}":    np.array(fractions, dtype=np.float32),
        f"guided_probs_{level_name}": guided_probs,
        f"random_probs_{level_name}": random_probs,
        f"y_true_{level_name}":       captum_level_data[f"y_true_{level_name}"][:n_faithfulness],
        f"y_pred_{level_name}":       captum_level_data[f"y_pred_{level_name}"][:n_faithfulness],
    }


@hydra.main(config_path="../../conf", config_name="entrypoint/explain_multilevel", version_base=None)
def main(cfg: DictConfig) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id              = cfg.run_id
    n_captum            = cfg.get("n_captum", 200)
    captum_batch_sz     = cfg.get("captum_batch_size", 4)
    captum_top_k        = cfg.get("captum_top_k", None)
    label_attn_top_k    = cfg.get("label_attn_top_k", None)
    batch_sz            = cfg.get("batch_size", 32)
    n_self_attn         = cfg.get("n_self_attn", 50)
    n_faithfulness      = cfg.get("n_faithfulness", 50)
    faithfulness_fracs  = list(cfg.get("faithfulness_fractions", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]))
    faithfulness_random = cfg.get("faithfulness_n_random", 3)
    seed                = cfg.get("seed", 42)
    dataset_cfg         = OmegaConf.to_container(cfg.dataset, resolve=True)

    # ── Load test data (same split logic src.multilevel.train uses) ────────────
    _, _, _, _, X_test, y_test, _ = load_naf(dataset_cfg, seed)
    rng      = np.random.default_rng(seed)
    n_ex     = min(n_captum, len(X_test))
    idx      = rng.choice(len(X_test), size=n_ex, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]  # (n_ex, 5)
    texts    = X_sample.tolist()

    log.info(f"Loading multi-level model from MLflow run {run_id}")
    with tempfile.TemporaryDirectory() as tmp_model:
        clf = _load_model(run_id, tmp_model)
    model = clf.pytorch_model

    has_label_attn  = model.sentence_embedders[0].label_attention_config is not None
    has_transformer = (
        model.token_embedder is not None and hasattr(model.token_embedder, "transformer")
    )
    log.info(f"label_attention={has_label_attn}  transformer={has_transformer}")

    artifacts: dict[str, dict] = {}

    # 1. Captum, per level — shared "texts" plus level-suffixed arrays.
    captum_data = {"texts": np.array(texts[:n_captum], dtype=object)}
    for level_idx, level_name in enumerate(LEVEL_NAMES):
        captum_data.update(_run_captum_level(
            clf, level_idx, level_name, texts, y_sample[:, level_idx],
            n_captum, captum_batch_sz, "multilevel", captum_top_k,
        ))
    artifacts["captum.npz"] = captum_data

    # 2. Label attention, per level — only if the model actually has the mechanism
    #    (it's a single global switch: either every level has it, or none do).
    if has_label_attn:
        label_attn_data = {}
        for level_idx, level_name in enumerate(LEVEL_NAMES):
            label_attn_data.update(_run_label_attention_level(
                clf, level_idx, level_name, texts, y_sample[:, level_idx],
                n_captum, batch_sz, "multilevel", label_attn_top_k,
            ))
        artifacts["label_attn.npz"] = label_attn_data

    # 3. Per-level class direction vectors — label_embeds if that level has label
    #    attention, otherwise the linear classification head's own weight rows.
    class_vectors_data = {}
    for level_idx, level_name in enumerate(LEVEL_NAMES):
        sentence_embedder = model.sentence_embedders[level_idx]
        if sentence_embedder.label_attention_config is not None:
            vectors = sentence_embedder.label_attention_module.label_embeds.weight
            kind = "label_embeds"
        else:
            vectors = model.classification_heads[level_idx].net.weight
            kind = "linear_weight"
        class_vectors_data[f"class_vectors_{level_name}"] = vectors.detach().cpu().numpy()
        class_vectors_data[f"kind_{level_name}"] = np.array(kind)
    artifacts["class_vectors.npz"] = class_vectors_data

    # 4. Self-attention — shared backbone, so src.explain's extractor works
    #    unchanged; only produced if the shared encoder actually has a
    #    transformer (it doesn't today, see module docstring).
    if has_transformer:
        artifacts["self_attn.npz"] = _run_self_attn(
            clf, texts, y_sample[:, LEVEL_NAMES.index("sub")],
            captum_data["y_pred_sub"], n_self_attn, batch_sz, "multilevel",
        )

    # 5. Faithfulness, per level.
    faithfulness_data = {}
    for level_idx, level_name in enumerate(LEVEL_NAMES):
        faithfulness_data.update(_run_faithfulness_level(
            clf, level_idx, level_name, captum_data,
            n_faithfulness, faithfulness_fracs, faithfulness_random, batch_sz, seed, "multilevel",
        ))
    artifacts["faithfulness.npz"] = faithfulness_data

    # ── Log everything to this run ──────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as tmp:
            for fname, data in artifacts.items():
                p = Path(tmp) / fname
                np.savez(str(p), **data)
                _log_artifact_safely(p, fname, run_id)

    log.info("Done. Artifacts logged under explainability/ in the run.")


if __name__ == "__main__":
    main()
