# Interpretable Text Classification at INSEE

Lightweight, explainable text classification architectures, benchmarked across three datasets of increasing difficulty (Amazon Reviews, CLINC150, NAF) and applied to INSEE's real-world use case: automatic coding of business creation declarations against the French economic activity nomenclature (**NAF 2025** / NACE Rev. 2.1).

This repository contains both the **experiment code** (Hydra + PyTorch Lightning + MLflow training pipelines, Captum/label-attention explainability tooling) and the **report** itself, published as a static [Quarto](https://quarto.org) website.

> Full write-up, figures, and discussion: see [The report](#the-report) (build/preview it locally, see below).

## Table of contents

- [Context](#context)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Architectures](#architectures)
- [Explainability](#explainability)
- [Getting started](#getting-started)
- [Training a model](#training-a-model)
- [Generating explainability artifacts](#generating-explainability-artifacts)
- [Experiment tracking and cached results](#experiment-tracking-and-cached-results)
- [The report](#the-report)
- [Training at scale (Argo Workflows)](#training-at-scale-argo-workflows)
- [Configuration reference](#configuration-reference)

## Context

At INSEE, when a company is founded, its activity is described in free text by the declarant and must be assigned a code from the **NAF** (*Nomenclature des Activités Françaises*), a hierarchical classification of economic activities aligned with NACE Rev. 2.1. A production model already automates part of this coding process — a single-level FastText classifier predicting the finest-grained code directly.

This project asks two questions:

1. Can **lightweight neural architectures** (FastText-style mean pooling, small transformer encoders, with or without a label-attention aggregation step) match or improve on that baseline, at a fraction of the cost of large language models — and how do they behave as hyperparameters, vocabulary size, and training-data volume change?
2. Since a bare prediction isn't enough for a human to act on with confidence in an ambiguous case, **can these models be explained** — which words drove a prediction, is that explanation faithful (does removing the highlighted words actually change the prediction), and does it hold up consistently across NAF's five nested hierarchy levels (section → division → group → class → sub-class)?

All models are built on top of [**`torchTextClassifiers`**](https://github.com/InseeFrLab/torchTextClassifiers), a PyTorch/Lightning package developed at INSEE providing a unified interface (tokenizers, model configs, training loop, Captum/label-attention explain hooks) across all the architectures compared here.

## Repository layout

```
.
├── src/                     # Training & explainability entry points (see below)
│   ├── train.py             #   single-level training  (python -m src.train)
│   ├── explain.py           #   single-level explainability (python -m src.explain)
│   ├── utils.py             #   shared helpers (accelerator resolution, log suppression)
│   └── multilevel/          #   multi-level NAF variant
│       ├── train.py         #     python -m src.multilevel.train
│       ├── explain.py       #     python -m src.multilevel.explain
│       ├── naf_data.py      #     NAF hierarchy constants + parquet loading
│       ├── naf_model.py     #     shared encoder + 5 independent per-level heads
│       └── __init__.py      #     MultiLevelTextClassificationModel, MultiLevelCrossEntropyLoss
├── conf/                    # Hydra configuration groups (see Configuration reference)
│   ├── entrypoint/          #   one file per script above (train / train_multilevel / explain / explain_multilevel)
│   ├── dataset/             #   one file per dataset
│   ├── model/, tokenizer/, training/
├── pages/                   # Quarto report pages (one .qmd per topic)
│   ├── amazon.qmd, clinc150.qmd, naf.qmd, naf_single.qmd, naf_multilevel.qmd, concepts.qmd
│   └── slides/               #   slide decks, one subfolder per presentation
├── index.qmd                # Report home page (introduction, study setup)
├── _quarto.yml               # Quarto project/website configuration
├── argo/                    # Argo Workflows manifests for cluster-scale training sweeps
├── assets/                  # Static assets used by the report (logos, precomputed JSON lookups)
├── results/                 # Local scratch space for predictions/explainability artifacts (gitignored)
├── runs_csvs/               # Local cache of MLflow run tables, mirrored from MinIO (gitignored)
├── .github/workflows/pages.yml  # CI: renders the Quarto site and deploys it to GitHub Pages
├── pyproject.toml / uv.lock # Python dependencies, managed with uv
```

## Datasets

Three datasets of increasing class count and difficulty, run in this order to progressively scale up towards the target application:

| Dataset | Classes | Train | Val | Test | Role |
|---|---|---|---|---|---|
| **Amazon Reviews** (English MARC split) | 5 | 195,000 | 5,000 | 5,000 | Warm-up: few classes, ordinal sentiment |
| **CLINC150** | 150 (+1 OOS variant) | 15,000 | 3,000 | 4,500 | Intermediate: many classes, short utterances |
| **NAF 2025** | ~700 (sub-class level) | ~1.2M | 50,000 | 50,000 | Target application: French business-activity coding |

`conf/dataset/` also ships configs for a handful of standard NLP classification benchmarks (AG News, SST-2, IMDB, 20 Newsgroups, Subj, Banking77) used only as lightweight smoke tests for the training pipeline during development — they are not part of the internship's reported results.

NAF itself is a **5-level nested hierarchy** (section → division → group → class → sub-class, from 21 down to ~750 official codes); see `pages/naf.qmd` for the full interactive breakdown and `src/multilevel/naf_data.py:NACE_LEVELS` for the levels as used in code.

## Architectures

Four architectures, combining two encoder families with two sequence-aggregation strategies:

- **FastText** — token embeddings averaged (mean pooling), fed directly to a classification head. No attention, minimal parameter count.
- **FastText + Label Attention** — same backbone, but the mean pool is replaced by a cross-attention step where one learned query vector per class attends over the token embeddings.
- **FATE** (*FastText Attentive Transformer Encoder*) — transformer encoder blocks stacked on top of the token embeddings before mean pooling.
- **FATE + Label Attention** — same transformer backbone, with label attention as the aggregation step instead of mean pooling.

`n_layers=0` (in `conf/model/default.yaml`) selects FastText; `n_layers>0` selects FATE. `n_heads_label_attention` (null by default) switches the aggregation method.

**Multi-level NAF** (`src/multilevel/`) extends this to predict all five NAF levels at once from a single model: one shared token encoder feeds five *independent* per-level classification heads (`src/multilevel/naf_model.py`), trained jointly with `MultiLevelCrossEntropyLoss`, which weights each level's contribution to the loss by its number of classes (`src/multilevel/__init__.py`).

## Explainability

Two complementary explanation mechanisms, both wired up in `src/explain.py` / `src/multilevel/explain.py`:

- **Captum Layer Integrated Gradients** — post-hoc, signed word-level attributions, available for every architecture.
- **Label Attention's own weights** — for models trained with label attention, the raw cross-attention matrix doubles as a built-in, architecture-native explanation.

Both are cross-checked with a **faithfulness (comprehensiveness) test**: progressively deleting the most-attributed words and tracking how fast the model's confidence in its own prediction actually drops, compared against deleting the same number of random words. See the `pages/concepts.qmd` and `pages/naf_multilevel.qmd` report pages for the full methodology and results.

## Getting started

**Requirements**: Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs all dependencies (declared in `pyproject.toml`, pinned in `uv.lock`), including PyTorch (CUDA build on Linux/Windows) and `torchTextClassifiers` (pulled directly from its GitHub repo) — no separate virtualenv setup needed, `uv run ...` and Quarto's own `execute.python: .venv/bin/python` setting (in `_quarto.yml`) both point at the environment `uv sync` creates.

**Environment variables** (all optional — every script has a working default):

| Variable | Used by | Default if unset |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `src.train`, `src.multilevel.train`, `src.explain`, `src.multilevel.explain` | Local `./mlruns` folder |
| `NAF_PARQUET_PATH` | `dataset=naf` (single-level) | Public MinIO URL (`conf/dataset/naf.yaml`) |
| `MLFLOW_S3_ENDPOINT_URL`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Only needed against a private/authenticated MLflow server (see `argo/*.yaml`) | — |

## Training a model

Every script is a [Hydra](https://hydra.cc) entry point — override any config value from the command line.

**Single-level** (`src/train.py`, config in `conf/entrypoint/train.yaml`):

```bash
uv run python -m src.train dataset=amazon
uv run python -m src.train dataset=clinc150 tokenizer=wordpiece model.embedding_dim=256
uv run python -m src.train dataset=naf model.n_layers=2 model.n_head=4 training.lr=0.001
uv run python -m src.train dataset=naf model.n_heads_label_attention=4 train_fraction=0.5
```

Config groups: `dataset` (one of `conf/dataset/*.yaml`), `tokenizer` (`ngram` | `wordpiece`), `model` (`conf/model/default.yaml` — override `embedding_dim`, `n_layers`, `n_head`, `n_heads_label_attention`, ...), `training` (`conf/training/default.yaml` — `lr`, `batch_size`, `num_epochs`, ...).

**Multi-level NAF** (`src/multilevel/train.py`, config in `conf/entrypoint/train_multilevel.yaml`):

```bash
uv run python -m src.multilevel.train
uv run python -m src.multilevel.train model.embedding_dim=256 model.n_heads_label_attention=4
```

Each run logs its hyperparameters, metrics, predictions, and full model checkpoint to MLflow (one experiment per dataset — `amazon`, `clinc150` / `clinc150_noos`, `naf`, `naf_multilevel`).

## Generating explainability artifacts

Once two comparable runs exist (Label Attention vs. mean pooling, same dataset/backbone), generate Captum/label-attention/faithfulness artifacts and log them back to those same MLflow runs:

```bash
# Single-level
uv run python -m src.explain run_id_labatt=<RUN_ID> run_id_pooling=<RUN_ID>

# Multi-level NAF (one run, all 5 levels)
uv run python -m src.multilevel.explain run_id=<RUN_ID>
```

See the module docstrings in `src/explain.py` / `src/multilevel/explain.py` for the exact artifact schema (`captum.npz`, `label_attn.npz`, `class_vectors.npz`, `self_attn.npz`, `faithfulness.npz`) — these are what the `pages/*.qmd` figures read back via `mlflow.MlflowClient().download_artifacts(...)`.

## Experiment tracking and cached results

Every training/explainability run is logged to **MLflow**. Rendering the report, however, does not require live MLflow access: each report page (`pages/amazon.qmd`, `clinc150.qmd`, `naf_single.qmd`, `naf_multilevel.qmd`) resolves its run table through a three-step fallback:

1. **Local cache** — `runs_csvs/<dataset>/*.csv`, if present.
2. **MinIO mirror** — a public, unauthenticated copy of those same tables at `https://minio.lab.sspcloud.fr/projet-text-classif/runs_csvs/...` (this is what CI uses — the GitHub Pages workflow has no MLflow credentials).
3. **Live MLflow query** (`mlflow.search_runs(...)`) — only reached if both of the above are unavailable, e.g. for a brand-new sweep not yet mirrored to MinIO.

`results/` is a separate, purely local and gitignored scratch space used by `src/explain.py` / `src/multilevel/explain.py` for prediction/explainability intermediates (`.npz`, `.parquet`, `.json`) — it is not part of this fallback chain.

## The report

The report is a multi-page [Quarto](https://quarto.org) website (`_quarto.yml`), automatically rendered and deployed to GitHub Pages on every push to `main` (`.github/workflows/pages.yml`).

To build or preview it locally:

```bash
uv run quarto render     # static build, output in _site/
uv run quarto preview    # live-reloading local preview
```

Always render through `uv run quarto ...` (not a bare `quarto` call) — this guarantees Quarto's Python code chunks execute inside the `uv`-managed environment, matching CI exactly.

## Training at scale (Argo Workflows)

`argo/` contains [Argo Workflows](https://argoproj.github.io/workflows/) manifests used to run the hyperparameter sweeps referenced in the report on a Kubernetes GPU cluster (SSP Cloud / Onyxia), each parallelizing several `uv run python -m src.train ...` invocations across GPUs:

| File | Sweep |
|---|---|
| `benchmark.yaml` | General architecture benchmark |
| `amazon_fate.yaml` | FATE hyperparameter sweep on Amazon Reviews |
| `data_efficiency.yaml` | Training-set-size ablation (Amazon) |
| `naf_fasttext.yaml` | FastText hyperparameter sweep on NAF |
| `naf_multilevel.yaml` | Multi-level NAF hyperparameter sweep |

These are cluster-specific (Onyxia service account, MLflow credentials via Kubernetes secrets) and not required to reproduce individual runs locally — see [Training a model](#training-a-model) for that.

## Configuration reference

All scripts share the same Hydra config groups under `conf/`:

| Group | Selects | Example |
|---|---|---|
| `entrypoint` | the script's own top-level settings (seed, experiment name, run IDs, ...) | `conf/entrypoint/train.yaml` |
| `dataset` | which dataset to load, and its loading quirks (HF path, column names, splits) | `dataset=naf` |
| `tokenizer` | `ngram` (FastText-style) or `wordpiece` (transformer-style) | `tokenizer=wordpiece` |
| `model` | architecture hyperparameters (embedding dim, transformer layers/heads, label attention) | `model.n_layers=4` |
| `training` | optimization hyperparameters (learning rate, batch size, epochs, patience) | `training.lr=0.001` |

Any value can be overridden from the CLI (`key=value` or `key.subkey=value`), Hydra's standard syntax.
