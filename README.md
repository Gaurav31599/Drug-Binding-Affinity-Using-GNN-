# Drug–Target Binding-Affinity Prediction (GNN vs. Classical ML)

A graph neural network for drug–target binding-affinity regression on the
**KIBA** benchmark, benchmarked head-to-head against an eight-model classical ML
sweep — under a **scaffold split**, with MLflow experiment tracking.

This is a **benchmarked ML exercise**, not a drug-discovery platform: it does not
identify real drug candidates. See [What this is *not*](#what-this-is-not).

---

## Why this exists (and what it replaces)

This is a ground-up rebuild of an earlier exploratory notebook whose "model" was
**circular**: 10 rows, a single feature, predicting a label derived directly
from that same feature — no real predictive task. This project replaces it with:

| Original problem | Fix here |
| --- | --- |
| Circular single Random Forest "predicting" a label derived from its only feature. | A genuine **8-model regression sweep on a real benchmark (KIBA)** that the GNN must beat. |
| Random train/test split (leaks structurally similar molecules). | **Scaffold split** — no drug scaffold appears in both train and test. |
| One ad-hoc model, no tracking, no metrics table. | Full **MLflow** nested-run tracking + a committed `results/metrics.csv`. |

---

## Repository layout

```
.
├── src/
│   ├── affinity_gnn/
│   │   ├── data/loader.py      # KIBA loader (+ toy slice for tests)
│   │   ├── splits.py           # Bemis-Murcko scaffold split
│   │   ├── features.py         # Morgan FP + AAC (baseline); graph + CNN (GNN)
│   │   ├── baseline_sweep.py   # Linear/Ridge/Lasso/ElasticNet/RF/GBM/XGB/SVR/KNN
│   │   ├── gnn_model.py        # GraphDTA-style GCN + protein CNN
│   │   └── evaluate.py         # RMSE / MAE / Pearson r / R²
│   └── train.py               # single entrypoint: sweep + GNN + MLflow logging
├── results/metrics.csv         # committed, reviewer-facing results table
├── tests/                      # pytest: split, featurization, metrics, GNN smoke
├── notebooks/                  # EDA only — not the pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Notes:
- `rdkit`, `torch`, and `torch-geometric` are the heavy dependencies. Install
  `torch` first per the [official selector](https://pytorch.org/get-started/locally/),
  then `torch-geometric`.

---

## The task

**Dataset:** **KIBA** (sourced without registration from the DeepDTA mirror) — a
standard drug–target affinity benchmark, so results are comparable to published
DeepDTA/GraphDTA numbers. (The KIBA score is an integrated affinity metric on a
unified scale, **not pKd** — we keep and label it as the KIBA score rather than
mislabel it.)

**Split:** **scaffold split** (Bemis-Murcko). A random split leaks structurally
similar molecules across train/test and inflates apparent performance; scaffold
splitting puts whole chemotypes on one side of the boundary, so the test set is
genuinely novel chemistry. A unit test asserts **zero scaffold overlap**.

**Baseline:** an 8-model regression sweep on Morgan fingerprints + amino-acid
composition, under identical folds — Linear, Ridge, Lasso, ElasticNet, Random
Forest, Gradient Boosting, XGBoost, SVR, KNN. This gives the GNN something real
to beat.

**GNN:** a GraphDTA-style network — GCN message passing over the molecular graph
(drug) + 1-D CNN over the protein sequence — regressing a **continuous** affinity
value.

**Metrics:** RMSE, MAE, Pearson r, R². This is **regression**, so classification
metrics (accuracy / precision / recall / F1 / ROC-AUC) do not apply and are
deliberately not reported.

```bash
python src/train.py                # full KIBA run (sweep + GNN), logs to MLflow
python src/train.py --smoke        # tiny toy slice, 1 GNN epoch (quick sanity)
python src/train.py --max-pairs 15000 --epochs 30   # subsample / control epochs
python src/train.py --gnn-only --epochs 50          # retrain just the GNN
```

Results are upserted into **`results/metrics.csv`** (committed) so a reviewer
sees the full sweep vs. GNN comparison without running anything locally.

### Results (bounded run: 15k-pair subsample, scaffold split)

| Model | RMSE | MAE | Pearson r | R² |
| --- | --- | --- | --- | --- |
| RandomForest | **1.003** | 0.751 | **0.355** | **0.117** |
| GradientBoosting | 1.006 | 0.746 | 0.349 | 0.112 |
| SVR | 1.019 | 0.737 | 0.300 | 0.089 |
| Lasso | 1.023 | 0.763 | 0.297 | 0.081 |
| XGBoost | 1.024 | 0.748 | 0.321 | 0.080 |
| ElasticNet | 1.037 | 0.777 | 0.289 | 0.057 |
| KNN | 1.179 | 0.840 | 0.234 | −0.221 |
| Ridge | 1.476 | 1.155 | 0.111 | −0.913 |
| GNN_GraphDTA (30 ep) | 1.287 | 0.959 | 0.261 | −0.526 |
| LinearRegression | 4.856 | 3.693 | 0.093 | −19.70 |

> On this bounded CPU run the **GNN does not beat the best classical baseline** —
> Random Forest leads. That is reported as-is: on KIBA this is a common, valid
> outcome, and landing in the published baseline ballpark is the credible claim,
> not beating state of the art. The GNN learns real rank correlation (r ≈ 0.26)
> but is undertrained on a 15k-pair fraction of KIBA; the full ~118k set, more
> epochs, and a GPU are the levers to close the gap (all exposed as CLI flags).

---

## Experiment Tracking with MLflow

Tracking is **local only** (`mlruns/`, no cloud server). Runs are **nested**:
one parent `algorithm_sweep` run logs the shared config (`cv_folds`,
`random_seed`, split strategy) once; each classical regressor and the GNN is a
child run.

Run training with tracking enabled:

```bash
python src/train.py
```

Launch the MLflow UI to inspect runs:

```bash
mlflow ui     # open http://127.0.0.1:5000
```

**Per child run we log:**
- **Params:** `model_name`, featurization (FP radius/bits, or graph scheme),
  `split_strategy="scaffold"`, key hyperparameters (`alpha`, `n_estimators`,
  `hidden_dim`/`num_layers`/`learning_rate`/`epochs`), `random_seed`.
- **Metrics:** RMSE, MAE, Pearson r, R² — regression metrics only (classification
  metrics are undefined for continuous affinity; this is a deliberate choice,
  noted in the code).
- **Artifact:** `mlflow.sklearn.log_model` for every classical regressor;
  `mlflow.pytorch.log_model` for the GNN (sklearn's flavor won't accept a
  PyTorch module).

**Comparing experiments:** open the parent `algorithm_sweep` run, view its child
runs, select the ones to compare, and sort by RMSE or R² to find the best
configuration. Use the parallel-coordinates view to see which hyperparameters
(e.g., GNN `hidden_dim`, `num_layers`) correlate with better scores.

---

## Testing

```bash
pytest
```

Covers:
- **Scaffold-split correctness** — no scaffold appears in both train and test.
- **Featurization** — fingerprint / graph output shape and dtype.
- **Metrics** — each function pinned against known input/output pairs.
- **GNN smoke test** — the full training loop runs one epoch end-to-end on a
  tiny slice and produces finite predictions.

---

## What this is *not*

- **Not** a drug-discovery platform, and it **cannot** identify real drug
  candidates.
- Reported RMSE/R² are whatever is actually achieved, including mediocre values.
  The honest target is the **published KIBA baseline range**, not state of the
  art.
