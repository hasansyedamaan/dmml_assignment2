# End-to-End Data Management Pipeline for a Recommendation System

**Student:** Syed Amaan Hasan  
**ID:** 2025AG05941  
**Course:** Data Management for Machine Learning (DM4ML), Assignment 2

This repository implements an end-to-end recommendation data pipeline: ingestion,
validation with quarantine, preparation, time-based splitting, train-only feature
engineering, two recommenders, ranking and rating evaluation, MLflow tracking, Prefect
orchestration, and reproducible evidence reports.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python run_pipeline.py                  # full pipeline (downloads MovieLens if needed)
python run_pipeline.py --inject-faults  # validation fault demo
python run_pipeline.py --skip-download  # reuse existing data/raw/ml-100k
python -m pytest                        # all tests
mlflow ui --backend-store-uri ./mlruns  # inspect the two model runs
```

The pipeline writes all artifacts under `reports/`, `data/`, `models/`, and `mlruns/`.
Run it from the repository root so relative configured paths resolve correctly.

## Running individual stages

The CLI runs the complete pipeline. For stage-level debugging, the stage functions in
`src/pipeline.py` are importable and use the same config and logging path:

```python
from src.utils.config import load_config
from src.pipeline import (
    finalize_run,
    stage_evaluate,
    stage_features,
    stage_ingest,
    stage_prepare,
    stage_report,
    stage_train,
    stage_transform,
    stage_validate,
)

config = load_config()
raw = stage_ingest(config, inject_faults_flag=False, skip_download=True)
validated = stage_validate(config, raw)
prepared = stage_prepare(config, validated)
transformed = stage_transform(config, prepared)
features = stage_features(config, transformed)
trained = stage_train(config, transformed)
evaluation = stage_evaluate(config, transformed, trained)
state = {
    "raw": raw,
    "evaluation": evaluation,
    "models": trained["models"],
    "model_params": trained["model_params"],
}
stage_report(config, state)
finalize_run(config, state)
```

Each stage consumes the preceding stage's dictionary. To run one stage in isolation,
construct its input from the persisted Parquet files or call the preceding stage first.

## Dataset and fallback

- Primary source: MovieLens 100K from
  `https://files.grouplens.org/datasets/movielens/ml-100k.zip`.
- Downloaded and extracted into `data/raw/ml-100k/`.
- The reported run used **MovieLens 100K** (943 users, 1,682 items, 100,000 raw
  interactions), not synthetic data.
- If the download or load fails, `src/ingestion/synthetic.py` provides a deterministic
  synthetic fallback. The selected source is recorded in
  `data/raw/ingestion_metadata.json` and `reports/run_summary.json`.

## Architecture

![Pipeline architecture](reports/figures/architecture.png)

Mermaid source: `reports/architecture.mmd`. The flow is:

```text
raw -> ingestion -> validation -> preparation -> time split -> derived datasets
    -> feature store -> popularity + matrix-factorization models -> evaluation
    -> reports / MLflow / orchestration evidence
```

Invalid rows never enter the clean path. They are written to `data/errors/` with a
`failed_rules` column and a `dataset` column.

## Project structure

```text
.
├── configs/                 # config.yaml: paths, K, split, seeds, model params
├── dags/                    # Prefect flow
├── data/
│   ├── raw/                 # unchanged raw copies + ingestion metadata
│   ├── processed/           # cleaned tables, train/test, derived datasets
│   ├── features/            # user/item/interaction feature parquet
│   └── errors/              # quarantined rows with failed_rules
├── models/                  # serialized recommenders + metadata
├── reports/                 # reports, figures, PDF, run log, stage status
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── preparation/
│   ├── transformation/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── reporting/
│   └── utils/
├── tests/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Configuration

All paths, seeds, K values, split rules, validation bounds, feature settings, model
parameters, and MLflow settings are read from `configs/config.yaml`. No pipeline path
or model hyperparameter is hard-coded.

Key defaults:

| Setting | Value |
|---|---|
| Primary K | 10 |
| Secondary K | 5 |
| Relevance | test rating >= 4 |
| Split | per-user last 20% by timestamp |
| Features as-of | train max timestamp |
| Personalized model | scikit-learn `TruncatedSVD` |
| MLflow store | local `./mlruns` |

## Validation and quarantine

The validation stage checks:

- missing required values
- duplicate records
- incorrect dtypes
- ratings outside `[1, 5]`
- unparseable, future, and too-old timestamps
- orphan `user_id` / `item_id` references
- Pandera schema validation on surviving rows

`python run_pipeline.py --inject-faults` deterministically injects each fault class.
The demo run produced `945 -> 943` users, `1684 -> 1682` items, and
`100001 -> 99991` interactions, with every rejected row quarantined under
`data/errors/`. See `reports/validation_report.md`, `.csv`, and `.json`.

Preparation applies documented fallbacks only to rows that survive validation: rows with
missing IDs, ratings, or timestamps are dropped; missing age uses the median; missing
gender becomes `U`; missing occupation/zip becomes `unknown`; missing titles become
`Untitled`; and duplicate IDs/interactions keep the first occurrence.

## Split and cold start

The split is **time-based, never random**. The default is a per-user last-fraction
split: each user's interactions are ordered by timestamp and the final 20% become
test data. This guarantees that no test interaction precedes that user's training
cutoff, and it keeps every test user warm.

In the reported run:

- train: 79,614 interactions, 943 users, 1,611 items
- test: 20,377 interactions, 943 users, 1,508 items
- cold-start test users: 0
- cold-start test items: 71

Cold-start handling: unseen users fall back to popularity if the model is asked for
them; unseen items are not recommendable and therefore count as misses. The alternative
`global_time_quantile` mode is available in `configs/config.yaml`.

## Features and leakage guard

`src/features/feature_store.py` builds user, item, and interaction features from the
**training split only**, using an explicit as-of date (train max timestamp by default).
The same `FeatureStore` serving methods (`get_user_features`, `get_item_features`,
`get_interaction_features`) are exercised by the pipeline and by the tests. The feature
catalog is generated from `src/features/registry.py` into
`reports/feature_catalog.md` and `reports/feature_catalog.csv`; it is not hand-written.

## Models

Both models implement `fit(train)` and `recommend(user_id, k, exclude_seen=True)`.
Both expose `predict(user_id, item_id)` because the pipeline reports explicit rating
metrics.

1. **Popularity baseline** (`src/training/popularity.py`): ranks unseen items by
   training interaction count, with stable item-id tie-breaking.
2. **Matrix factorization** (`src/training/matrix_factorization.py`): scikit-learn
   `TruncatedSVD` on the sparse user-item rating matrix for ranking, plus a separate
   bias-aware residual SVD (global mean + user/item offsets) for rating prediction.
   This scikit-learn/SciPy implementation is used because `implicit` and `surprise`
   do not have compatible wheels in the project environment.

## Metrics

`src/evaluation/metrics.py` implements hand-testable versions of:

- Precision@K
- Recall@K
- NDCG@K (binary relevance, test rating >= 4)
- Catalog coverage = unique recommended items / training catalog size
- Intra-list diversity = mean pairwise cosine distance between item genre vectors
- RMSE and MAE for explicit rating predictions

Metrics are computed at K=10 (primary) and K=5 (secondary). Results are in
`reports/model_comparison.md` / `.csv` and `reports/figures/model_comparison.png`.
The reported run produced:

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0754 | 0.0749 | 0.0943 | 0.0447 | 0.7478 | 1.0735 | 0.8587 |
| matrix_factorization | 0.1320 | 0.1543 | 0.1763 | 0.2862 | 0.7128 | 1.0082 | 0.7915 |

## Orchestration and tracking

- `run_pipeline.py` is the plain-Python CLI and runs the same stage functions in order.
- `dags/recommendation_flow.py` is a Prefect flow with one retryable task per stage and
  explicit dependencies. It can be run directly (Prefect starts an ephemeral local
  server automatically), or against a local Prefect server:

  ```powershell
  prefect server start          # terminal 1, optional
  python dags/recommendation_flow.py
  python dags/recommendation_flow.py --inject-faults
  ```

- MLflow uses the local file store `./mlruns`, one experiment, and one run per model.
  Parameters include seed, K, split method/fraction, cutoff timestamps, and cold-start
  counts; tags include the dataset and git commit. Metrics and small report/model
  artifacts are logged.
- `reports/run_log.txt` is the run log.
- `reports/figures/orchestration_success.png` is generated from the actual stage timings
  and is labelled as not a Prefect UI screenshot.

## Evidence checklist

| Evidence | Location |
|---|---|
| Validation report | `reports/validation_report.md`, `.csv`, `.json` |
| Feature catalog | `reports/feature_catalog.md`, `.csv` |
| Model comparison | `reports/model_comparison.md`, `.csv` |
| Figures | `reports/figures/` |
| Run log | `reports/run_log.txt` |
| Final PDF (3 pages) | `reports/final_report.pdf` |
| Architecture | `reports/architecture.mmd`, `reports/figures/architecture.png` |
| Quarantined rows | `data/errors/` |
| Processed/feature data | `data/processed/`, `data/features/` |
| Models | `models/*.joblib` |
| MLflow | `mlruns/` (local file store) |

## Tests

```powershell
python -m pytest
python -m pytest tests/test_metrics.py -k test_name
```

Tests cover validation/quarantine, per-user and global time splits, hand-computed
metrics, train-only feature building/serving defaults, deterministic fault injection,
and both recommender interfaces.

## Environment notes

- Python 3.15 was used for the reported run; all pinned versions were verified there.
- `implicit` and `surprise` are not installed because compatible wheels were unavailable;
  the scikit-learn/SciPy fallback is used instead.
- PyYAML 6.0.3 has no cp315 wheel and the machine lacks MSVC build tools. A pure-Python
  wheel built from the same source is vendored at
  `vendor/pyyaml-6.0.3-py3-none-any.whl` and referenced by `requirements.txt`.
