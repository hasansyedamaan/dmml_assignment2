# AGENTS.md

## Source of truth
- `DM4ML_Assignment2_Instructions_and_Agent_Prompt.md` is the brief. **Part B** (inside the `text` block) is the full build spec and quality bar; Part A is the summary. The `.docx` contains Part A only — do not treat it as newer.
- No pipeline code is provided. Implement the brief's `recommendation-pipeline/` tree at this repo root (map the tree onto this directory; avoid an extra nested project folder).
- Individual assignment: **Syed Amaan Hasan, ID 2025AG05941**. Use these in README/report headers.

## Required stack and decisions
- Python 3.10+; Parquet for data, CSV/Markdown for reports.
- Dataset: MovieLens 100K from `https://files.grouplens.org/datasets/movielens/ml-100k.zip` into `data/raw/`. If download fails, use the documented synthetic fallback in `src/ingestion/synthetic.py` and state which source was used.
- Orchestration: a Prefect flow in `dags/` **and** a plain-Python CLI `run_pipeline.py` that runs the same stages in order.
- Tracking: MLflow with local file store `./mlruns`; one experiment, one run per model.
- Read paths, K, split rule, seeds, and model params from `configs/config.yaml`. No hard-coded paths. Pin working versions in `requirements.txt` and fix seeds.
- Every package directory needs `__init__.py`. Keep functions small, typed, documented, and logged via `logging`.

## Pipeline invariants (easy to get wrong)
- Validation must catch missing values, duplicates, dtype errors, ratings outside the valid range, unparseable/future/too-old timestamps, and referential integrity failures. Put rejected rows in `data/errors/` with the failed-rule column and log them.
- `inject_faults` defaults **off**; the demo run turns it on and must prove every injected fault type is caught and quarantined.
- Train/test split must be **time-based**, never random. Document the cutoff and cold-start handling.
- Features must be computed from **train data only** with an explicit as-of/reference date (no leakage), and the same feature code path must serve training and prediction.
- Generate `reports/feature_catalog.md`/`.csv` from the code feature registry, not by hand.
- Train at least two models: a popularity baseline and a personalized model (MF/CF/hybrid). Both must expose `fit(train)` and `recommend(user_id, k, exclude_seen=True)`; add `predict(user_id, item_id)` if rating prediction is supported.
- Metrics: Precision@K, Recall@K, NDCG@K, Coverage, Diversity (document its definition), plus RMSE/MAE when ratings are predicted. Default K=10; define relevance as test rating >= 4.
- Required evidence: `reports/validation_report.*`, `reports/feature_catalog.*`, `reports/model_comparison.*`, `reports/figures/`, `reports/run_log.txt`, `reports/final_report.pdf`, plus the architecture diagram (`reports/figures/architecture.png` and `reports/architecture.mmd`).
- Reports and metrics must come from actual runs.

## Required commands / interfaces
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python run_pipeline.py                 # full pipeline
python run_pipeline.py --inject-faults # validation fault demo
python run_pipeline.py --skip-download # reuse existing raw data
python -m pytest                       # all tests
python -m pytest tests/test_metrics.py -k test_name  # focused test
mlflow ui --backend-store-uri ./mlruns
```
- Tests live in `tests/`; metric tests must use tiny hand-computed examples.
- Keep the Prefect flow runnable from its module/file entrypoint, with retries and explicit stage dependencies.

## Environment gotchas (verified on this machine)
- System interpreter is **Python 3.15**, and none of the project libraries are preinstalled. Create the venv and install dependencies; if `implicit`/`surprise`/`pandera` lack 3.15 wheels, use the brief's permitted scikit-learn/scipy fallback and document it.
- **`git` is not on PATH.** Install it before attempting `git init`/commits, and never claim a commit history that was not actually created.

## Verification bar
- Run the full pipeline from a clean state, then again; confirm metrics/output tables match (ignore timestamps and MLflow run IDs).
- Run `pytest`; all tests must pass. Run with `--inject-faults` and confirm the error datasets and validation report show each fault.
- Never fabricate metrics, screenshots, or results. `reports/final_report.pdf` must be 2-3 pages and use numbers from real runs.
- For orchestration evidence, save `reports/run_log.txt` and `reports/figures/orchestration_success.png`; label generated images as generated and tell the user they may replace them with a real Prefect UI screenshot.
- `.gitignore`: exclude venv, `__pycache__`, bulk `mlruns`, and large raw data; keep `reports/`, figures, and small processed outputs as evidence.
- Initialize git with meaningful per-stage commits. Do not push or invent a remote; give the user the exact `git remote add`/`git push` commands.
