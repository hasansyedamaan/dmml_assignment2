# DM4ML Assignment 2 — Instructions and AI Agent Prompt

**Student:** Syed Amaan Hasan
**ID:** 2025AG05941
**Type:** Individual project (no group)

This file has two parts:

- **Part A**: the full assignment instructions, restated completely.
- **Part B**: a ready-to-paste prompt for an AI coding agent that builds the project end to end.

---

# PART A — ASSIGNMENT INSTRUCTIONS

## Assignment Details

- **Course:** Data Management for Machine Learning (DM4ML), Assignment 2
- **Title:** End-to-End Data Management Pipeline for a Recommendation System
- **Weightage:** 10%
- **Submission date:** Refer to the assignment announcement

## Objective

Design and implement an end-to-end data management pipeline for a recommendation system. The pipeline should:

1. ingest raw data,
2. validate its quality,
3. prepare and transform it,
4. generate features,
5. train and evaluate a recommendation model, and
6. automate the complete workflow.

## Problem Statement

Build a recommendation pipeline for an online platform (e-commerce site, movie platform, online course portal, or music application). The system recommends relevant items to users based on their past interactions and item information.

The pipeline must cover these stages:

```
Data Ingestion → Data Validation → Data Preparation → Data Transformation →
Feature Management → Model Training → Model Evaluation → Orchestration
```

## Requirements

### 1. Data Ingestion
- Ingest **user, item, and interaction** data from CSV files, a public dataset, a database, or synthetic data.
- Required data: user details, item details, and user-item interactions (ratings, views, clicks, or purchases).

### 2. Data Validation
Implement **automated checks** for:
- missing values
- duplicates
- incorrect data types
- invalid ratings
- invalid timestamps
- referential integrity

Invalid records must be **logged or separated into an error dataset**.

### 3. Data Preparation and Transformation
- Clean the data.
- Handle missing values and duplicates.
- Standardize formats.
- Create train/test datasets using a **time-based split**.
- Create derived datasets: **user activity summaries, item popularity, and a user-item interaction matrix**.

### 4. Feature Management
- Create and **document** user, item, and interaction features.
- Examples: user average rating, user activity count, item popularity, item average rating, preferred category, days since last interaction.
- Features must be **reusable for both training and prediction**.

### 5. Model Training
Train **at least two models**:
- A **baseline popularity-based recommender**.
- A **personalized recommender** (collaborative filtering, matrix factorization, content-based filtering, or a hybrid).

**Track model parameters, metrics, and artifacts** using MLflow or another experiment-tracking tool.

### 6. Model Evaluation
- Evaluate with **Precision@K, Recall@K, NDCG@K, Coverage, and Diversity**.
- Use **K = 5 or K = 10**.
- If explicit ratings are predicted, also report **RMSE and MAE**.
- **Compare both models in a results table.**

### 7. Orchestration
- Automate the complete workflow with **Apache Airflow, Prefect, Dagster, or a clearly structured scheduled Python workflow**.
- The pipeline must be **reproducible** and handle **task dependencies** correctly.

## Suggested Tools

Python, Pandas, Great Expectations or Pandera, Feast or a custom feature layer, Scikit-learn, Surprise, LightFM, Implicit, MLflow, Apache Airflow or Prefect, and Parquet/CSV/PostgreSQL for storage.

## Deliverables

A **GitHub repository** containing:

- [ ] Complete source code organized into modules
- [ ] `README.md` with setup and execution instructions
- [ ] `requirements.txt`
- [ ] Architecture diagram of the pipeline
- [ ] Data validation report or logs
- [ ] Feature catalog in table format
- [ ] Model evaluation results comparing both models
- [ ] Orchestration DAG/workflow file **and a screenshot of a successful run**
- [ ] Short final report (2–3 pages) explaining design decisions, tools used, results, and limitations

## Suggested Project Structure

```
recommendation-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── errors/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── preparation/
│   ├── transformation/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── dags/
├── configs/
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## Evaluation Criteria

| Component | Weight |
|---|---|
| Data ingestion and validation | 20% |
| Data preparation and transformation | 20% |
| Feature management | 15% |
| Model training and evaluation | 25% |
| Orchestration and reproducibility | 10% |
| Code quality, documentation, and report | 10% |

## Submission Guidelines

- Submit the **GitHub repository link** and the **final report as PDF**.
- Use a public dataset (MovieLens, Amazon Reviews, Last.fm) or generate clearly documented synthetic data.
- Focus on a **clean, modular, reproducible pipeline**, not only high model accuracy.
- The pipeline must run **end to end using the instructions in the README**.

---

# PART B — PROMPT FOR YOUR AI AGENT

Copy everything inside the block below and paste it into your AI coding agent.

````text
You are a senior ML/data engineer. Build a complete, working, individual university assignment project end to end. Read all instructions carefully, follow every requirement, and verify your work by actually running the pipeline. Do not stop at a plan: write the code, run it, fix errors, and produce every deliverable.

=====================================================================
STUDENT INFO (use in README, report, architecture diagram footer)
=====================================================================
Name: Syed Amaan Hasan
ID: 2025AG05941
Course: Data Management for Machine Learning (DM4ML), Assignment 2
This is an INDIVIDUAL project (no group members).

=====================================================================
PROJECT
=====================================================================
Title: End-to-End Data Management Pipeline for a Recommendation System

Build a recommendation pipeline for an online platform. Pipeline stages:
Data Ingestion -> Data Validation -> Data Preparation -> Data Transformation ->
Feature Management -> Model Training -> Model Evaluation -> Orchestration

The focus is a clean, modular, reproducible pipeline, not high model accuracy.

=====================================================================
DECISIONS (use these defaults unless something fails)
=====================================================================
- Dataset: MovieLens 100K (users, movies with genres, ratings with timestamps). Download it from
  https://files.grouplens.org/datasets/movielens/ml-100k.zip into data/raw/.
  If the download is blocked (no network), fall back to a clearly documented synthetic
  generator (src/ingestion/synthetic.py) producing users, items with categories, and
  timestamped ratings with a realistic popularity skew. Document which source was used.
- Language: Python 3.10+. Libraries: pandas, numpy, scipy, pyarrow, pandera, scikit-learn,
  implicit or surprise (choose whichever installs cleanly; scikit-learn TruncatedSVD or
  item-based KNN via scipy is an acceptable fallback), mlflow, prefect, pyyaml, matplotlib, pytest.
- Storage: Parquet (CSV for human-readable reports).
- Orchestration: Prefect flow (preferred). Also provide a plain-Python runner
  (run_pipeline.py) that executes the same stages in order.
- Tracking: MLflow with a local file store (./mlruns).
- Fix all random seeds (config-driven) for reproducibility. Pin versions in requirements.txt.

=====================================================================
REQUIRED PROJECT STRUCTURE (create exactly this, plus extras where noted)
=====================================================================
recommendation-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── errors/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── preparation/
│   ├── transformation/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── dags/                 (Prefect flow file, e.g. dags/recommendation_flow.py)
├── configs/              (config.yaml: paths, K, split rule, seeds, model params)
├── models/
├── reports/              (validation report, feature catalog, eval results, figures, final report)
├── tests/                (pytest unit tests for validators, split, metrics, features)
├── run_pipeline.py
├── requirements.txt
└── README.md
Every package folder needs __init__.py. Keep functions small, typed, documented, and logged
via the logging module. No hard-coded paths: read everything from configs/config.yaml.

=====================================================================
REQUIREMENTS TO IMPLEMENT (every item is graded)
=====================================================================

1) DATA INGESTION (src/ingestion)
- Load three datasets: users, items, interactions (ratings + timestamps).
- Save unchanged raw copies to data/raw/.
- Log row counts and schema on load.

2) DATA VALIDATION (src/validation) — use Pandera (or Great Expectations)
Automated checks for ALL of:
  - missing values
  - duplicates
  - incorrect data types
  - invalid ratings (outside the valid range, e.g. 1-5)
  - invalid timestamps (unparseable, in the future, or before a sane minimum)
  - referential integrity (interaction.user_id in users, interaction.item_id in items)
Invalid records must be separated into error datasets in data/errors/ (parquet/CSV) with a
column stating which rule failed, and logged. Generate a data validation report
(reports/validation_report.md and .json/.csv) with per-check counts and pass/fail, plus
rows in / rows valid / rows rejected.
To demonstrate the validators work, add a configurable "inject_faults" option (off by
default, ON for the demo run) that injects a small number of faults into a COPY of the raw
data: null values, duplicate rows, a rating of 9 or -1, a future timestamp, wrong dtype, and
an orphan user_id/item_id. Show in the report that these were caught and quarantined.

3) DATA PREPARATION AND TRANSFORMATION (src/preparation, src/transformation)
- Clean data: handle missing values (documented strategy), drop duplicates, standardize
  formats (UTC timestamps, consistent id dtypes, normalized genre/category strings).
- TIME-BASED train/test split (e.g. global cutoff by timestamp quantile, or per-user last
  N interactions / last X%). No random split. Document the choice and guarantee no test
  interaction precedes the train cutoff for the global approach. Handle cold-start in test
  (document how users/items unseen in train are treated).
- Derived datasets saved to data/processed/:
    * user activity summary (interaction count, avg rating, first/last interaction time)
    * item popularity (interaction count, avg rating, popularity rank)
    * user-item interaction matrix (scipy sparse CSR saved as .npz plus id-index mappings)
- Save train.parquet and test.parquet.

4) FEATURE MANAGEMENT (src/features)
- Build a reusable feature layer (a custom feature store module, or Feast if it works
  smoothly). The SAME code path must serve training and prediction (e.g. a FeatureStore
  class with build_features(train_df) -> writes to data/features/, and
  get_user_features(user_ids) / get_item_features(item_ids) for online/batch lookup).
- Compute features from TRAIN data only (no leakage), with an explicit as-of/reference date
  for "days since last interaction".
- Features (minimum):
    User: avg_rating, activity_count, preferred_category (top genre), days_since_last_interaction,
          rating_std, distinct_items_count
    Item: popularity (interaction count), avg_rating, rating_count, category/genre one-hot or list,
          item age if available
    Interaction: rating, timestamp, hour/day-of-week, user-item genre match flag,
          recency rank of the interaction for that user
- Save to data/features/ as parquet.
- Write a FEATURE CATALOG in table format (reports/feature_catalog.md and .csv) with columns:
  Feature name | Entity (user/item/interaction) | Description | Formula/logic | Data type |
  Source table | Used by (training/prediction). Generate it programmatically from a feature
  registry in code so docs never drift from the implementation.

5) MODEL TRAINING (src/training)
Train at least two models:
  a) Baseline: popularity-based recommender (top-N most popular unseen items per user).
  b) Personalized: matrix factorization (e.g. implicit ALS / Surprise SVD / TruncatedSVD)
     and/or item-based collaborative filtering. A hybrid with content/genre features is a
     bonus, not required.
Both models must expose the same interface: fit(train), recommend(user_id, k, exclude_seen=True).
If the personalized model predicts explicit ratings, also expose predict(user_id, item_id).
Log with MLflow: parameters (hyperparameters, seed, K, split cutoff), metrics, and
artifacts (serialized model in models/, evaluation tables, plots, feature catalog, validation
report). Use one MLflow experiment with one run per model, plus tags (dataset, git commit if
available).

6) MODEL EVALUATION (src/evaluation)
- On the time-based test set, compute at K=10 (configurable; may also report K=5):
  Precision@K, Recall@K, NDCG@K, Coverage (catalog coverage of recommended items), and
  Diversity (e.g. average intra-list dissimilarity using genre vectors or item-embedding
  cosine distance; document the definition).
- If explicit ratings are predicted, also compute RMSE and MAE.
- Define "relevant" clearly (e.g. test rating >= 4) and document it.
- Produce a results table comparing both models (reports/model_comparison.md and .csv), log
  it to MLflow, and include a bar chart in reports/figures/.
- Write unit tests for the metrics using tiny hand-computed examples.

7) ORCHESTRATION (dags/ and run_pipeline.py)
- Implement the whole workflow as a Prefect flow with one task per stage and explicit
  dependencies: ingest -> validate -> prepare -> transform -> features -> train (baseline,
  personalized) -> evaluate -> report. Add retries and logging.
- Idempotent and reproducible: re-running gives the same outputs given the same seed/config.
- Also provide run_pipeline.py (CLI, argparse) executing the same stages sequentially, with
  flags like --inject-faults and --skip-download.
- Run the flow successfully at least once. Capture evidence of a successful run: save the
  full run log to reports/run_log.txt, and produce a screenshot-style image of the
  successful run (reports/figures/orchestration_success.png). If a Prefect UI screenshot is
  possible (Prefect server + headless Chromium/Playwright screenshot), do that; otherwise
  render a clear task-status/DAG image from the actual run results (state per task, duration)
  with matplotlib and clearly label it as generated from the real run. Tell the user
  explicitly that they may replace it with a real UI screenshot of their own run.

=====================================================================
OTHER DELIVERABLES
=====================================================================
- README.md: overview, architecture diagram, dataset source, setup (venv, pip install -r
  requirements.txt), how to run end to end (one command), how to run each stage, how to
  open MLflow UI, how to run tests, how to start Prefect, project structure, outputs location,
  results summary, author name and ID.
- requirements.txt with pinned versions that actually install.
- Architecture diagram: reports/figures/architecture.png (and the Mermaid source in
  reports/architecture.mmd). Show stages, data stores (raw/processed/features/errors),
  feature store, models, MLflow, and orchestrator. Embed it in README.
- Data validation report (reports/validation_report.md + logs in reports/ or logs/).
- Feature catalog table (generated, see above).
- Model evaluation results comparing both models (table + chart).
- Orchestration flow file in dags/ plus the successful-run evidence described above.
- FINAL REPORT (2-3 pages, PDF): reports/final_report.pdf. Sections: 1) Introduction and
  problem, 2) Architecture and design decisions (with the diagram), 3) Tools used and why,
  4) Data validation findings, 5) Preparation/transformation and split strategy,
  6) Feature management, 7) Models, 8) Results (comparison table + short analysis of why
  the personalized model does or does not beat the baseline), 9) Orchestration and
  reproducibility, 10) Limitations and future work. Header with Name: Syed Amaan Hasan,
  ID: 2025AG05941. Use REAL numbers from the actual run, never invented ones. VERIFY the PDF is
  2-3 pages (render and count pages); trim or expand to fit.
- .gitignore (exclude venv, mlruns bulk if large, __pycache__, large raw data; but KEEP
  reports/, figures, and the small processed outputs needed as evidence), and a LICENSE-free,
  clean repo.
- Initialize a git repository with meaningful commits per stage. Do NOT push anywhere or
  invent a remote URL; give the user the exact commands to create a public GitHub repo
  and push (git remote add origin ..., git push -u origin main).

=====================================================================
QUALITY BAR AND VERIFICATION (do all of these before finishing)
=====================================================================
1. Run the full pipeline from a clean state with one command and confirm it completes.
2. Run it a second time and confirm results are identical (reproducibility).
3. Run pytest; all tests pass.
4. Run with --inject-faults and confirm the error datasets and validation report show the
   injected faults being caught.
5. Check every checklist item below and report any item you could not complete and why.
6. Never fabricate results, screenshots, or metrics. Everything reported must come from
   real execution output.

DELIVERABLES CHECKLIST (confirm each at the end, with the file path):
[ ] Modular source code in src/ (ingestion, validation, preparation, transformation, features, training, evaluation)
[ ] README.md with setup + execution instructions
[ ] requirements.txt
[ ] Architecture diagram
[ ] Data validation report/logs
[ ] Feature catalog (table)
[ ] Model evaluation results (both models compared; Precision@K, Recall@K, NDCG@K, Coverage, Diversity; RMSE/MAE if applicable)
[ ] Orchestration DAG/workflow file in dags/ + successful-run evidence
[ ] MLflow tracking (params, metrics, artifacts) for both models
[ ] Final report PDF (2-3 pages) with name and ID
[ ] Unit tests
[ ] Git repo initialized with commit history + push instructions

When done, give a short summary: what was built, the comparison table, where each
deliverable lives, and anything the user must do manually (create the GitHub repo and push,
optionally take a real Prefect UI screenshot, and submit the repo link and PDF).
````

---

## Tips for using the prompt

- If your agent has no internet access, it should use the synthetic data fallback. Tell it so if you want a real MovieLens run, and place `ml-100k` in `data/raw/` yourself.
- Check the final PDF is 2–3 pages and that every number in it matches `reports/model_comparison.md`.
- Check that the repo is public before you submit the link.
- Submit the **GitHub link plus `final_report.pdf`** by the date in the assignment announcement.
