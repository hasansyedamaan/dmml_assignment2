"""Prefect flow with one retryable task per pipeline stage.

Run locally with:
    python dags/recommendation_flow.py
    python dags/recommendation_flow.py --inject-faults
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the repository root importable when this file is executed directly
# (``python dags/recommendation_flow.py``), not only as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid background telemetry noise in local, offline assignment runs.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_ANALYTICS_ENABLED", "false")

from prefect import flow, task

from src.pipeline import (
    finalize_run,
    reset_stage_status,
    run_stage,
    stage_evaluate,
    stage_features,
    stage_ingest,
    stage_prepare,
    stage_report,
    stage_train,
    stage_transform,
    stage_validate,
)
from src.utils.config import ensure_directories, load_config, project_path
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


@task(retries=2, retry_delay_seconds=3)
def ingest_task(config, inject_faults_flag, skip_download):
    return run_stage(config, "ingest", stage_ingest, config, inject_faults_flag, skip_download)


@task(retries=2, retry_delay_seconds=3)
def validate_task(config, raw):
    return run_stage(config, "validate", stage_validate, config, raw)


@task(retries=2, retry_delay_seconds=3)
def prepare_task(config, validated):
    return run_stage(config, "prepare", stage_prepare, config, validated)


@task(retries=2, retry_delay_seconds=3)
def transform_task(config, prepared):
    return run_stage(config, "transform", stage_transform, config, prepared)


@task(retries=2, retry_delay_seconds=3)
def features_task(config, transformed):
    return run_stage(config, "features", stage_features, config, transformed)


@task(retries=2, retry_delay_seconds=3)
def train_task(config, transformed):
    return run_stage(config, "train", stage_train, config, transformed)


@task(retries=2, retry_delay_seconds=3)
def evaluate_task(config, transformed, trained):
    return run_stage(config, "evaluate", stage_evaluate, config, transformed, trained)


@task(retries=2, retry_delay_seconds=3)
def report_task(config, state):
    return run_stage(config, "report", stage_report, config, state)


@flow(name="dm4ml-recommendation-pipeline")
def recommendation_flow(inject_faults: bool = False, skip_download: bool = False):
    """Run the full pipeline under Prefect with explicit stage dependencies."""
    config = load_config()
    ensure_directories(config)
    reset_stage_status(config)

    state = {}
    state["raw"] = ingest_task(config, inject_faults, skip_download)
    state["validated"] = validate_task(config, state["raw"])
    state["prepared"] = prepare_task(config, state["validated"])
    state["transformed"] = transform_task(config, state["prepared"])
    state["features"] = features_task(config, state["transformed"])
    state["trained"] = train_task(config, state["transformed"])
    state["evaluation"] = evaluate_task(config, state["transformed"], state["trained"])
    state["models"] = state["trained"]["models"]
    state["model_params"] = state["trained"]["model_params"]
    state["report"] = report_task(config, state)
    state["summary"] = finalize_run(config, state)
    LOGGER.info("Prefect flow complete: %s", state["summary"])
    return state["summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Prefect recommendation flow.")
    parser.add_argument("--inject-faults", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config = load_config()
    ensure_directories(config)
    setup_logging(log_file=project_path(config, "reports") / "run_log.txt")
    recommendation_flow(inject_faults=args.inject_faults, skip_download=args.skip_download)
