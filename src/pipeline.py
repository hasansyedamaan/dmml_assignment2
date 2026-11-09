"""Stage functions shared by the CLI runner and the Prefect flow."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from src.evaluation import evaluate_models
from src.features import FeatureStore
from src.ingestion import ingest_data
from src.preparation import prepare_data
from src.reporting import (
    plot_architecture,
    plot_orchestration,
    write_architecture_mermaid,
    write_final_report_pdf,
)
from src.training import train_models
from src.training.tracking import log_mlflow_runs
from src.transformation import create_derived_datasets, time_based_split
from src.utils.config import ensure_directories, project_path, write_json
from src.validation import validate_data

LOGGER = logging.getLogger(__name__)


def reset_stage_status(config: Dict[str, Any]) -> None:
    """Clear prior stage evidence so the orchestration image reflects this run."""
    status_dir = project_path(config, "stage_status")
    status_dir.mkdir(parents=True, exist_ok=True)
    for path in status_dir.glob("*.json"):
        path.unlink()


def run_stage(
    config: Dict[str, Any],
    name: str,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a stage, timing it and persisting its real status."""
    status_dir = project_path(config, "stage_status")
    status_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    LOGGER.info("Starting stage: %s", name)
    try:
        result = function(*args, **kwargs)
    except Exception as exc:
        finished = datetime.now(timezone.utc)
        write_json(
            status_dir / f"{name}.json",
            {
                "stage": name,
                "status": "FAILED",
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "duration_seconds": (finished - started).total_seconds(),
                "error": str(exc),
            },
        )
        LOGGER.exception("Stage failed: %s", name)
        raise
    finished = datetime.now(timezone.utc)
    write_json(
        status_dir / f"{name}.json",
        {
            "stage": name,
            "status": "SUCCESS",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "error": None,
        },
    )
    LOGGER.info("Finished stage: %s in %.2fs", name, (finished - started).total_seconds())
    return result


def stage_ingest(
    config: Dict[str, Any], inject_faults_flag: bool, skip_download: bool
) -> Dict[str, Any]:
    return ingest_data(config, skip_download=skip_download, inject_faults_flag=inject_faults_flag)


def stage_validate(config: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    validated = validate_data(
        raw["users"],
        raw["items"],
        raw["interactions"],
        config,
        faults_injected=bool(raw.get("faults_injected", False)),
    )
    validated["faults_injected"] = bool(raw.get("faults_injected", False))
    validated["source"] = raw.get("source", "unknown")
    return validated


def stage_prepare(config: Dict[str, Any], validated: Dict[str, Any]) -> Dict[str, Any]:
    return prepare_data(
        validated["users"],
        validated["items"],
        validated["interactions"],
        config,
    )


def stage_transform(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    train, test, split_metadata = time_based_split(prepared["interactions"], config)
    derived = create_derived_datasets(train, prepared["items"], config)
    return {
        "train": train,
        "test": test,
        "split_metadata": split_metadata,
        "derived": derived,
        "users": prepared["users"],
        "items": prepared["items"],
    }


def stage_features(config: Dict[str, Any], transformed: Dict[str, Any]) -> Dict[str, Any]:
    store = FeatureStore(config)
    features = store.build_features(
        transformed["train"], transformed["users"], transformed["items"]
    )
    # Exercise the same serving path used at prediction time.
    store.get_user_features(transformed["train"]["user_id"].unique())
    store.get_item_features(transformed["train"]["item_id"].unique())
    return features


def stage_train(config: Dict[str, Any], transformed: Dict[str, Any]) -> Dict[str, Any]:
    models, params = train_models(transformed["train"], config)
    return {"models": models, "model_params": params}


def stage_evaluate(
    config: Dict[str, Any], transformed: Dict[str, Any], trained: Dict[str, Any]
) -> Dict[str, Any]:
    return evaluate_models(
        trained["models"],
        transformed["test"],
        transformed["items"],
        transformed["train"],
        config,
    )


def stage_report(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    del state
    write_architecture_mermaid(config)
    architecture = plot_architecture(config)
    pdf_path = write_final_report_pdf(config)
    return {"architecture": str(architecture), "final_report": str(pdf_path)}


def _artifact_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    reports_dir = project_path(config, "reports")
    figures_dir = project_path(config, "figures")
    return {
        "validation_report.md": reports_dir / "validation_report.md",
        "validation_report.json": reports_dir / "validation_report.json",
        "validation_report.csv": reports_dir / "validation_report.csv",
        "feature_catalog.md": reports_dir / "feature_catalog.md",
        "feature_catalog.csv": reports_dir / "feature_catalog.csv",
        "model_comparison.md": reports_dir / "model_comparison.md",
        "model_comparison.csv": reports_dir / "model_comparison.csv",
        "architecture.mmd": reports_dir / "architecture.mmd",
        "architecture.png": figures_dir / "architecture.png",
        "model_comparison.png": figures_dir / "model_comparison.png",
        "orchestration_success.png": figures_dir / "orchestration_success.png",
        "final_report.pdf": reports_dir / "final_report.pdf",
        "run_log.txt": reports_dir / "run_log.txt",
    }


def finalize_run(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final orchestration evidence and log both models to MLflow."""
    plot_orchestration(config)
    evaluation = state["evaluation"]
    log_mlflow_runs(
        state["models"],
        state["model_params"],
        evaluation,
        config,
        _artifact_paths(config),
    )
    summary = {
        "source": state.get("raw", {}).get("source", "unknown"),
        "faults_injected": bool(state.get("raw", {}).get("faults_injected", False)),
        "metrics": {
            model: result.get("metrics", {})
            for model, result in evaluation.items()
            if isinstance(result, dict) and "metrics" in result
        },
    }
    write_json(project_path(config, "reports") / "run_summary.json", summary)
    return summary


def run_pipeline(
    config: Dict[str, Any],
    inject_faults_flag: bool = False,
    skip_download: bool = False,
) -> Dict[str, Any]:
    """Run every stage in dependency order (used by the plain CLI runner)."""
    ensure_directories(config)
    reset_stage_status(config)
    state: Dict[str, Any] = {}

    state["raw"] = run_stage(config, "ingest", stage_ingest, config, inject_faults_flag, skip_download)
    state["validated"] = run_stage(config, "validate", stage_validate, config, state["raw"])
    state["prepared"] = run_stage(config, "prepare", stage_prepare, config, state["validated"])
    state["transformed"] = run_stage(config, "transform", stage_transform, config, state["prepared"])
    state["features"] = run_stage(config, "features", stage_features, config, state["transformed"])
    state["trained"] = run_stage(config, "train", stage_train, config, state["transformed"])
    state["evaluation"] = run_stage(
        config, "evaluate", stage_evaluate, config, state["transformed"], state["trained"]
    )
    state["models"] = state["trained"]["models"]
    state["model_params"] = state["trained"]["model_params"]
    state["report"] = run_stage(config, "report", stage_report, config, state)
    state["summary"] = finalize_run(config, state)
    LOGGER.info("Pipeline complete. Summary: %s", json.dumps(state["summary"], default=str))
    return state
