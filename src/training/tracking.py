"""MLflow experiment tracking for both recommenders."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

# The project uses the local file store at ./mlruns. MLflow 3 keeps this backend
# behind an opt-in flag, so set it before importing mlflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow

from src.utils.config import project_path

LOGGER = logging.getLogger(__name__)


def _git_commit() -> str:
    """Return the current commit, falling back to reading .git when git is absent."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    git_dir = Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = git_dir / head.split("ref: ", 1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return head or "unavailable"
    except OSError:
        return "unavailable"


def _param(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def log_mlflow_runs(
    models: Dict[str, Any],
    model_params: Dict[str, Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    artifact_paths: Dict[str, Path],
) -> None:
    """Create one MLflow run per model with params, metrics, tags, and artifacts."""
    tracking_path = project_path(config, "mlruns")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.resolve().as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    dataset_source = "movielens100k"
    ingestion_metadata = project_path(config, "data_raw") / "ingestion_metadata.json"
    if ingestion_metadata.exists():
        dataset_source = json.loads(ingestion_metadata.read_text(encoding="utf-8")).get(
            "source", dataset_source
        )

    split_metadata_path = project_path(config, "data_processed") / "split_metadata.json"
    split_metadata: Dict[str, Any] = {}
    if split_metadata_path.exists():
        split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags(
                {
                    "model_type": model_name,
                    "dataset": dataset_source,
                    "student": "Syed Amaan Hasan",
                    "student_id": "2025AG05941",
                    "git_commit": _git_commit(),
                }
            )
            params = dict(model_params[model_name])
            params.update(
                {
                    "k": int(config["models"]["k"]),
                    "secondary_k": int(config["models"]["secondary_k"]),
                    "seed": int(config["project"]["seed"]),
                    "split_method": split_metadata.get(
                        "method", config["split"].get("method", "unknown")
                    ),
                    "split_test_fraction": split_metadata.get(
                        "test_fraction", config["split"].get("test_fraction")
                    ),
                    "split_min_test_interactions": split_metadata.get(
                        "min_test_interactions", config["split"].get("min_test_interactions")
                    ),
                    "split_quantile": split_metadata.get(
                        "quantile", config["split"].get("quantile")
                    ),
                    "split_train_rows": split_metadata.get("train_rows"),
                    "split_test_rows": split_metadata.get("test_rows"),
                    "split_train_users": split_metadata.get("train_users"),
                    "split_test_users": split_metadata.get("test_users"),
                    "split_train_items": split_metadata.get("train_items"),
                    "split_test_items": split_metadata.get("test_items"),
                    "split_cold_start_test_users": split_metadata.get("cold_start_test_users"),
                    "split_cold_start_test_items": split_metadata.get("cold_start_test_items"),
                    "split_cutoff_min": split_metadata.get("cutoff_min"),
                    "split_cutoff_median": split_metadata.get("cutoff_median"),
                    "split_cutoff_max": split_metadata.get("cutoff_max"),
                    "split_train_max_timestamp": split_metadata.get("train_max_timestamp"),
                    "split_test_min_timestamp": split_metadata.get("test_min_timestamp"),
                    "relevance_threshold": int(config["evaluation"]["relevance_threshold"]),
                }
            )
            mlflow.log_params({key: _param(value) for key, value in params.items()})

            model_result = results[model_name]
            for metric_name, value in model_result["metrics"].items():
                if value is None:
                    continue
                mlflow.log_metric(metric_name, float(value))
            for metric_name, value in model_result.get("metrics_secondary", {}).items():
                if value is None:
                    continue
                mlflow.log_metric(metric_name, float(value))

            for artifact_name, path in artifact_paths.items():
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=artifact_name)

            model_path = project_path(config, "models") / f"{model_name}.joblib"
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
            LOGGER.info("Logged MLflow run for %s", model_name)
