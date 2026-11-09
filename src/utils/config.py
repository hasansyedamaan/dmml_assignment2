"""Configuration loading and path resolution.

All project paths are relative to the repository root and are resolved here so
pipeline code never needs hard-coded absolute paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML configuration and attach the resolved project root."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def project_path(config: Dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    raw = config["paths"][key]
    path = Path(raw)
    if not path.is_absolute():
        path = Path(config.get("_project_root", PROJECT_ROOT)) / path
    return path


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create every output directory declared in the config."""
    for key in (
        "data_raw",
        "data_processed",
        "data_features",
        "data_errors",
        "models",
        "reports",
        "figures",
        "stage_status",
        "mlruns",
    ):
        project_path(config, key).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable indentation and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
