"""Shared pytest fixtures: isolated config rooted in a temporary directory."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import pytest

from src.utils.config import ensure_directories, load_config


@pytest.fixture()
def temp_config(tmp_path: Path) -> Dict[str, Any]:
    """Return the real config with every output path redirected under tmp_path."""
    config = copy.deepcopy(load_config())
    config["_project_root"] = str(tmp_path)
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
        "dataset_dir",
        "dataset_zip",
    ):
        config["paths"][key] = config["paths"][key].replace("\\", "/")
    ensure_directories(config)
    return config
