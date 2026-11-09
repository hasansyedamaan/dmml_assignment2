"""Fit, persist, and describe the two required recommenders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from src.training.matrix_factorization import MatrixFactorizationRecommender
from src.training.popularity import PopularityRecommender
from src.utils.config import project_path, write_json

LOGGER = logging.getLogger(__name__)


def train_models(
    train: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Train the popularity baseline and the personalized SVD model."""
    models_dir = project_path(config, "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    popularity = PopularityRecommender().fit(train)
    mf_cfg = config["models"]["matrix_factorization"]
    personalized = MatrixFactorizationRecommender(
        n_components=int(mf_cfg["n_components"]),
        n_iter=int(mf_cfg["n_iter"]),
        random_state=int(mf_cfg["random_state"]),
        clip_min=float(mf_cfg["clip_min"]),
        clip_max=float(mf_cfg["clip_max"]),
    ).fit(train)

    models = {"popularity": popularity, "matrix_factorization": personalized}
    paths = {
        "popularity": models_dir / "popularity.joblib",
        "matrix_factorization": models_dir / "matrix_factorization.joblib",
    }
    for name, model in models.items():
        joblib.dump(model, paths[name])

    params = {
        "popularity": {
            "model": "popularity",
            "weighting": config["models"]["popularity"]["weighting"],
            "train_rows": int(len(train)),
            "catalog_size": int(train["item_id"].nunique()),
        },
        "matrix_factorization": {
            "model": "matrix_factorization",
            "n_components": int(mf_cfg["n_components"]),
            "n_iter": int(mf_cfg["n_iter"]),
            "random_state": int(mf_cfg["random_state"]),
            "clip_min": float(mf_cfg["clip_min"]),
            "clip_max": float(mf_cfg["clip_max"]),
            "train_rows": int(len(train)),
            "catalog_size": int(train["item_id"].nunique()),
            "explained_variance": float(personalized.explained_variance),
        },
    }
    write_json(
        models_dir / "model_metadata.json",
        {"models": {name: str(path) for name, path in paths.items()}, "params": params},
    )
    LOGGER.info("Trained models: popularity and matrix_factorization")
    return models, params
