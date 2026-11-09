"""Download, load, persist, and optionally corrupt raw recommendation data."""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.ingestion.synthetic import GENRES, generate_synthetic_data
from src.utils.config import ensure_directories, project_path, write_json

LOGGER = logging.getLogger(__name__)

GENRE_COLUMNS = [f"genre_{i}" for i in range(len(GENRES))]
ITEM_COLUMNS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip"]


def download_movielens(config: Dict[str, Any], skip_download: bool = False) -> Path:
    """Ensure MovieLens 100K is present locally and return its directory."""
    dataset_dir = project_path(config, "dataset_dir")
    zip_path = project_path(config, "dataset_zip")
    expected = dataset_dir / "u.data"
    if expected.exists():
        LOGGER.info("MovieLens already present at %s", dataset_dir)
        return dataset_dir
    if skip_download:
        raise FileNotFoundError(
            f"MovieLens data not found at {dataset_dir} and --skip-download was requested."
        )

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading MovieLens 100K from %s", config["dataset"]["url"])
    urllib.request.urlretrieve(config["dataset"]["url"], zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_dir.parent)
    if not expected.exists():
        raise FileNotFoundError(f"Downloaded archive did not contain {expected}")
    LOGGER.info("Extracted MovieLens 100K to %s", dataset_dir)
    return dataset_dir


def _read_genre_names(dataset_dir: Path) -> List[str]:
    genre_path = dataset_dir / "u.genre"
    if not genre_path.exists():
        return GENRES
    names: List[Tuple[int, str]] = []
    for line in genre_path.read_text(encoding="latin-1").splitlines():
        if "|" not in line:
            continue
        name, index = line.rsplit("|", 1)
        names.append((int(index), name.strip()))
    return [name for _, name in sorted(names)]


def _load_movielens(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    dataset_dir = project_path(config, "dataset_dir")
    genres = _read_genre_names(dataset_dir)
    genre_columns = [f"genre_{i}" for i in range(len(genres))]
    users = pd.read_csv(
        dataset_dir / "u.user",
        sep="|",
        names=USER_COLUMNS,
        dtype={"user_id": "int64", "age": "int64", "occupation": "string", "zip": "string"},
    )
    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_columns,
        encoding="latin-1",
        dtype={"item_id": "int64"},
    )
    interactions = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=INTERACTION_COLUMNS,
        dtype={"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"},
    )
    items["genres"] = items.apply(
        lambda row: [genres[i] for i in range(len(genres)) if int(row[f"genre_{i}"]) == 1],
        axis=1,
    )
    items["primary_genre"] = items["genres"].apply(lambda values: values[0] if values else "unknown")
    return users, items, interactions, "movielens100k"


def _load_synthetic(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    users, items, interactions = generate_synthetic_data(config)
    return users, items, interactions, "synthetic"


def _save_raw_copy(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _raw_copy_frame(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return the raw columns only, so raw evidence stays close to the source."""
    if kind == "users":
        return df[USER_COLUMNS].copy()
    if kind == "items":
        columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS
        return df[columns].copy()
    return df[INTERACTION_COLUMNS].copy()


def inject_faults(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return corrupted copies covering every validation rule in the brief.

    Faults are deterministic and are only applied when ``inject_faults`` is on.
    New user/item rows are used for missing-value faults so they do not cascade
    into referential-integrity failures for otherwise-valid interactions.
    """
    del seed  # Kept for a stable signature; the injected positions are fixed.
    users_faulty = users.copy()
    items_faulty = items.copy()
    interactions_faulty = interactions.copy()

    # Missing values in required fields on extra rows that no interaction references.
    missing_user = {column: users_faulty.iloc[0][column] for column in USER_COLUMNS}
    missing_user.update({"user_id": 999998, "gender": np.nan})
    users_faulty = pd.concat([users_faulty, pd.DataFrame([missing_user])], ignore_index=True)

    missing_item = {column: items_faulty.iloc[0][column] for column in items_faulty.columns}
    missing_item.update({"item_id": 999998, "title": np.nan})
    items_faulty = pd.concat([items_faulty, pd.DataFrame([missing_item])], ignore_index=True)

    # Duplicate rows.
    users_faulty = pd.concat([users_faulty, users_faulty.iloc[[1]]], ignore_index=True)
    items_faulty = pd.concat([items_faulty, items_faulty.iloc[[1]]], ignore_index=True)
    interactions_faulty = pd.concat([interactions_faulty, interactions_faulty.iloc[[1]]], ignore_index=True)

    # Missing rating, invalid ratings, future timestamp, too-old timestamp.
    interactions_faulty.loc[10, "rating"] = np.nan
    interactions_faulty.loc[12, "rating"] = 9
    interactions_faulty.loc[13, "rating"] = -1
    interactions_faulty.loc[14, "timestamp"] = 2524608000  # 2050-01-01 UTC
    interactions_faulty.loc[18, "timestamp"] = 0

    # Wrong dtype and unparseable timestamp.
    interactions_faulty["user_id"] = interactions_faulty["user_id"].astype(object)
    interactions_faulty.loc[15, "user_id"] = "not-an-integer"
    interactions_faulty["timestamp"] = interactions_faulty["timestamp"].astype(object)
    interactions_faulty.loc[19, "timestamp"] = "not-a-timestamp"

    # Referential integrity failures.
    interactions_faulty.loc[16, "user_id"] = 999999
    interactions_faulty.loc[17, "item_id"] = 999999

    LOGGER.warning(
        "Injected validation faults: missing values, duplicates, invalid ratings, "
        "invalid timestamps, wrong dtypes, and orphan user/item references."
    )
    return users_faulty, items_faulty, interactions_faulty


def ingest_data(
    config: Dict[str, Any],
    skip_download: bool = False,
    inject_faults_flag: bool = False,
) -> Dict[str, Any]:
    """Load the dataset, persist raw copies, and optionally return corrupted copies."""
    ensure_directories(config)
    raw_dir = project_path(config, "data_raw")
    source = "synthetic"
    try:
        if config["dataset"]["source"] == "movielens100k":
            download_movielens(config, skip_download=skip_download)
            users, items, interactions, source = _load_movielens(config)
        else:
            users, items, interactions, source = _load_synthetic(config)
    except Exception as exc:  # pragma: no cover - exercised only on network failure
        if not config["dataset"].get("fallback_synthetic", True):
            raise
        LOGGER.warning("Falling back to synthetic data because MovieLens could not be loaded: %s", exc)
        users, items, interactions, source = _load_synthetic(config)

    # Save untouched canonical raw copies before any fault injection.
    _save_raw_copy(_raw_copy_frame(users, "users"), raw_dir / "users.csv")
    _save_raw_copy(_raw_copy_frame(items, "items"), raw_dir / "items.csv")
    _save_raw_copy(_raw_copy_frame(interactions, "interactions"), raw_dir / "interactions.csv")

    faulty = False
    if inject_faults_flag:
        users, items, interactions = inject_faults(
            users, items, interactions, seed=int(config["project"]["seed"])
        )
        faulty = True
        _save_raw_copy(_raw_copy_frame(users, "users"), raw_dir / "users_faulty.csv")
        _save_raw_copy(_raw_copy_frame(items, "items"), raw_dir / "items_faulty.csv")
        _save_raw_copy(_raw_copy_frame(interactions, "interactions"), raw_dir / "interactions_faulty.csv")

    metadata = {
        "source": source,
        "faults_injected": faulty,
        "rows": {
            "users": int(len(users)),
            "items": int(len(items)),
            "interactions": int(len(interactions)),
        },
        "columns": {
            "users": list(users.columns),
            "items": list(items.columns),
            "interactions": list(interactions.columns),
        },
    }
    write_json(raw_dir / "ingestion_metadata.json", metadata)
    LOGGER.info("Ingested %s data: users=%d items=%d interactions=%d", source, len(users), len(items), len(interactions))
    LOGGER.info("Users schema: %s", list(users.columns))
    LOGGER.info("Items schema: %s", list(items.columns))
    LOGGER.info("Interactions schema: %s", list(interactions.columns))

    return {
        "users": users,
        "items": items,
        "interactions": interactions,
        "source": source,
        "faults_injected": faulty,
        "metadata": metadata,
    }
