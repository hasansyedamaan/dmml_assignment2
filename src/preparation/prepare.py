"""Clean validated tables and standardise their formats."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from src.utils.config import project_path

LOGGER = logging.getLogger(__name__)


def _clean_users(users: pd.DataFrame) -> pd.DataFrame:
    cleaned = users.copy()
    cleaned["user_id"] = pd.to_numeric(cleaned["user_id"], errors="coerce")
    cleaned = cleaned.dropna(subset=["user_id"]).copy()
    cleaned["user_id"] = cleaned["user_id"].astype("int64")
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median()).round().astype("int64")
    cleaned["gender"] = cleaned["gender"].astype("string").str.upper().str.strip().fillna("U")
    cleaned["occupation"] = cleaned["occupation"].astype("string").str.strip().fillna("unknown")
    cleaned["zip"] = cleaned["zip"].astype("string").str.strip().fillna("unknown")
    cleaned = cleaned.drop_duplicates(subset=["user_id"], keep="first").reset_index(drop=True)
    return cleaned


def _clean_items(items: pd.DataFrame) -> pd.DataFrame:
    cleaned = items.copy()
    cleaned["item_id"] = pd.to_numeric(cleaned["item_id"], errors="coerce")
    cleaned = cleaned.dropna(subset=["item_id"]).copy()
    cleaned["item_id"] = cleaned["item_id"].astype("int64")
    cleaned["title"] = cleaned["title"].astype("string").str.strip().fillna("Untitled")

    if not pd.api.types.is_datetime64_any_dtype(cleaned["release_date"]):
        cleaned["release_date"] = pd.to_datetime(
            cleaned["release_date"], errors="coerce", format="%d-%b-%Y", utc=True
        ).dt.tz_localize(None)
    else:
        if getattr(cleaned["release_date"].dt, "tz", None) is not None:
            cleaned["release_date"] = cleaned["release_date"].dt.tz_convert("UTC").dt.tz_localize(None)

    if "genres" not in cleaned.columns:
        genre_columns = [column for column in cleaned.columns if column.startswith("genre_")]
        cleaned["genres"] = cleaned[genre_columns].apply(
            lambda row: [column.replace("genre_", "") for column, value in row.items() if int(value) == 1],
            axis=1,
        )
    cleaned["genres"] = cleaned["genres"].apply(
        lambda values: [str(value).strip() for value in values if str(value).strip()]
    )
    cleaned["primary_genre"] = cleaned["genres"].apply(lambda values: values[0] if values else "unknown")
    for column in ["video_release_date", "imdb_url"]:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=[column])
    cleaned = cleaned.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    return cleaned


def _clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    cleaned = interactions.copy()
    for column in ["user_id", "item_id", "rating", "timestamp"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()
    cleaned["user_id"] = cleaned["user_id"].astype("int64")
    cleaned["item_id"] = cleaned["item_id"].astype("int64")
    cleaned["rating"] = cleaned["rating"].astype("float64")
    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"].astype("int64"), unit="s", errors="coerce", utc=True
    )
    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned = cleaned.drop_duplicates(
        subset=["user_id", "item_id", "timestamp"], keep="first"
    )
    cleaned = cleaned.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    return cleaned


def prepare_data(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Clean validated tables and persist the full cleaned versions."""
    cleaned_users = _clean_users(users)
    cleaned_items = _clean_items(items)
    cleaned_interactions = _clean_interactions(interactions)

    processed_dir = project_path(config, "data_processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_users.to_parquet(processed_dir / "users.parquet", index=False)
    cleaned_items.to_parquet(processed_dir / "items.parquet", index=False)
    cleaned_interactions.to_parquet(processed_dir / "interactions.parquet", index=False)

    LOGGER.info(
        "Prepared data: users=%d items=%d interactions=%d",
        len(cleaned_users),
        len(cleaned_items),
        len(cleaned_interactions),
    )
    return {
        "users": cleaned_users,
        "items": cleaned_items,
        "interactions": cleaned_interactions,
    }
