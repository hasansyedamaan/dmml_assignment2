"""Derived train-only datasets: user activity, item popularity, sparse matrix."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.config import project_path, write_json

LOGGER = logging.getLogger(__name__)


def _user_activity_summary(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    summary = (
        train.groupby("user_id")
        .agg(
            interaction_count=("item_id", "size"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            distinct_items_count=("item_id", "nunique"),
            first_interaction_time=("timestamp", "min"),
            last_interaction_time=("timestamp", "max"),
        )
        .reset_index()
    )
    summary["rating_std"] = summary["rating_std"].fillna(0.0)
    genre_counts = (
        train.merge(items[["item_id", "primary_genre"]], on="item_id", how="left")
        .groupby(["user_id", "primary_genre"])
        .size()
        .reset_index(name="genre_count")
        .sort_values(["user_id", "genre_count", "primary_genre"], ascending=[True, False, True])
        .drop_duplicates("user_id")
    )
    summary = summary.merge(
        genre_counts[["user_id", "primary_genre"]].rename(
            columns={"primary_genre": "preferred_category"}
        ),
        on="user_id",
        how="left",
    )
    summary["preferred_category"] = summary["preferred_category"].fillna("unknown")
    return summary


def _item_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    popularity = (
        train.groupby("item_id")
        .agg(
            interaction_count=("user_id", "size"),
            avg_rating=("rating", "mean"),
            rating_count=("rating", "size"),
            distinct_users=("user_id", "nunique"),
        )
        .reset_index()
    )
    popularity = popularity.merge(
        items[["item_id", "title", "primary_genre", "release_date"]], on="item_id", how="left"
    )
    popularity["popularity_rank"] = (
        popularity["interaction_count"].rank(method="first", ascending=False).astype("int64")
    )
    return popularity.sort_values("popularity_rank").reset_index(drop=True)


def _user_item_matrix(train: pd.DataFrame, processed_dir) -> Dict[str, Any]:
    user_ids = np.sort(train["user_id"].unique())
    item_ids = np.sort(train["item_id"].unique())
    user_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    item_index = {int(item_id): index for index, item_id in enumerate(item_ids)}
    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["item_id"].map(item_index).to_numpy()
    values = train["rating"].to_numpy(dtype=float)
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    sparse.save_npz(processed_dir / "user_item_matrix.npz", matrix)
    pd.DataFrame({"user_id": user_ids, "user_index": range(len(user_ids))}).to_csv(
        processed_dir / "user_index.csv", index=False
    )
    pd.DataFrame({"item_id": item_ids, "item_index": range(len(item_ids))}).to_csv(
        processed_dir / "item_index.csv", index=False
    )
    return {
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "user_index_file": str(processed_dir / "user_index.csv"),
        "item_index_file": str(processed_dir / "item_index.csv"),
        "matrix_file": str(processed_dir / "user_item_matrix.npz"),
    }


def create_derived_datasets(
    train: pd.DataFrame,
    items: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create and save derived datasets using train interactions only."""
    processed_dir = project_path(config, "data_processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    user_activity = _user_activity_summary(train, items)
    item_popularity = _item_popularity(train, items)
    matrix_metadata = _user_item_matrix(train, processed_dir)

    user_activity.to_parquet(processed_dir / "user_activity_summary.parquet", index=False)
    item_popularity.to_parquet(processed_dir / "item_popularity.parquet", index=False)

    metadata = {
        "train_rows": int(len(train)),
        "user_activity_rows": int(len(user_activity)),
        "item_popularity_rows": int(len(item_popularity)),
        "user_item_matrix": matrix_metadata,
        "source": "train split only (no leakage)",
    }
    write_json(processed_dir / "derived_metadata.json", metadata)
    LOGGER.info(
        "Created derived datasets: user_activity=%d item_popularity=%d matrix_shape=%s nnz=%d",
        len(user_activity),
        len(item_popularity),
        matrix_metadata["shape"],
        matrix_metadata["nnz"],
    )
    return {
        "user_activity_summary": user_activity,
        "item_popularity": item_popularity,
        "matrix_metadata": matrix_metadata,
        "metadata": metadata,
    }
