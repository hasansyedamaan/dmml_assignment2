"""Reusable train/prediction feature layer."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.features.registry import write_feature_catalog
from src.ingestion.synthetic import GENRES
from src.utils.config import project_path, write_json

LOGGER = logging.getLogger(__name__)


def _sanitize_genre(genre: str) -> str:
    return "genre_" + re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")


class FeatureStore:
    """Build offline features once and serve the same tables for prediction."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_dir = project_path(config, "data_features")
        self.reports_dir = project_path(config, "reports")
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.user_features_path = self.features_dir / "user_features.parquet"
        self.item_features_path = self.features_dir / "item_features.parquet"
        self.interaction_features_path = self.features_dir / "interaction_features.parquet"
        self.metadata_path = self.features_dir / "feature_metadata.json"

    # ------------------------------------------------------------------ build
    def _resolve_as_of(self, train: pd.DataFrame) -> pd.Timestamp:
        configured = self.config["features"].get("as_of_date", "train_max")
        train_max = pd.Timestamp(train["timestamp"].max())
        if train_max.tzinfo is None:
            train_max = train_max.tz_localize("UTC")
        if configured in (None, "", "train_max"):
            return train_max
        as_of = pd.Timestamp(configured)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        return as_of

    def _build_user_features(
        self, train: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        users = (
            train.groupby("user_id")
            .agg(
                avg_rating=("rating", "mean"),
                activity_count=("item_id", "size"),
                rating_std=("rating", "std"),
                distinct_items_count=("item_id", "nunique"),
                last_interaction_time=("timestamp", "max"),
            )
            .reset_index()
        )
        users["rating_std"] = users["rating_std"].fillna(0.0)
        users["days_since_last_interaction"] = (
            (as_of - users["last_interaction_time"]).dt.total_seconds() / 86400.0
        ).clip(lower=0.0)
        users = users.drop(columns=["last_interaction_time"])

        genre_counts = (
            train.merge(items[["item_id", "primary_genre"]], on="item_id", how="left")
            .groupby(["user_id", "primary_genre"])
            .size()
            .reset_index(name="genre_count")
            .sort_values(["user_id", "genre_count", "primary_genre"], ascending=[True, False, True])
            .drop_duplicates("user_id")
        )
        users = users.merge(
            genre_counts[["user_id", "primary_genre"]].rename(
                columns={"primary_genre": "preferred_category"}
            ),
            on="user_id",
            how="left",
        )
        users["preferred_category"] = users["preferred_category"].fillna("unknown")
        return users[
            [
                "user_id",
                "avg_rating",
                "activity_count",
                "preferred_category",
                "days_since_last_interaction",
                "rating_std",
                "distinct_items_count",
            ]
        ]

    def _genre_matrix(self, items: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, int]] = []
        for _, item in items.iterrows():
            genres = item.get("genres")
            if not isinstance(genres, (list, tuple, np.ndarray)):
                genres = []
            present = {str(genre) for genre in genres}
            row = {_sanitize_genre(genre): int(genre in present) for genre in GENRES}
            rows.append(row)
        return pd.DataFrame(rows, index=items.index)

    def _build_item_features(
        self, train: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        popularity = (
            train.groupby("item_id")
            .agg(
                popularity=("user_id", "size"),
                avg_rating=("rating", "mean"),
                rating_count=("rating", "size"),
            )
            .reset_index()
        )
        base = items[["item_id", "primary_genre", "release_date"]].rename(
            columns={"primary_genre": "category"}
        ).copy()
        features = base.merge(popularity, on="item_id", how="left")
        features["popularity"] = features["popularity"].fillna(0).astype("int64")
        features["rating_count"] = features["rating_count"].fillna(0).astype("int64")
        features["avg_rating"] = features["avg_rating"].fillna(float(train["rating"].mean()))
        features["category"] = features["category"].fillna("unknown")
        features["release_year"] = pd.to_datetime(features["release_date"], errors="coerce").dt.year
        features["item_age_years"] = as_of.year - features["release_year"]
        features["item_age_years"] = (
            features["item_age_years"]
            .fillna(features["item_age_years"].median())
            .fillna(0)
            .astype("int64")
        )
        features = features.drop(columns=["release_date", "release_year"])

        genre_matrix = self._genre_matrix(items)
        genre_matrix.insert(0, "item_id", items["item_id"].to_numpy())
        features = features.merge(genre_matrix, on="item_id", how="left")
        genre_columns = [column for column in genre_matrix.columns if column != "item_id"]
        features[genre_columns] = features[genre_columns].fillna(0).astype("int64")

        ordered = ["item_id", "popularity", "avg_rating", "rating_count", "category", "item_age_years"] + genre_columns
        return features[ordered]

    def _build_interaction_features(
        self, train: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame
    ) -> pd.DataFrame:
        interactions = train.copy()
        interactions["hour"] = interactions["timestamp"].dt.hour.astype("int64")
        interactions["day_of_week"] = interactions["timestamp"].dt.day_of_week.astype("int64")
        interactions["recency_rank"] = (
            interactions.groupby("user_id")["timestamp"].rank(method="first").astype("int64")
        )
        interactions = interactions.merge(
            user_features[["user_id", "preferred_category"]], on="user_id", how="left"
        ).merge(
            item_features[["item_id", "category"]], on="item_id", how="left"
        )
        interactions["preferred_category"] = interactions["preferred_category"].fillna("unknown")
        interactions["category"] = interactions["category"].fillna("unknown")
        interactions["user_item_genre_match"] = (
            interactions["preferred_category"] == interactions["category"]
        )
        return interactions[
            [
                "user_id",
                "item_id",
                "rating",
                "timestamp",
                "hour",
                "day_of_week",
                "user_item_genre_match",
                "recency_rank",
            ]
        ]

    def build_features(
        self, train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build user, item, and interaction features from train data only."""
        del users  # User demographics are available, but the required feature set is behavioural.
        as_of = self._resolve_as_of(train)
        user_features = self._build_user_features(train, items, as_of)
        item_features = self._build_item_features(train, items, as_of)
        interaction_features = self._build_interaction_features(train, user_features, item_features)

        user_features.to_parquet(self.user_features_path, index=False)
        item_features.to_parquet(self.item_features_path, index=False)
        interaction_features.to_parquet(self.interaction_features_path, index=False)

        global_mean = float(train["rating"].mean())
        defaults = {
            "user": {
                "avg_rating": global_mean,
                "activity_count": 0,
                "preferred_category": "unknown",
                "days_since_last_interaction": 9999.0,
                "rating_std": 0.0,
                "distinct_items_count": 0,
            },
            "item": {
                "popularity": 0,
                "avg_rating": global_mean,
                "rating_count": 0,
                "category": "unknown",
                "item_age_years": None,
            },
        }
        metadata = {
            "as_of_date": str(as_of),
            "as_of_source": self.config["features"].get("as_of_date", "train_max"),
            "train_max_timestamp": str(train["timestamp"].max()),
            "train_rows": int(len(train)),
            "user_feature_rows": int(len(user_features)),
            "item_feature_rows": int(len(item_features)),
            "interaction_feature_rows": int(len(interaction_features)),
            "defaults": defaults,
            "leakage_guard": "All features are computed from the train split only.",
        }
        write_json(self.metadata_path, metadata)
        write_feature_catalog(self.reports_dir)
        LOGGER.info(
            "Built features as of %s: users=%d items=%d interactions=%d",
            as_of,
            len(user_features),
            len(item_features),
            len(interaction_features),
        )
        return {
            "user_features": user_features,
            "item_features": item_features,
            "interaction_features": interaction_features,
            "metadata": metadata,
        }

    # -------------------------------------------------------------- serve path
    def _default_user_row(self, user_id: int, defaults: Dict[str, Any]) -> Dict[str, Any]:
        row = {"user_id": user_id, **defaults["user"]}
        return row

    def _default_item_row(self, item_id: int, defaults: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
        row = {"item_id": item_id, **defaults["item"]}
        for column in columns:
            if column.startswith("genre_"):
                row[column] = 0
        return row

    def _metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {"defaults": {"user": {}, "item": {}}}
        import json

        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def get_user_features(self, user_ids: Iterable[int]) -> pd.DataFrame:
        """Return stored user features, with documented defaults for unseen users."""
        if not self.user_features_path.exists():
            raise FileNotFoundError("Call build_features before get_user_features.")
        stored = pd.read_parquet(self.user_features_path).set_index("user_id")
        defaults = self._metadata().get("defaults", {"user": {}})
        rows = []
        for user_id in user_ids:
            if int(user_id) in stored.index:
                row = stored.loc[int(user_id)].to_dict()
                row["user_id"] = int(user_id)
            else:
                row = self._default_user_row(int(user_id), defaults)
            rows.append(row)
        columns = ["user_id"] + [column for column in stored.columns]
        return pd.DataFrame(rows).reindex(columns=columns)

    def get_item_features(self, item_ids: Iterable[int]) -> pd.DataFrame:
        """Return stored item features, with documented defaults for unseen items."""
        if not self.item_features_path.exists():
            raise FileNotFoundError("Call build_features before get_item_features.")
        stored = pd.read_parquet(self.item_features_path).set_index("item_id")
        defaults = self._metadata().get("defaults", {"item": {}})
        rows = []
        for item_id in item_ids:
            if int(item_id) in stored.index:
                row = stored.loc[int(item_id)].to_dict()
                row["item_id"] = int(item_id)
            else:
                row = self._default_item_row(int(item_id), defaults, list(stored.columns))
            rows.append(row)
        columns = ["item_id"] + list(stored.columns)
        return pd.DataFrame(rows).reindex(columns=columns)

    def get_interaction_features(
        self,
        user_ids: Iterable[int],
        item_ids: Iterable[int],
        timestamps: Iterable[Any] | None = None,
        ratings: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        """Build interaction features for training or prediction using stored user/item features."""
        user_ids = list(user_ids)
        item_ids = list(item_ids)
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have the same length.")
        if timestamps is None:
            timestamps = [pd.Timestamp.now(tz="UTC")] * len(user_ids)
        timestamps = [pd.Timestamp(value) for value in timestamps]
        timestamps = [
            value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
            for value in timestamps
        ]
        if ratings is None:
            ratings = [np.nan] * len(user_ids)

        user_features = self.get_user_features(user_ids)
        item_features = self.get_item_features(item_ids)
        frame = pd.DataFrame(
            {
                "user_id": user_ids,
                "item_id": item_ids,
                "rating": list(ratings),
                "timestamp": timestamps,
            }
        )
        frame = frame.merge(
            user_features[["user_id", "preferred_category", "activity_count"]],
            on="user_id",
            how="left",
        ).merge(
            item_features[["item_id", "category"]], on="item_id", how="left"
        )
        frame["preferred_category"] = frame["preferred_category"].fillna("unknown")
        frame["category"] = frame["category"].fillna("unknown")
        frame["hour"] = frame["timestamp"].dt.hour.astype("int64")
        frame["day_of_week"] = frame["timestamp"].dt.day_of_week.astype("int64")
        frame["user_item_genre_match"] = frame["preferred_category"] == frame["category"]
        frame["recency_rank"] = frame["activity_count"].fillna(0).astype("int64") + 1
        return frame[
            [
                "user_id",
                "item_id",
                "rating",
                "timestamp",
                "hour",
                "day_of_week",
                "user_item_genre_match",
                "recency_rank",
            ]
        ]
