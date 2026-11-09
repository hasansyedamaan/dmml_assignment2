"""Popularity-based baseline recommender."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.training.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend the most popular unseen items, ranked by training frequency."""

    name = "popularity"

    def __init__(self) -> None:
        self.item_popularity: pd.Series | None = None
        self.item_means: pd.Series | None = None
        self.global_mean: float = 0.0
        self.seen: Dict[int, set[int]] = {}
        self.catalog: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        counts = train.groupby("item_id").size().sort_values(ascending=False)
        # Stable tie-break by item id.
        counts = counts.sort_index().sort_values(ascending=False, kind="stable")
        self.item_popularity = counts
        self.item_means = train.groupby("item_id")["rating"].mean()
        self.global_mean = float(train["rating"].mean())
        self.catalog = counts.index.to_numpy()
        self.seen = train.groupby("user_id")["item_id"].apply(lambda values: set(values)).to_dict()
        return self

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        if self.item_popularity is None:
            raise RuntimeError("PopularityRecommender must be fitted first.")
        seen = self.seen.get(int(user_id), set()) if exclude_seen else set()
        recommendations: List[int] = []
        for item_id in self.item_popularity.index:
            if int(item_id) in seen:
                continue
            recommendations.append(int(item_id))
            if len(recommendations) >= k:
                break
        return recommendations

    def predict(self, user_id: int, item_id: int) -> float:
        del user_id
        if self.item_means is None:
            raise RuntimeError("PopularityRecommender must be fitted first.")
        return float(self.item_means.get(int(item_id), self.global_mean))
