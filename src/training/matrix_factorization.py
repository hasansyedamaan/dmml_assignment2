"""Truncated-SVD matrix-factorization recommender.

This is the scikit-learn/SciPy fallback explicitly allowed by the brief when
implicit/surprise wheels are unavailable. It learns latent user and item factors
from the sparse training rating matrix and supports both ranking and explicit
rating prediction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.training.base import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    """Latent-factor recommender with a popularity fallback for cold-start users."""

    name = "matrix_factorization"

    def __init__(
        self,
        n_components: int = 32,
        n_iter: int = 12,
        random_state: int = 42,
        clip_min: float = 1.0,
        clip_max: float = 5.0,
    ) -> None:
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_index: Dict[int, int] = {}
        self.item_index: Dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.rating_user_factors: np.ndarray | None = None
        self.rating_item_factors: np.ndarray | None = None
        self.seen: Dict[int, set[int]] = {}
        self.popular_items: List[int] = []
        self.item_means: pd.Series | None = None
        self.user_bias: pd.Series | None = None
        self.item_bias: pd.Series | None = None
        self.global_mean: float = 0.0
        self.explained_variance: float = 0.0

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationRecommender":
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = np.sort(train["item_id"].unique())
        self.user_index = {int(value): index for index, value in enumerate(self.user_ids)}
        self.item_index = {int(value): index for index, value in enumerate(self.item_ids)}

        rows = train["user_id"].map(self.user_index).to_numpy()
        cols = train["item_id"].map(self.item_index).to_numpy()
        values = train["rating"].to_numpy(dtype=float)

        raw_matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))
        )
        max_components = max(1, min(raw_matrix.shape) - 1)
        components = min(self.n_components, max_components)

        # Ranking uses factors learned from the raw rating matrix: this preserves
        # the strong implicit-feedback-style ordering observed for SVD on ratings.
        ranking_svd = TruncatedSVD(
            n_components=components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.user_factors = ranking_svd.fit_transform(raw_matrix)
        self.item_factors = ranking_svd.components_.T
        self.explained_variance = float(ranking_svd.explained_variance_ratio_.sum())

        # Explicit rating prediction uses a bias-aware residual SVD: global mean
        # plus user/item offsets, then low-rank residual structure. This keeps
        # RMSE/MAE sensible without sacrificing the ranking factors above.
        self.global_mean = float(train["rating"].mean())
        self.user_bias = train.groupby("user_id")["rating"].mean() - self.global_mean
        self.item_bias = train.groupby("item_id")["rating"].mean() - self.global_mean
        user_bias_values = train["user_id"].map(self.user_bias).to_numpy(dtype=float)
        item_bias_values = train["item_id"].map(self.item_bias).to_numpy(dtype=float)
        residuals = values - (self.global_mean + user_bias_values + item_bias_values)
        residual_matrix = sparse.csr_matrix(
            (residuals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))
        )
        rating_svd = TruncatedSVD(
            n_components=components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.rating_user_factors = rating_svd.fit_transform(residual_matrix)
        self.rating_item_factors = rating_svd.components_.T

        self.item_means = train.groupby("item_id")["rating"].mean()
        popularity = train.groupby("item_id").size().sort_values(ascending=False)
        self.popular_items = [int(item_id) for item_id in popularity.index]
        self.seen = train.groupby("user_id")["item_id"].apply(lambda values: set(values)).to_dict()
        return self

    def _scores_for_user(self, user_id: int) -> np.ndarray | None:
        if self.user_factors is None or self.item_factors is None:
            return None
        index = self.user_index.get(int(user_id))
        if index is None:
            return None
        return self.user_factors[index] @ self.item_factors.T

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        if self.item_ids is None:
            raise RuntimeError("MatrixFactorizationRecommender must be fitted first.")
        seen = self.seen.get(int(user_id), set()) if exclude_seen else set()
        scores = self._scores_for_user(user_id)
        if scores is None:
            candidates = [item_id for item_id in self.popular_items if item_id not in seen]
            return candidates[:k]

        order = np.argsort(scores)[::-1]
        recommendations: List[int] = []
        for index in order:
            item_id = int(self.item_ids[index])
            if item_id in seen:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= k:
                break
        return recommendations

    def predict(self, user_id: int, item_id: int) -> float:
        if self.rating_user_factors is None or self.rating_item_factors is None:
            raise RuntimeError("MatrixFactorizationRecommender must be fitted first.")
        user_position = self.user_index.get(int(user_id))
        item_position = self.item_index.get(int(item_id))
        if user_position is None or item_position is None:
            if self.item_means is not None:
                return float(self.item_means.get(int(item_id), self.global_mean))
            return self.global_mean
        user_offset = float(self.user_bias.get(int(user_id), 0.0)) if self.user_bias is not None else 0.0
        item_offset = float(self.item_bias.get(int(item_id), 0.0)) if self.item_bias is not None else 0.0
        score = (
            self.global_mean
            + user_offset
            + item_offset
            + float(self.rating_user_factors[user_position] @ self.rating_item_factors[item_position])
        )
        return float(np.clip(score, self.clip_min, self.clip_max))
