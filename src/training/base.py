"""Common recommender interface used by both models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List

import pandas as pd


class BaseRecommender(ABC):
    """Minimal interface shared by the baseline and personalized models."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        """Fit the model on a training interaction table."""

    @abstractmethod
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        """Return the top-k item ids for one user."""

    def recommend_batch(
        self, user_ids: Iterable[int], k: int = 10, exclude_seen: bool = True
    ) -> Dict[int, List[int]]:
        """Return recommendations for multiple users."""
        return {
            int(user_id): self.recommend(int(user_id), k=k, exclude_seen=exclude_seen)
            for user_id in user_ids
        }

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict an explicit rating when the model supports it."""
        raise NotImplementedError(f"{self.name} does not predict explicit ratings.")
