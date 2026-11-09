"""Interface tests for the two required recommenders."""

from __future__ import annotations

import pandas as pd
import pytest

from src.training.matrix_factorization import MatrixFactorizationRecommender
from src.training.popularity import PopularityRecommender


def _train() -> pd.DataFrame:
    rows = []
    for user_id in range(1, 6):
        for item_id in range(1, 21):
            if (user_id + item_id) % 4 != 0:
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": float(((user_id + item_id) % 5) + 1),
                        "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                        + pd.Timedelta(days=user_id * 20 + item_id),
                    }
                )
    return pd.DataFrame(rows)


def test_popularity_interface() -> None:
    train = _train()
    model = PopularityRecommender().fit(train)
    recommendations = model.recommend(1, k=5, exclude_seen=True)
    assert len(recommendations) == 5
    seen = set(train.loc[train["user_id"] == 1, "item_id"])
    assert not seen.intersection(recommendations)
    assert 1.0 <= model.predict(1, 1) <= 5.0


def test_matrix_factorization_interface() -> None:
    train = _train()
    model = MatrixFactorizationRecommender(n_components=4, n_iter=5, random_state=42).fit(train)
    recommendations = model.recommend(2, k=5, exclude_seen=True)
    assert len(recommendations) == 5
    seen = set(train.loc[train["user_id"] == 2, "item_id"])
    assert not seen.intersection(recommendations)
    prediction = model.predict(2, 1)
    assert 1.0 <= prediction <= 5.0
    # Cold-start user falls back to popularity rather than failing.
    assert len(model.recommend(999, k=5)) == 5


def test_exclude_seen_false_can_return_seen_items() -> None:
    train = _train()
    model = PopularityRecommender().fit(train)
    assert set(model.recommend(1, k=5, exclude_seen=False)).issubset(
        set(train["item_id"].unique())
    )
