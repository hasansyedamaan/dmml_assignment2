"""Time-based split tests."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from src.transformation.split import time_based_split


def _interactions() -> pd.DataFrame:
    rows = []
    for user_id in (1, 2, 3):
        for offset in range(5):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": user_id * 10 + offset,
                    "rating": 4.0,
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(days=user_id * 10 + offset),
                }
            )
    return pd.DataFrame(rows)


def test_per_user_last_fraction_is_temporal(temp_config: Dict[str, Any]) -> None:
    temp_config["split"]["method"] = "per_user_last_fraction"
    temp_config["split"]["test_fraction"] = 0.4
    temp_config["split"]["min_test_interactions"] = 1
    train, test, metadata = time_based_split(_interactions(), temp_config)

    assert metadata["method"] == "per_user_last_fraction"
    assert metadata["no_test_before_cutoff"] is True
    assert set(train["user_id"]) == {1, 2, 3}
    assert set(test["user_id"]) == {1, 2, 3}
    assert len(train) == 9
    assert len(test) == 6

    for user_id in (1, 2, 3):
        user_train = train[train["user_id"] == user_id].sort_values("timestamp")
        user_test = test[test["user_id"] == user_id].sort_values("timestamp")
        assert len(user_train) == 3
        assert len(user_test) == 2
        assert user_test["timestamp"].min() >= user_train["timestamp"].max()


def test_per_user_split_keeps_all_test_users_warm(temp_config: Dict[str, Any]) -> None:
    temp_config["split"]["method"] = "per_user_last_fraction"
    _, test, metadata = time_based_split(_interactions(), temp_config)
    assert metadata["cold_start_test_users"] == 0
    assert test["user_seen_in_train"].all()


def test_global_quantile_split_has_single_cutoff(temp_config: Dict[str, Any]) -> None:
    temp_config["split"]["method"] = "global_time_quantile"
    temp_config["split"]["quantile"] = 0.6
    train, test, metadata = time_based_split(_interactions(), temp_config)
    assert metadata["method"] == "global_time_quantile"
    assert metadata["no_test_before_cutoff"] is True
    assert test["timestamp"].min() >= train["timestamp"].max()
    assert len(train) + len(test) == 15


def test_empty_interactions_raise(temp_config: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        time_based_split(_interactions().iloc[0:0], temp_config)
