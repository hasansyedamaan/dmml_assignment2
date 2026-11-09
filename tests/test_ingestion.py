"""Ingestion and deterministic fault-injection tests."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.ingestion.ingest import inject_faults
from src.ingestion.synthetic import generate_synthetic_data


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame(
        {
            "user_id": range(1, 5),
            "age": [20, 30, 40, 50],
            "gender": ["M", "F", "M", "F"],
            "occupation": ["a", "b", "c", "d"],
            "zip": ["1", "2", "3", "4"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": range(1, 6),
            "title": [f"Item {index}" for index in range(1, 6)],
            "release_date": ["01-Jan-1995"] * 5,
            "genres": [["Action"]] * 5,
            "primary_genre": ["Action"] * 5,
        }
    )
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "item_id": [1, 2, 1, 3, 2, 4, 3, 5, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "rating": [5.0, 4.0, 3.0, 4.0, 2.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "timestamp": list(range(800000000, 800000020)),
        }
    )
    return users, items, interactions


def test_fault_injection_adds_every_fault_class() -> None:
    users, items, interactions = _tables()
    faulty_users, faulty_items, faulty_interactions = inject_faults(users, items, interactions)

    assert len(faulty_users) == len(users) + 2
    assert len(faulty_items) == len(items) + 2
    assert len(faulty_interactions) == len(interactions) + 1
    assert faulty_interactions["user_id"].astype(str).str.contains("not-an-integer").any()
    assert faulty_interactions["timestamp"].astype(str).str.contains("not-a-timestamp").any()
    assert (faulty_interactions["rating"] == 9).any()
    assert (faulty_interactions["rating"] == -1).any()
    assert (faulty_interactions["timestamp"] == 0).any()
    assert (faulty_interactions["timestamp"] == 2524608000).any()
    assert faulty_interactions.duplicated(subset=["user_id", "item_id", "timestamp"]).any()


def test_synthetic_fallback_is_deterministic() -> None:
    config = {
        "project": {"seed": 7},
        "dataset": {
            "synthetic_users": 20,
            "synthetic_items": 15,
            "synthetic_interactions": 200,
            "synthetic_start": "2023-01-01",
            "synthetic_end": "2024-12-31",
        },
    }
    first = generate_synthetic_data(config)
    second = generate_synthetic_data(config)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
    assert len(first[0]) == 20
    assert len(first[1]) == 15
    assert len(first[2]) == 200
