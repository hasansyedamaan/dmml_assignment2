"""Feature-store tests: train-only builds, as-of metadata, and serving defaults."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from src.features import FeatureStore


def _tables() -> Dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": [20, 40],
            "gender": ["M", "F"],
            "occupation": ["student", "writer"],
            "zip": ["11111", "22222"],
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["One", "Two", "Three"],
            "release_date": ["01-Jan-1990", "01-Jan-2000", None],
            "genres": [["Action"], ["Comedy"], ["Drama"]],
            "primary_genre": ["Action", "Comedy", "Drama"],
        }
    )
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [1, 2, 1],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-02T11:00:00Z", "2024-01-03T12:00:00Z"],
                utc=True,
            ),
        }
    )
    return {"users": users, "items": items, "train": train}


def test_features_are_built_from_train_only(temp_config: Dict[str, Any]) -> None:
    tables = _tables()
    store = FeatureStore(temp_config)
    result = store.build_features(tables["train"], tables["users"], tables["items"])

    user_features = result["user_features"]
    item_features = result["item_features"]
    interaction_features = result["interaction_features"]

    assert set(user_features["user_id"]) == {1, 2}
    assert len(item_features) == 3
    assert len(interaction_features) == len(tables["train"])
    assert result["metadata"]["leakage_guard"].startswith("All features are computed")
    assert result["metadata"]["as_of_date"] == "2024-01-03 12:00:00+00:00"

    # Item 3 has no training interactions: popularity is zero and avg_rating
    # falls back to the training global mean instead of NaN.
    item3 = item_features.set_index("item_id").loc[3]
    assert item3["popularity"] == 0
    assert item3["rating_count"] == 0
    assert item3["avg_rating"] == pytest.approx(tables["train"]["rating"].mean())
    assert pd.notna(item3["item_age_years"])


def test_feature_catalog_is_generated_from_registry(temp_config: Dict[str, Any]) -> None:
    tables = _tables()
    store = FeatureStore(temp_config)
    store.build_features(tables["train"], tables["users"], tables["items"])

    catalog_csv = temp_config["paths"]["reports"] + "/feature_catalog.csv"
    catalog_md = temp_config["paths"]["reports"] + "/feature_catalog.md"
    from pathlib import Path

    csv_path = Path(temp_config["_project_root"]) / catalog_csv
    md_path = Path(temp_config["_project_root"]) / catalog_md
    assert csv_path.exists()
    assert md_path.exists()
    assert "FeatureStore" in md_path.read_text(encoding="utf-8")


def test_serving_path_uses_documented_defaults(temp_config: Dict[str, Any]) -> None:
    tables = _tables()
    store = FeatureStore(temp_config)
    store.build_features(tables["train"], tables["users"], tables["items"])

    users = store.get_user_features([1, 999])
    assert set(users["user_id"]) == {1, 999}
    unseen_user = users.set_index("user_id").loc[999]
    assert unseen_user["activity_count"] == 0
    assert unseen_user["preferred_category"] == "unknown"

    items = store.get_item_features([1, 999])
    assert set(items["item_id"]) == {1, 999}
    unseen_item = items.set_index("item_id").loc[999]
    assert unseen_item["popularity"] == 0
    assert unseen_item["avg_rating"] == pytest.approx(tables["train"]["rating"].mean())

    interactions = store.get_interaction_features([1, 999], [2, 999])
    assert len(interactions) == 2
    assert "user_item_genre_match" in interactions.columns
    assert "recency_rank" in interactions.columns
