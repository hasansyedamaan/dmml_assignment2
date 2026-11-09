"""Validation and quarantine tests covering every required fault class."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.validation import validate_data


def _users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "age": [25, 31, 44],
            "gender": ["M", "F", "M"],
            "occupation": ["student", "writer", "engineer"],
            "zip": ["11111", "22222", "33333"],
        }
    )


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["One", "Two", "Three"],
            "release_date": ["01-Jan-1995", "01-Jan-1996", "01-Jan-1997"],
            "genres": [["Action"], ["Comedy"], ["Drama"]],
            "primary_genre": ["Action", "Comedy", "Drama"],
        }
    )


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 3.0, 4.0, 2.0, 5.0],
            "timestamp": [800000000, 800000100, 800000200, 800000300, 800000400, 800000500],
        }
    )


def _faulty_tables() -> Dict[str, pd.DataFrame]:
    users = _users()
    items = _items()
    interactions = _interactions()

    missing_user = users.iloc[[0]].copy()
    missing_user["user_id"] = 999998
    missing_user["gender"] = None
    users = pd.concat([users, missing_user, users.iloc[[1]]], ignore_index=True)

    missing_item = items.iloc[[0]].copy()
    missing_item["item_id"] = 999998
    missing_item["title"] = None
    items = pd.concat([items, missing_item, items.iloc[[1]]], ignore_index=True)

    faulty_rows = []
    missing_rating = interactions.iloc[[0]].copy()
    missing_rating["rating"] = None
    faulty_rows.append(missing_rating)

    rating_high = interactions.iloc[[1]].copy()
    rating_high["rating"] = 9
    faulty_rows.append(rating_high)

    rating_low = interactions.iloc[[2]].copy()
    rating_low["rating"] = -1
    faulty_rows.append(rating_low)

    future = interactions.iloc[[3]].copy()
    future["timestamp"] = 2524608000  # 2050-01-01
    faulty_rows.append(future)

    too_old = interactions.iloc[[4]].copy()
    too_old["timestamp"] = 0
    faulty_rows.append(too_old)

    wrong_dtype = interactions.iloc[[5]].copy()
    wrong_dtype["user_id"] = "not-an-integer"
    faulty_rows.append(wrong_dtype)

    unparseable = interactions.iloc[[0]].copy()
    unparseable["timestamp"] = "not-a-timestamp"
    faulty_rows.append(unparseable)

    orphan_user = interactions.iloc[[0]].copy()
    orphan_user["user_id"] = 999999
    faulty_rows.append(orphan_user)

    orphan_item = interactions.iloc[[1]].copy()
    orphan_item["item_id"] = 999999
    faulty_rows.append(orphan_item)

    faulty_rows.append(interactions.iloc[[2]].copy())  # exact duplicate

    interactions = pd.concat([interactions] + faulty_rows, ignore_index=True)
    return {"users": users, "items": items, "interactions": interactions}


def test_validation_quarantines_every_required_fault(temp_config: Dict[str, Any]) -> None:
    tables = _faulty_tables()
    result = validate_data(
        tables["users"],
        tables["items"],
        tables["interactions"],
        temp_config,
        faults_injected=True,
    )

    assert result["report"]["overall_status"] == "FAULTS_DETECTED"
    for dataset in ("users", "items", "interactions"):
        errors = result["errors"][dataset]
        assert "failed_rules" in errors.columns
        assert "dataset" in errors.columns
        assert (errors["dataset"] == dataset).all()

    interaction_rules = set(
        ";".join(result["errors"]["interactions"]["failed_rules"].astype(str)).split(";")
    )
    assert {
        "missing_required_value",
        "duplicate_record",
        "incorrect_dtype",
        "invalid_rating",
        "invalid_timestamp",
        "orphan_user_id",
        "orphan_item_id",
    }.issubset(interaction_rules)

    user_rules = set(
        ";".join(result["errors"]["users"]["failed_rules"].astype(str)).split(";")
    )
    assert {"missing_required_value", "duplicate_record"}.issubset(user_rules)

    item_rules = set(
        ";".join(result["errors"]["items"]["failed_rules"].astype(str)).split(";")
    )
    assert {"missing_required_value", "duplicate_record"}.issubset(item_rules)

    # The six clean source rows continue; every appended fault is quarantined.
    assert len(result["users"]) == 3
    assert len(result["items"]) == 3
    assert len(result["interactions"]) == 6


def test_clean_data_passes_without_quarantine(temp_config: Dict[str, Any]) -> None:
    result = validate_data(_users(), _items(), _interactions(), temp_config, faults_injected=False)
    assert result["report"]["overall_status"] == "PASS"
    assert all(len(result["errors"][name]) == 0 for name in ("users", "items", "interactions"))
    assert result["report"]["datasets"]["interactions"]["rows_valid"] == 6
