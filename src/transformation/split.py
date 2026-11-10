"""Leakage-safe time-based train/test splitting.

The default is a per-user last-fraction split: each user's interactions are
ordered by timestamp and the final ``test_fraction`` becomes test data. This
avoids the severe cold-start-user problem of a single global cutoff on
MovieLens 100K. A global-quantile mode is also available for comparison.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.config import project_path, write_json

LOGGER = logging.getLogger(__name__)


def _per_user_split(
    interactions: pd.DataFrame, fraction: float, min_test_interactions: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    ordered = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    cutoffs: List[Dict[str, Any]] = []
    for user_id, group in ordered.groupby("user_id", sort=True):
        count = len(group)
        if count < 2:
            train_parts.append(group)
            continue
        requested = max(int(min_test_interactions), int(math.ceil(count * fraction)))
        test_count = min(max(1, requested), count - 1)
        train_count = count - test_count
        train_parts.append(group.iloc[:train_count])
        test_parts.append(group.iloc[train_count:])
        cutoffs.append(
            {
                "user_id": int(user_id),
                "interaction_count": int(count),
                "train_count": int(train_count),
                "test_count": int(test_count),
                "train_cutoff": str(group.iloc[train_count - 1]["timestamp"]),
                "test_start": str(group.iloc[train_count]["timestamp"]),
            }
        )
    train = pd.concat(train_parts, ignore_index=True) if train_parts else ordered.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ordered.iloc[0:0].copy()
    return train, test, cutoffs


def _global_split(
    interactions: pd.DataFrame, quantile: float
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    cutoff = interactions["timestamp"].quantile(quantile)
    train = interactions[interactions["timestamp"] < cutoff].copy()
    test = interactions[interactions["timestamp"] >= cutoff].copy()
    if train.empty or test.empty:
        ordered = interactions.sort_values("timestamp").reset_index(drop=True)
        split_index = max(1, min(len(ordered) - 1, int(len(ordered) * quantile)))
        train = ordered.iloc[:split_index].copy()
        test = ordered.iloc[split_index:].copy()
        cutoff = test["timestamp"].min()
    cutoffs = [
        {
            "user_id": int(user_id),
            "train_cutoff": str(group[group["timestamp"] < cutoff]["timestamp"].max()),
            "test_start": str(group[group["timestamp"] >= cutoff]["timestamp"].min()),
        }
        for user_id, group in interactions.groupby("user_id")
    ]
    return train, test, cutoffs


def time_based_split(
    interactions: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split by time, never randomly.

    Per-user mode guarantees that each test interaction is at or after that
    user's training cutoff. Global mode guarantees every test interaction is at
    or after one global cutoff.
    """
    if interactions.empty:
        raise ValueError("Cannot split an empty interaction table.")

    split_cfg = config["split"]
    method = split_cfg.get("method", "per_user_last_fraction")
    if method == "global_time_quantile":
        train, test, cutoffs = _global_split(interactions, float(split_cfg["quantile"]))
    else:
        train, test, cutoffs = _per_user_split(
            interactions,
            float(split_cfg.get("test_fraction", 0.2)),
            int(split_cfg.get("min_test_interactions", 1)),
        )

    train_users = set(train["user_id"].unique())
    train_items = set(train["item_id"].unique())
    test["user_seen_in_train"] = test["user_id"].isin(train_users)
    test["item_seen_in_train"] = test["item_id"].isin(train_items)

    cutoff_frame = pd.DataFrame(cutoffs)
    if not cutoff_frame.empty:
        cutoff_frame["train_cutoff_ts"] = pd.to_datetime(cutoff_frame["train_cutoff"], utc=True)
        cutoff_frame["test_start_ts"] = pd.to_datetime(cutoff_frame["test_start"], utc=True)
        valid_cutoffs = cutoff_frame.dropna(subset=["train_cutoff_ts", "test_start_ts"])
        if method == "global_time_quantile":
            # The global split uses one cutoff; per-user cutoffs are descriptive.
            no_test_before_cutoff = bool(test["timestamp"].min() >= train["timestamp"].max())
        else:
            # No test interaction may precede its user's training cutoff.
            no_test_before_cutoff = bool(
                (valid_cutoffs["test_start_ts"] >= valid_cutoffs["train_cutoff_ts"]).all()
            )
        if valid_cutoffs.empty:
            cutoff_min = cutoff_median = cutoff_max = None
        else:
            cutoff_min = str(valid_cutoffs["train_cutoff_ts"].min())
            cutoff_median = str(valid_cutoffs["train_cutoff_ts"].median())
            cutoff_max = str(valid_cutoffs["train_cutoff_ts"].max())
        cutoff_frame = cutoff_frame.drop(columns=["train_cutoff_ts", "test_start_ts"])
    else:
        no_test_before_cutoff = True
        cutoff_min = cutoff_median = cutoff_max = None

    unseen_test_users = test.loc[~test["user_seen_in_train"], "user_id"]
    unseen_test_items = test.loc[~test["item_seen_in_train"], "item_id"]

    processed_dir = project_path(config, "data_processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_dir / "train.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)
    if not cutoff_frame.empty:
        cutoff_frame.to_csv(processed_dir / "split_cutoffs.csv", index=False)

    metadata = {
        "method": method,
        "test_fraction": float(split_cfg.get("test_fraction", 0.2)),
        "min_test_interactions": int(split_cfg.get("min_test_interactions", 1)),
        "quantile": float(split_cfg.get("quantile", 0.8)),
        "cutoff_min": cutoff_min,
        "cutoff_median": cutoff_median,
        "cutoff_max": cutoff_max,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_users": int(train["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "train_items": int(train["item_id"].nunique()),
        "test_items": int(test["item_id"].nunique()),
        "cold_start_test_users": int(unseen_test_users.nunique()),
        "cold_start_test_user_rows": int(len(unseen_test_users)),
        "cold_start_test_items": int(unseen_test_items.nunique()),
        "cold_start_test_item_rows": int(len(unseen_test_items)),
        "train_max_timestamp": str(train["timestamp"].max()),
        "test_min_timestamp": str(test["timestamp"].min()),
        "no_test_before_cutoff": no_test_before_cutoff,
    }
    write_json(processed_dir / "split_metadata.json", metadata)
    LOGGER.info(
        "Time split (%s): train=%d test=%d; cold-start test users=%d items=%d; "
        "no test before per-user cutoff=%s",
        method,
        len(train),
        len(test),
        metadata["cold_start_test_users"],
        metadata["cold_start_test_items"],
        no_test_before_cutoff,
    )
    return train, test, metadata
