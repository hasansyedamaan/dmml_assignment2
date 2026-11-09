"""Single source of truth for the feature catalog.

The markdown and CSV catalog are generated from this registry so documentation
cannot drift from the implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

FEATURE_REGISTRY: List[Dict[str, str]] = [
    {
        "feature": "avg_rating",
        "entity": "user",
        "description": "Mean rating given by the user in the training window.",
        "formula": "mean(rating) grouped by user_id",
        "dtype": "float",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "activity_count",
        "entity": "user",
        "description": "Number of training interactions for the user.",
        "formula": "count(interaction) grouped by user_id",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "preferred_category",
        "entity": "user",
        "description": "Most frequent item genre in the user's training history.",
        "formula": "mode(primary_genre) grouped by user_id",
        "dtype": "string",
        "source_table": "train interactions + items",
        "used_by": "training, prediction",
    },
    {
        "feature": "days_since_last_interaction",
        "entity": "user",
        "description": "Days between the as-of date and the user's latest training interaction.",
        "formula": "(as_of_date - max(timestamp)) / 1 day",
        "dtype": "float",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "rating_std",
        "entity": "user",
        "description": "Standard deviation of the user's training ratings (0 for one rating).",
        "formula": "std(rating) grouped by user_id",
        "dtype": "float",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "distinct_items_count",
        "entity": "user",
        "description": "Number of distinct items the user rated in training.",
        "formula": "nunique(item_id) grouped by user_id",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "popularity",
        "entity": "item",
        "description": "Number of training interactions for the item.",
        "formula": "count(interaction) grouped by item_id",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "avg_rating",
        "entity": "item",
        "description": "Mean rating received by the item in training.",
        "formula": "mean(rating) grouped by item_id",
        "dtype": "float",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "rating_count",
        "entity": "item",
        "description": "Number of ratings received by the item in training.",
        "formula": "count(rating) grouped by item_id",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "category",
        "entity": "item",
        "description": "Primary genre of the item.",
        "formula": "first genre in the item's genre list",
        "dtype": "string",
        "source_table": "items",
        "used_by": "training, prediction",
    },
    {
        "feature": "genre_*",
        "entity": "item",
        "description": "One-hot genre indicators for the item.",
        "formula": "1 if the item belongs to the genre, else 0",
        "dtype": "int (one-hot)",
        "source_table": "items",
        "used_by": "training, prediction, diversity",
    },
    {
        "feature": "item_age_years",
        "entity": "item",
        "description": "Age of the item at the as-of date, when a release date is known.",
        "formula": "as_of_year - release_year",
        "dtype": "float",
        "source_table": "items",
        "used_by": "training, prediction",
    },
    {
        "feature": "rating",
        "entity": "interaction",
        "description": "Observed explicit rating (missing at prediction time).",
        "formula": "raw rating",
        "dtype": "float",
        "source_table": "train interactions",
        "used_by": "training",
    },
    {
        "feature": "timestamp",
        "entity": "interaction",
        "description": "UTC timestamp of the interaction.",
        "formula": "unix timestamp converted to UTC",
        "dtype": "datetime",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "hour",
        "entity": "interaction",
        "description": "Hour of day of the interaction.",
        "formula": "timestamp.hour",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "day_of_week",
        "entity": "interaction",
        "description": "Day of week of the interaction (Monday=0).",
        "formula": "timestamp.dayofweek",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
    {
        "feature": "user_item_genre_match",
        "entity": "interaction",
        "description": "Whether the item's category matches the user's preferred category.",
        "formula": "user.preferred_category == item.category",
        "dtype": "bool",
        "source_table": "user and item features",
        "used_by": "training, prediction",
    },
    {
        "feature": "recency_rank",
        "entity": "interaction",
        "description": "Chronological rank of the interaction within the user's history (1=oldest).",
        "formula": "rank(timestamp) grouped by user_id",
        "dtype": "int",
        "source_table": "train interactions",
        "used_by": "training, prediction",
    },
]


def catalog_dataframe() -> pd.DataFrame:
    """Return the feature registry as a DataFrame."""
    return pd.DataFrame(FEATURE_REGISTRY)


def write_feature_catalog(reports_dir: Path) -> pd.DataFrame:
    """Write the feature catalog as Markdown and CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    catalog = catalog_dataframe()
    catalog.to_csv(reports_dir / "feature_catalog.csv", index=False)

    lines = [
        "# Feature Catalog",
        "",
        "Generated from `src/features/registry.py`. The same feature code path "
        "(`FeatureStore`) serves both training and prediction.",
        "",
        "| Feature | Entity | Description | Formula / logic | Data type | Source table | Used by |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in catalog.to_dict(orient="records"):
        lines.append(
            "| {feature} | {entity} | {description} | {formula} | {dtype} | {source_table} | {used_by} |".format(
                **row
            )
        )
    lines.append("")
    (reports_dir / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")
    return catalog
