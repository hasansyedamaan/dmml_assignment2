"""Evaluate both recommenders and write comparison evidence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    catalog_coverage,
    genre_vectors,
    intra_list_diversity,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from src.utils.config import project_path

LOGGER = logging.getLogger(__name__)


def _metric_bundle(
    model: Any,
    test: pd.DataFrame,
    items: pd.DataFrame,
    config: Dict[str, Any],
    k: int,
) -> Dict[str, Any]:
    threshold = float(config["evaluation"]["relevance_threshold"])
    catalog_size = int(config["_catalog_size"])
    test_users = sorted(int(user_id) for user_id in test["user_id"].unique())
    recommendations = model.recommend_batch(test_users, k=k, exclude_seen=True)

    relevant_by_user = (
        test[test["rating"] >= threshold]
        .groupby("user_id")["item_id"]
        .apply(lambda values: set(int(item_id) for item_id in values))
        .to_dict()
    )
    precision_values: List[float] = []
    recall_values: List[float] = []
    ndcg_values: List[float] = []
    for user_id, relevant in relevant_by_user.items():
        recommended = recommendations.get(int(user_id), [])
        precision_values.append(precision_at_k(recommended, relevant, k))
        recall_values.append(recall_at_k(recommended, relevant, k))
        ndcg_values.append(ndcg_at_k(recommended, relevant, k))

    vectors = genre_vectors(items)
    actual = test["rating"].to_numpy(dtype=float)
    predicted = np.array(
        [model.predict(int(row.user_id), int(row.item_id)) for row in test.itertuples(index=False)],
        dtype=float,
    )

    suffix = f"_at_{k}"
    return {
        f"precision{suffix}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall{suffix}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"ndcg{suffix}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        f"coverage{suffix}": catalog_coverage(recommendations, catalog_size),
        f"diversity{suffix}": intra_list_diversity(recommendations, vectors),
        "rmse": rmse(actual.tolist(), predicted.tolist()),
        "mae": mae(actual.tolist(), predicted.tolist()),
        "evaluated_users": len(test_users),
        "users_with_relevant_items": len(relevant_by_user),
        "catalog_size": catalog_size,
        "k": k,
        "recommendations": recommendations,
    }


def _write_comparison(
    results: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> pd.DataFrame:
    reports_dir = project_path(config, "reports")
    figures_dir = project_path(config, "figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, result in results.items():
        row: Dict[str, Any] = {"model": model_name}
        row.update(result["metrics"])
        row.update(result.get("metrics_secondary", {}))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    metadata_columns = {
        "evaluated_users",
        "users_with_relevant_items",
        "catalog_size",
        "k",
    }
    metric_columns = [
        column for column in comparison.columns if column != "model" and column not in metadata_columns
    ]
    comparison = comparison[["model"] + metric_columns]
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    lines = [
        "# Model Comparison",
        "",
        f"Primary K = {config['models']['k']}; secondary K = {config['models']['secondary_k']}. "
        f"Relevant means a test rating >= {config['evaluation']['relevance_threshold']}.",
        "",
        "Coverage = unique recommended items / training catalog size. "
        "Diversity = mean intra-list pairwise cosine distance between item genre vectors.",
        "",
        "| Model | " + " | ".join(metric_columns) + " |",
        "|---|" + "---:|" * len(metric_columns),
    ]
    for row in comparison.to_dict(orient="records"):
        values = []
        for column in metric_columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Ranking metrics are averaged over test users with at least one relevant item. "
            "Cold-start users receive the popularity fallback; unseen test items are not "
            "recommendable and therefore count as misses.",
            "",
        ]
    )
    (reports_dir / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    plot_metrics = [
        column
        for column in metric_columns
        if column.startswith(("precision", "recall", "ndcg", "coverage", "diversity"))
    ]
    primary = [column for column in plot_metrics if column.endswith(f"_at_{config['models']['k']}")]
    if primary:
        x = np.arange(len(primary))
        width = 0.8 / max(1, len(comparison))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for index, row in comparison.iterrows():
            values = [float(row[column]) for column in primary]
            ax.bar(x + index * width, values, width, label=row["model"])
        ax.set_xticks(x + width * (len(comparison) - 1) / 2)
        ax.set_xticklabels([column.split("_at_")[0] for column in primary])
        ax.set_ylabel("Score")
        ax.set_title(f"Model comparison at K={config['models']['k']}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "model_comparison.png", dpi=160)
        plt.close(fig)
    return comparison


def evaluate_models(
    models: Dict[str, Any],
    test: pd.DataFrame,
    items: pd.DataFrame,
    train: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate every model at the primary and secondary K values."""
    config = dict(config)
    config["_catalog_size"] = int(train["item_id"].nunique())
    primary_k = int(config["models"]["k"])
    secondary_k = int(config["models"]["secondary_k"])

    results: Dict[str, Dict[str, Any]] = {}
    for model_name, model in models.items():
        primary = _metric_bundle(model, test, items, config, primary_k)
        secondary = _metric_bundle(model, test, items, config, secondary_k)
        results[model_name] = {
            "metrics": {key: value for key, value in primary.items() if key != "recommendations"},
            "metrics_secondary": {
                key: value for key, value in secondary.items() if key != "recommendations"
            },
            "recommendations": primary["recommendations"],
        }
        LOGGER.info(
            "Evaluated %s: precision@%d=%.4f recall@%d=%.4f ndcg@%d=%.4f coverage=%.4f diversity=%.4f",
            model_name,
            primary_k,
            results[model_name]["metrics"][f"precision_at_{primary_k}"],
            primary_k,
            results[model_name]["metrics"][f"recall_at_{primary_k}"],
            primary_k,
            results[model_name]["metrics"][f"ndcg_at_{primary_k}"],
            results[model_name]["metrics"][f"coverage_at_{primary_k}"],
            results[model_name]["metrics"][f"diversity_at_{primary_k}"],
        )

    comparison = _write_comparison(results, config)
    results["comparison"] = comparison.to_dict(orient="records")
    return results
