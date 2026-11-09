"""Hand-testable ranking, coverage, diversity, and rating metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.ingestion.synthetic import GENRES


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of relevant items recovered in the top-k recommendations."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(recommended[:k]) & relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@K."""
    relevant_set = set(relevant)
    if not relevant_set or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def catalog_coverage(recommendations: Dict[int, List[int]], catalog_size: int) -> float:
    """Unique recommended items divided by the training catalog size."""
    if catalog_size <= 0:
        return 0.0
    unique_items = {item_id for values in recommendations.values() for item_id in values}
    return len(unique_items) / float(catalog_size)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return 1.0 - max(-1.0, min(1.0, similarity))


def intra_list_diversity(
    recommendations: Dict[int, List[int]], genre_vectors: Dict[int, np.ndarray]
) -> float:
    """Average pairwise cosine distance between items in each recommendation list.

    Lists with fewer than two known item vectors contribute 0; the final score is
    the mean over users with at least one valid list.
    """
    scores: List[float] = []
    for items in recommendations.values():
        vectors = [genre_vectors[item_id] for item_id in items if item_id in genre_vectors]
        if len(vectors) < 2:
            continue
        distances = [
            _cosine_distance(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        scores.append(float(np.mean(distances)))
    return float(np.mean(scores)) if scores else 0.0


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error."""
    if not actual:
        return 0.0
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error."""
    if not actual:
        return 0.0
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(errors)))


def genre_vectors(items: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Build one genre vector per item from the normalised genre list."""
    vectors: Dict[int, np.ndarray] = {}
    for row in items.itertuples(index=False):
        genres = getattr(row, "genres", [])
        if not isinstance(genres, (list, tuple, np.ndarray)):
            genres = []
        present = set(str(value) for value in genres)
        vectors[int(row.item_id)] = np.array([1.0 if genre in present else 0.0 for genre in GENRES])
    return vectors
