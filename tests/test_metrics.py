"""Tiny hand-computed metric examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_precision_at_k_hand_computed() -> None:
    assert precision_at_k([1, 2, 3], {2, 4}, 2) == pytest.approx(0.5)
    assert precision_at_k([1, 2, 3], {2, 4}, 3) == pytest.approx(1.0 / 3.0)


def test_recall_at_k_hand_computed() -> None:
    assert recall_at_k([1, 2, 3], {2, 4}, 2) == pytest.approx(0.5)
    assert recall_at_k([1, 2, 3], {2, 4}, 1) == pytest.approx(0.0)


def test_ndcg_at_k_hand_computed() -> None:
    # One relevant item at rank 2: DCG = 1/log2(3), IDCG = 1.
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k([1, 2, 3], {2}, 2) == pytest.approx(expected)
    # Relevant items at ranks 1 and 2: DCG = 1 + 1/log2(3), IDCG = same.
    assert ndcg_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)
    # Relevant items at ranks 1 and 3: DCG = 1.5, IDCG = 1 + 1/log2(3).
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k([1, 2, 3], {1, 3}, 3) == pytest.approx(expected)


def test_ndcg_empty_relevant_is_zero() -> None:
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0


def test_coverage_hand_computed() -> None:
    recommendations = {1: [10, 20], 2: [20, 30]}
    assert catalog_coverage(recommendations, catalog_size=4) == pytest.approx(0.75)
    assert catalog_coverage({}, catalog_size=4) == 0.0


def test_diversity_hand_computed() -> None:
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([1.0, 0.0]),
    }
    assert intra_list_diversity({1: [1, 2]}, vectors) == pytest.approx(1.0)
    assert intra_list_diversity({1: [1, 3]}, vectors) == pytest.approx(0.0)
    # A list with only one known vector contributes no score.
    assert intra_list_diversity({1: [1]}, vectors) == 0.0


def test_rmse_and_mae_hand_computed() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 4.0, 3.0]
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(4.0 / 3.0))
    assert mae(actual, predicted) == pytest.approx(2.0 / 3.0)
    assert rmse([], []) == 0.0
    assert mae([], []) == 0.0
