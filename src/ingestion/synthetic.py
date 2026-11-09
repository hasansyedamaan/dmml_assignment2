"""Synthetic MovieLens-compatible data generator.

This is the documented offline fallback. It produces the same logical tables as
MovieLens 100K: users, items with genre indicators, and timestamped ratings with
a realistic popularity skew.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

GENRES: List[str] = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

OCCUPATIONS: List[str] = [
    "student",
    "educator",
    "engineer",
    "programmer",
    "writer",
    "artist",
    "administrator",
    "scientist",
    "other",
]


def generate_synthetic_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate users, items, and interactions with a long-tail popularity skew."""
    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed)
    dataset_cfg = config["dataset"]
    n_users = int(dataset_cfg["synthetic_users"])
    n_items = int(dataset_cfg["synthetic_items"])
    n_interactions = int(dataset_cfg["synthetic_interactions"])
    start = pd.Timestamp(dataset_cfg["synthetic_start"], tz="UTC")
    end = pd.Timestamp(dataset_cfg["synthetic_end"], tz="UTC")

    user_ids = np.arange(1, n_users + 1)
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "age": rng.integers(18, 70, size=n_users),
            "gender": rng.choice(["M", "F"], size=n_users, p=[0.55, 0.45]),
            "occupation": rng.choice(OCCUPATIONS, size=n_users),
            "zip": [f"{rng.integers(10000, 99999):05d}" for _ in range(n_users)],
        }
    )

    item_ids = np.arange(1, n_items + 1)
    primary_genres = rng.choice(GENRES[1:], size=n_items)
    genre_matrix = np.zeros((n_items, len(GENRES)), dtype=int)
    for idx, genre in enumerate(primary_genres):
        genre_matrix[idx, GENRES.index(genre)] = 1
        # Give each item a second genre some of the time.
        if rng.random() < 0.35:
            second = int(rng.integers(1, len(GENRES)))
            genre_matrix[idx, second] = 1
    genre_columns = {f"genre_{i}": genre_matrix[:, i] for i in range(len(GENRES))}
    release_dates = pd.to_datetime("1980-01-01") + pd.to_timedelta(
        rng.integers(0, 45 * 365, size=n_items), unit="D"
    )
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "title": [f"Synthetic Movie {i:04d} ({d.year})" for i, d in zip(item_ids, release_dates)],
            "release_date": release_dates.strftime("%d-%b-%Y"),
            "video_release_date": np.nan,
            "imdb_url": [f"https://example.invalid/item/{i}" for i in item_ids],
            **genre_columns,
        }
    )

    # Long-tail item popularity: rank-based weights with noise.
    popularity = 1.0 / np.power(np.arange(1, n_items + 1), 1.15)
    popularity = popularity[rng.permutation(n_items)]
    popularity = popularity / popularity.sum()
    chosen_items = rng.choice(item_ids, size=n_interactions, p=popularity)
    chosen_users = rng.choice(user_ids, size=n_interactions)
    item_quality = 3.2 + 0.8 * (popularity - popularity.mean()) / (popularity.std() + 1e-9)
    ratings = np.clip(
        np.rint(item_quality[np.searchsorted(item_ids, chosen_items)] + rng.normal(0, 1.0, n_interactions)),
        1,
        5,
    ).astype(int)
    offsets = rng.integers(0, int((end - start).total_seconds()), size=n_interactions)
    timestamps = (start.value // 10**9) + offsets
    interactions = pd.DataFrame(
        {
            "user_id": chosen_users,
            "item_id": chosen_items,
            "rating": ratings,
            "timestamp": timestamps,
        }
    )
    # Avoid exact duplicate interactions while keeping the dataset realistic.
    interactions = interactions.drop_duplicates(["user_id", "item_id", "timestamp"]).reset_index(drop=True)

    items["genres"] = items.apply(
        lambda row: [GENRES[i] for i in range(len(GENRES)) if int(row[f"genre_{i}"]) == 1],
        axis=1,
    )
    items["primary_genre"] = items["genres"].apply(lambda values: values[0] if values else "unknown")
    return users, items, interactions
