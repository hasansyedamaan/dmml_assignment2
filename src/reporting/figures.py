"""Generate architecture and orchestration evidence figures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils.config import project_path

LOGGER = logging.getLogger(__name__)

STAGE_ORDER = [
    "ingest",
    "validate",
    "prepare",
    "transform",
    "features",
    "train",
    "evaluate",
    "report",
]

STAGE_LABELS = {
    "ingest": "Data Ingestion",
    "validate": "Data Validation",
    "prepare": "Data Preparation",
    "transform": "Data Transformation",
    "features": "Feature Management",
    "train": "Model Training",
    "evaluate": "Model Evaluation",
    "report": "Reporting / Orchestration",
}


def write_architecture_mermaid(config: Dict[str, Any]) -> Path:
    """Write the Mermaid source used for the architecture diagram."""
    reports_dir = project_path(config, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "architecture.mmd"
    path.write_text(
        """flowchart TD
    A[Raw CSVs / MovieLens 100K] --> B[Ingestion]
    B --> C[Validation]
    C -->|valid rows| D[Preparation]
    C -->|invalid rows| E[(data/errors)]
    D --> F[Time-based Split]
    F --> G[Train / Test]
    G --> H[Derived Datasets]
    H --> I[(data/processed)]
    G --> J[Feature Store]
    J --> K[(data/features)]
    K --> L[Popularity Baseline]
    K --> M[Matrix Factorization]
    L --> N[Evaluation]
    M --> N
    N --> O[Model Comparison + Figures]
    N --> P[(MLflow ./mlruns)]
    O --> Q[(reports)]
    R[Prefect Flow / run_pipeline.py] -. orchestrates .-> B
    R -. orchestrates .-> C
    R -. orchestrates .-> D
    R -. orchestrates .-> F
    R -. orchestrates .-> J
    R -. orchestrates .-> L
    R -. orchestrates .-> M
    R -. orchestrates .-> N
""",
        encoding="utf-8",
    )
    return path


def _box(ax, x, y, width, height, text, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#475569",
        )
    )


def plot_architecture(config: Dict[str, Any]) -> Path:
    """Render a readable architecture diagram with matplotlib."""
    figures_dir = project_path(config, "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    stage_color = "#dbeafe"
    store_color = "#dcfce7"
    tool_color = "#fef3c7"
    row1 = [
        ("Data\nIngestion", 0.4),
        ("Data\nValidation", 3.7),
        ("Data\nPreparation", 7.0),
        ("Data\nTransformation", 10.3),
    ]
    row2 = [
        ("Feature\nManagement", 0.4),
        ("Model\nTraining", 3.7),
        ("Model\nEvaluation", 7.0),
        ("Reports /\nOrchestration", 10.3),
    ]
    for label, x in row1:
        _box(ax, x, 5.9, 2.8, 1.0, label, stage_color, fontsize=11)
    for label, x in row2:
        _box(ax, x, 3.6, 2.8, 1.0, label, stage_color, fontsize=11)
    for x in [3.2, 6.5, 9.8]:
        _arrow(ax, (x, 6.4), (x + 0.5, 6.4))
    _arrow(ax, (11.7, 5.9), (11.7, 4.6))
    _arrow(ax, (10.3, 4.1), (7.0, 4.1))
    _arrow(ax, (3.7, 4.1), (3.2, 4.1))
    _arrow(ax, (3.2, 4.1), (1.8, 5.9))

    stores = [
        ("data/raw", 0.4, 1.9),
        ("data/processed", 3.4, 1.9),
        ("data/features", 6.4, 1.9),
        ("data/errors", 9.4, 1.9),
    ]
    for label, x, y in stores:
        _box(ax, x, y, 2.2, 0.7, label, store_color, fontsize=10)
        _arrow(ax, (x + 1.1, y + 0.7), (x + 1.1, 3.6))

    _box(ax, 0.4, 0.4, 3.0, 0.8, "MLflow tracking\n./mlruns", tool_color, fontsize=10)
    _box(ax, 3.8, 0.4, 3.2, 0.8, "Prefect flow\n+ run_pipeline.py", tool_color, fontsize=10)
    _box(ax, 7.4, 0.4, 3.0, 0.8, "Models\nmodels/*.joblib", tool_color, fontsize=10)
    _box(ax, 10.8, 0.4, 2.8, 0.8, "Reports\nreports/*", tool_color, fontsize=10)
    _arrow(ax, (5.4, 1.2), (5.4, 3.6))
    _arrow(ax, (8.9, 1.2), (8.4, 3.6))

    ax.text(0.4, 7.65, "End-to-End Recommendation Data Management Pipeline", fontsize=16, weight="bold")
    ax.text(
        0.4,
        7.35,
        "Stages, data stores, feature layer, models, tracking, and orchestration",
        fontsize=10,
        color="#475569",
    )
    ax.text(
        13.6,
        0.1,
        "Syed Amaan Hasan | 2025AG05941",
        ha="right",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    path = figures_dir / "architecture.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_stage_statuses(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    status_dir = project_path(config, "stage_status")
    statuses = []
    for stage in STAGE_ORDER:
        path = status_dir / f"{stage}.json"
        if path.exists():
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
    return statuses


def plot_orchestration(config: Dict[str, Any]) -> Path:
    """Render the actual run's stage states and durations.

    This is explicitly labelled as generated from the real run, not as a Prefect
    UI screenshot.
    """
    figures_dir = project_path(config, "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    statuses = _load_stage_statuses(config)
    if not statuses:
        statuses = [
            {"stage": stage, "status": "UNKNOWN", "duration_seconds": 0.0}
            for stage in STAGE_ORDER
        ]
    labels = [STAGE_LABELS.get(item["stage"], item["stage"]) for item in statuses]
    durations = [float(item.get("duration_seconds", 0.0)) for item in statuses]
    colors = ["#16a34a" if item.get("status") == "SUCCESS" else "#dc2626" for item in statuses]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = range(len(labels))
    ax.barh(list(y), durations, color=colors, alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Duration (seconds)")
    ax.set_title(
        "Orchestration Run Evidence\n"
        "Generated from the actual pipeline run — not a Prefect UI screenshot",
        fontsize=13,
    )
    for index, (duration, status) in enumerate(zip(durations, statuses)):
        ax.text(
            duration,
            index,
            f"  {status.get('status', 'UNKNOWN')} ({duration:.1f}s)",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "orchestration_success.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote orchestration evidence to %s", path)
    return path
