"""Plain-Python CLI for the complete recommendation pipeline.

Examples:
    python run_pipeline.py
    python run_pipeline.py --inject-faults
    python run_pipeline.py --skip-download
"""

from __future__ import annotations

import argparse
import logging
import sys
# from pathlib import Path

from src.pipeline import run_pipeline
from src.utils.config import ensure_directories, load_config, project_path
from src.utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DM4ML recommendation data-management pipeline end to end."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--inject-faults",
        action="store_true",
        help="Inject validation faults into a copy of the raw data for the demo run.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download MovieLens; fail if the local dataset is missing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_directories(config)
    log_file = project_path(config, "reports") / "run_log.txt"
    setup_logging(log_file=log_file)
    logger = logging.getLogger("run_pipeline")
    logger.info("Starting DM4ML recommendation pipeline")
    logger.info("Config: %s", config["_config_path"])
    try:
        state = run_pipeline(
            config,
            inject_faults_flag=bool(args.inject_faults),
            skip_download=bool(args.skip_download),
        )
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    metrics = state.get("summary", {}).get("metrics", {})
    for model, values in metrics.items():
        logger.info("%s metrics: %s", model, values)
    logger.info("Artifacts: reports/, reports/figures/, data/features/, models/, mlruns/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
