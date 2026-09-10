#!/usr/bin/env python3
"""Reproduce every ChurnGuard output with a single command.

Usage:
    python scripts/run_pipeline.py [--raw-data PATH] [--no-save]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churnguard import config  # noqa: E402
from churnguard.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ChurnGuard pipeline.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Path to the raw Telco churn CSV.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the analysis without writing artifacts to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_pipeline(raw_path=args.raw_data, save_artifacts=not args.no_save)

    log = logging.getLogger("run_pipeline")
    log.info("Selected model: %s", result.selected_model_name)
    log.info(
        "Calibration: %s (improved validation Brier: %s)",
        result.calibration_choice,
        result.calibration_improved_brier,
    )
    log.info("Test metrics: %s", {k: round(v, 4) for k, v in result.test_metrics.items()})
    log.info("Wrote %d artifacts", len(result.artifacts))
    print("\n=== Validation model comparison ===")
    print(result.validation_metrics[
        ["model", "pr_auc", "roc_auc", "brier_score", "log_loss", "f1"]
    ].to_string(index=False))
    print("\n=== Calibration comparison (validation) ===")
    print(result.calibration_comparison[
        ["variant", "brier_score", "log_loss", "pr_auc", "roc_auc"]
    ].to_string(index=False))
    print("\n=== Retention policy (test set) ===")
    print(result.policy_table[
        [
            "capacity_fraction",
            "n_contacted",
            "churners_captured",
            "churner_capture_rate",
            "precision_in_contacted",
            "lift_vs_random",
        ]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
