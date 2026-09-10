#!/usr/bin/env python3
"""Derive the recruiter landing page's headline numbers from canonical artifacts.

The landing page must not hard-code metrics. This script reads the frozen
analytical outputs in ``reports/`` plus the live pytest collection count and
writes ``artifacts/landing/landing_facts.json``, which the served page renders.
Regenerate it whenever the pipeline or the test suite changes.

Usage:
    python scripts/build_landing_facts.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SUMMARY = ROOT / "reports" / "run_summary.json"
OUTPUT = ROOT / "artifacts" / "landing" / "landing_facts.json"

PUBLIC_NAME = "RetentionLab"
PUBLIC_TAGLINE = "End-to-End Customer Retention Analytics Platform"

CAPABILITIES = (
    "Leakage-Safe Modeling",
    "Gradient Boosting",
    "Probability Calibration",
    "Capacity-Constrained Targeting",
    "Statistical Experiment Design",
    "Vectorized Batch Inference",
    "Model Artifact Verification",
    "Drift Monitoring",
    "Prometheus Observability",
)


def count_tests() -> int:
    """Return the number of tests pytest collects for this repository."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:warnings"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    if not match:
        raise RuntimeError("Could not determine the collected test count from pytest output.")
    return int(match.group(1))


def build_facts() -> dict:
    summary = json.loads(RUN_SUMMARY.read_text())
    test_metrics = summary["test_metrics"]
    top20 = next(
        row for row in summary["policy"] if abs(row["capacity_fraction"] - 0.20) < 1e-9
    )

    return {
        "name": PUBLIC_NAME,
        "tagline": PUBLIC_TAGLINE,
        "model_version": "churnguard-hgb-calibrated-v1",
        "source": "reports/run_summary.json",
        "headline_metrics": [
            {
                "value": f"{summary['n_rows']:,}",
                "label": "Customer Records",
                "detail": "IBM fictional Telco sample",
            },
            {
                "value": f"{test_metrics['roc_auc']:.3f}",
                "label": "ROC-AUC",
                "detail": "Held-out test split, scored once",
            },
            {
                "value": f"{test_metrics['pr_auc']:.3f}",
                "label": "PR-AUC",
                "detail": f"Against {test_metrics['positive_rate'] * 100:.1f}% prevalence",
            },
            {
                "value": f"{top20['churner_capture_rate'] * 100:.1f}%",
                "label": "Observed Churners Captured in Top 20% Risk Cohort",
                "detail": (
                    f"{top20['churners_captured']} of {top20['total_churners']} "
                    "test-set churners"
                ),
            },
            {
                "value": f"{top20['lift_vs_random']:.2f}x",
                "label": "Lift Over Random Targeting",
                "detail": "Retrospective ranking, not a causal effect",
            },
            {
                "value": str(count_tests()),
                "label": "Automated Tests",
                "detail": "Analytics, serving, and monitoring",
            },
        ],
        "capabilities": list(CAPABILITIES),
        "brier_score": round(float(test_metrics["brier_score"]), 4),
        "experiment_note": (
            "A 3-point absolute churn reduction would need about 3,915 customers per arm "
            "at 80% power."
        ),
    }


def main() -> int:
    facts = build_facts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(facts, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(facts['headline_metrics'])} metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
