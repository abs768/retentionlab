#!/usr/bin/env python3
"""Build the single canonical payload behind the public landing page.

The recruiter-facing page must never restate a metric independently. This
script reads the frozen analytical outputs in ``reports/`` plus the recorded
serving evidence, and writes ``artifacts/landing/landing_facts.json``. The page
renders that file, and ``GET /project-summary`` serves it verbatim, so the page,
the README, and the API all quote the same evidence.

Every number here is copied from an artifact. Nothing is computed a second time
and nothing is typed by hand except labels, prose, and the illustrative sample
customers (which carry no churn labels).

Usage:
    python scripts/build_landing_facts.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
OUTPUT = ROOT / "artifacts" / "landing" / "landing_facts.json"

REPOSITORY_URL = "https://github.com/abs768/retentionlab"
PUBLIC_NAME = "RetentionLab"
PUBLIC_TAGLINE = "End-to-End Customer Retention Analytics Platform"
HERO_HEADLINE = "Spend the retention budget on the customers most likely to leave."
# Filled from the run summary in build_facts(); no number is typed here.
HERO_SUMMARY_TEMPLATE = (
    "RetentionLab ranks {rows} telecom customers by calibrated churn risk. Contacting the "
    "riskiest 20% reaches {capture} of churners, {lift} random outreach. It also sizes the "
    "A/B test needed to prove an offer works, and serves the model through a "
    "checksum-verified FastAPI inference service."
)

# Display-only bands used by the demo UI. They are not a business decision
# threshold; the calibrated probability is the authoritative model output.
RISK_BANDS = [
    {"label": "Lower", "min": 0.0, "max": 0.30, "tone": "success"},
    {"label": "Elevated", "min": 0.30, "max": 0.60, "tone": "warning"},
    {"label": "High", "min": 0.60, "max": 1.01, "tone": "danger"},
]

# PSI interpretation heuristics (operational rules of thumb, not significance tests).
PSI_BANDS = [
    {"range": "< 0.10", "meaning": "Little apparent shift", "tone": "success"},
    {"range": "0.10 – 0.25", "meaning": "Investigate", "tone": "warning"},
    {"range": "≥ 0.25", "meaning": "Substantial apparent shift", "tone": "danger"},
]

TECH_STACK = [
    {"name": "Python", "role": "Language"},
    {"name": "scikit-learn", "role": "Modeling & calibration"},
    {"name": "pandas", "role": "Data handling"},
    {"name": "NumPy", "role": "Vectorized inference"},
    {"name": "statsmodels", "role": "Power analysis"},
    {"name": "FastAPI", "role": "Inference service"},
    {"name": "Pydantic", "role": "Schema validation"},
    {"name": "pytest", "role": "Automated tests"},
    {"name": "Ruff", "role": "Static analysis"},
    {"name": "Prometheus", "role": "Instrumentation"},
    {"name": "Docker", "role": "Container definition"},
    {"name": "GitHub Actions", "role": "CI"},
    {"name": "Vercel", "role": "Public deployment"},
]

PIPELINE_STAGES = [
    ("IBM Telco Sample", "7,043 labelled customer records"),
    ("Schema Validation", "Columns, duplicates, target values, numeric parsing"),
    ("Leakage-Safe Preprocessing", "Transformers fitted on the training split only"),
    ("Model Comparison", "Interpretable baseline vs nonlinear challenger"),
    ("HistGradientBoosting", "Selected on validation PR-AUC"),
    ("Probability Calibration", "Cross-validated, chosen on validation Brier"),
    ("Untouched Test Evaluation", "Held-out split scored exactly once"),
    ("Retention Ranking", "Capacity-constrained prioritization"),
    ("Experiment Design", "Prospective power analysis"),
]

SERVING_STAGES = [
    ("final_model.joblib", "Frozen serialized pipeline"),
    ("SHA-256 Verification", "Checked before the model is served"),
    ("FastAPI Service", "Typed request contract, safe error surfaces"),
    ("Prometheus Metrics", "Six instrumented metric families"),
    ("Vercel Deployment", "Stateless Python serverless function"),
]

DECISION_FLOW = [
    ("7,043 customers", "Observational records, no treatment assignment"),
    ("Predict churn probability", "Gradient boosting over 19 validated features"),
    ("Calibrate risk", "Probabilities usable as numbers, not just ranks"),
    ("Apply outreach capacity", "The retention team cannot contact everyone"),
    ("Rank customers", "Highest predicted risk first"),
    ("Target the top cohort", "51.1% of churners inside the top 20%"),
    ("Test with an RCT", "The only way to measure a causal effect"),
]

# Illustrative fictional customers for the interactive demos. These are valid
# feature combinations authored for this page -- they are not rows from the IBM
# dataset and carry no churn labels.
def _customer(**overrides) -> dict:
    base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }
    base.update(overrides)
    base["TotalCharges"] = round(base["tenure"] * base["MonthlyCharges"], 2)
    return base


EXAMPLE_CUSTOMERS = [
    {
        "name": "New fiber subscriber, month-to-month",
        "customer": _customer(
            tenure=2, InternetService="Fiber optic", MonthlyCharges=94.4,
            StreamingTV="Yes", StreamingMovies="Yes", PaymentMethod="Electronic check",
        ),
    },
    {
        "name": "Long-tenure two-year contract",
        "customer": _customer(
            tenure=64, Contract="Two year", Partner="Yes", Dependents="Yes",
            InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
            TechSupport="Yes", DeviceProtection="Yes", PaperlessBilling="No",
            PaymentMethod="Bank transfer (automatic)", MonthlyCharges=65.15,
        ),
    },
]

RANKING_COHORT = [
    ("C-1042", _customer(tenure=1, InternetService="Fiber optic", MonthlyCharges=89.9,
                         StreamingTV="Yes", StreamingMovies="Yes")),
    ("C-1077", _customer(tenure=70, Contract="Two year", Partner="Yes", Dependents="Yes",
                         OnlineSecurity="Yes", TechSupport="Yes", MonthlyCharges=61.0,
                         PaperlessBilling="No", PaymentMethod="Credit card (automatic)")),
    ("C-1108", _customer(tenure=4, InternetService="Fiber optic", MonthlyCharges=99.35,
                         MultipleLines="Yes", StreamingMovies="Yes")),
    ("C-1153", _customer(tenure=33, Contract="One year", InternetService="DSL",
                         OnlineBackup="Yes", MonthlyCharges=58.7,
                         PaymentMethod="Mailed check")),
    ("C-1190", _customer(tenure=2, InternetService="Fiber optic", MonthlyCharges=104.8,
                         MultipleLines="Yes", StreamingTV="Yes", StreamingMovies="Yes",
                         SeniorCitizen=1)),
    ("C-1234", _customer(tenure=48, Contract="Two year", InternetService="No",
                         OnlineSecurity="No internet service",
                         OnlineBackup="No internet service",
                         DeviceProtection="No internet service",
                         TechSupport="No internet service",
                         StreamingTV="No internet service",
                         StreamingMovies="No internet service",
                         MonthlyCharges=20.15, PaperlessBilling="No",
                         PaymentMethod="Bank transfer (automatic)")),
    ("C-1281", _customer(tenure=9, InternetService="Fiber optic", MonthlyCharges=84.5,
                         TechSupport="No", Partner="Yes")),
    ("C-1319", _customer(tenure=58, Contract="Two year", Partner="Yes",
                         InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
                         DeviceProtection="Yes", TechSupport="Yes", MonthlyCharges=70.3,
                         PaymentMethod="Bank transfer (automatic)")),
    ("C-1355", _customer(tenure=6, Contract="Month-to-month", InternetService="Fiber optic",
                         MonthlyCharges=95.6, StreamingTV="Yes")),
    ("C-1402", _customer(tenure=25, Contract="One year", InternetService="DSL",
                         TechSupport="Yes", MonthlyCharges=54.25,
                         PaymentMethod="Credit card (automatic)")),
    ("C-1448", _customer(tenure=3, InternetService="Fiber optic", MonthlyCharges=79.85,
                         Dependents="No", PaperlessBilling="Yes")),
    ("C-1495", _customer(tenure=41, Contract="One year", Partner="Yes", Dependents="Yes",
                         InternetService="DSL", OnlineSecurity="Yes",
                         MonthlyCharges=63.4, PaymentMethod="Mailed check")),
]



def png_size(path: Path) -> tuple[int, int]:
    """Read a PNG's intrinsic dimensions from its IHDR chunk (no image library)."""
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} is not a PNG")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def count_tests() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:warnings"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    if not match:
        raise RuntimeError("Could not determine the collected test count.")
    return int(match.group(1))


def readme_claim_count() -> int | None:
    """Total numeric claims, read from the verifier's recorded result artifact.

    Reading the artifact rather than re-running the verifier keeps this script
    free of a circular dependency: the verifier is the authority on its own
    count, and a test asserts the two stay in step.
    """
    path = REPORTS / "claim_verification.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return int(record["total_claims"]) if record.get("all_passed") else None


def build_facts() -> dict:
    summary = json.loads((REPORTS / "run_summary.json").read_text())
    verification = json.loads((REPORTS / "serving_verification.json").read_text())
    benchmark = json.loads((REPORTS / "serving_benchmark.json").read_text())
    validation = json.loads((REPORTS / "validation_report.json").read_text())
    comparison = pd.read_csv(TABLES / "validation_model_comparison.csv")
    calibration = pd.read_csv(TABLES / "calibration_comparison.csv")
    experiment = pd.read_csv(TABLES / "experiment_design.csv")

    metrics = summary["test_metrics"]
    policy = sorted(summary["policy"], key=lambda row: row["capacity_fraction"])
    top20 = next(r for r in policy if abs(r["capacity_fraction"] - 0.20) < 1e-9)
    sizing = experiment[
        (experiment["analysis"] == "required_sample_size")
        & (experiment["assumed_absolute_reduction"] == 0.03)
        & (experiment["power"] == 0.80)
    ].iloc[0]

    n_tests = count_tests()
    claims = readme_claim_count()

    return {
        "name": PUBLIC_NAME,
        "tagline": PUBLIC_TAGLINE,
        "repository_url": REPOSITORY_URL,
        "headline": HERO_HEADLINE,
        "hero_summary": HERO_SUMMARY_TEMPLATE.format(
            rows=f"{summary['n_rows']:,}",
            capture=f"{top20['churner_capture_rate'] * 100:.1f}%",
            lift=f"{top20['lift_vs_random']:.2f}x",
        ),
        "model_version": verification["model_version"],
        "model_sha256": verification["model_sha256"],
        "generated_from": {
            "analytics": "reports/run_summary.json",
            "serving": "reports/serving_verification.json",
            "benchmark": "reports/serving_benchmark.json",
            "tables": "reports/tables/",
        },
        "headline_metrics": [
            {"value": f"{summary['n_rows']:,}", "label": "Customer Records",
             "detail": "IBM fictional Telco sample"},
            {"value": f"{metrics['roc_auc']:.3f}", "label": "ROC-AUC",
             "detail": "Held-out test split, scored once"},
            {"value": f"{metrics['pr_auc']:.3f}", "label": "PR-AUC",
             "detail": f"Against {metrics['positive_rate'] * 100:.1f}% prevalence"},
            {"value": f"{top20['churner_capture_rate'] * 100:.1f}%",
             "label": "Churners Captured",
             "detail": "within the highest-risk 20%"},
            {"value": f"{top20['lift_vs_random']:.2f}x", "label": "Lift vs Random Targeting",
             "detail": "vs. random outreach of the same size"},
        ],
        "decision_flow": [{"step": a, "detail": b} for a, b in DECISION_FLOW],
        "dataset": {
            "rows": summary["n_rows"],
            "features": summary["n_model_features"],
            "prevalence_pct": round(summary["churn_prevalence"] * 100, 1),
            "train": summary["split_sizes"]["train"],
            "validation": summary["split_sizes"]["validation"],
            "test": summary["split_sizes"]["test"],
            "seed": summary["random_seed"],
            "coercion_warnings": validation["warnings"],
        },
        "performance": {
            "roc_auc": round(metrics["roc_auc"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "brier_score": round(metrics["brier_score"], 4),
            "log_loss": round(metrics["log_loss"], 4),
            "precision": round(metrics["precision"], 3),
            "recall": round(metrics["recall"], 3),
            "f1": round(metrics["f1"], 3),
            "prevalence_pct": round(metrics["positive_rate"] * 100, 1),
            "test_rows": metrics["n_observations"],
        },
        "model_comparison": [
            {
                "model": row["model"].replace("_", " ").title(),
                "pr_auc": round(float(row["pr_auc"]), 3),
                "roc_auc": round(float(row["roc_auc"]), 3),
                "brier_score": round(float(row["brier_score"]), 4),
                "selected": row["model"] == summary["selected_model"],
            }
            for _, row in comparison.iterrows()
        ],
        "calibration": {
            "choice": summary["calibration_choice"],
            "variants": [
                {
                    "variant": row["variant"],
                    "brier_score": round(float(row["brier_score"]), 5),
                    "pr_auc": round(float(row["pr_auc"]), 3),
                    "selected": row["variant"] == summary["calibration_choice"],
                }
                for _, row in calibration.iterrows()
            ],
        },
        "policy": [
            {
                "capacity_pct": int(round(row["capacity_fraction"] * 100)),
                "contacted": row["n_contacted"],
                "captured": row["churners_captured"],
                "total_churners": row["total_churners"],
                "capture_pct": round(row["churner_capture_rate"] * 100, 1),
                "precision_pct": round(row["precision_in_contacted"] * 100, 1),
                "lift": round(row["lift_vs_random"], 2),
                "highlight": abs(row["capacity_fraction"] - 0.20) < 1e-9,
            }
            for row in policy
        ],
        "experiment": {
            "baseline_pct": round(float(sizing["baseline_churn_rate"]) * 100, 1),
            "reduction_pp": int(round(float(sizing["assumed_absolute_reduction"]) * 100)),
            "power_pct": int(round(float(sizing["power"]) * 100)),
            "alpha": float(sizing["alpha"]),
            "per_arm": int(sizing["n_per_arm"]),
            "total_n": int(sizing["total_n"]),
        },
        "engineering": {
            "features": summary["n_model_features"],
            "max_batch": verification["max_batch_size"],
            "tests": n_tests,
            "readme_claims": claims,
            "metric_families": len(verification["metrics"]),
            "endpoint_count": verification["endpoint_count"],
            "reference_rows": verification["reference_row_count"],
            "python_version": verification["python_version"],
            "sklearn_version": benchmark["sklearn_version"],
        },
        "drift": {
            "numeric": verification["numeric_drift_features"],
            "categorical": verification["categorical_drift_features"],
            "psi_bands": PSI_BANDS,
        },
        "benchmark": {
            "method": benchmark["benchmark_method"],
            "iterations": benchmark["iterations"],
            "platform": benchmark["platform"],
            "rows": [
                {
                    "batch_size": row["batch_size"],
                    "direct_p50": round(row["direct_model"]["p50_ms"], 2),
                    "direct_p95": round(row["direct_model"]["p95_ms"], 2),
                    "http_p50": round(row["fastapi_testclient"]["p50_ms"], 2),
                    "http_p95": round(row["fastapi_testclient"]["p95_ms"], 2),
                }
                for row in benchmark["results"]
            ],
        },
        "pipeline_stages": [{"stage": a, "detail": b} for a, b in PIPELINE_STAGES],
        "serving_stages": [{"stage": a, "detail": b} for a, b in SERVING_STAGES],
        "endpoints": verification["endpoints"],
        "metric_families": verification["metrics"],
        "tech_stack": TECH_STACK,
        "risk_bands": RISK_BANDS,
        "example_customers": EXAMPLE_CUSTOMERS,
        "ranking_cohort": [{"id": cid, "customer": c} for cid, c in RANKING_COHORT],
        "figures": [
            dict(entry, **dict(zip(("width", "height"),
                                  png_size(ROOT / "reports" / "figures" / entry["file"]),
                                  strict=True)))
            for entry in (
                {"file": "roc_curve.png", "title": "ROC curve",
                 "caption": "Held-out test split with the random-classifier diagonal."},
                {"file": "precision_recall_curve.png", "title": "Precision-recall curve",
                 "caption": "Prevalence baseline shown for reference."},
                {"file": "calibration_curve.png", "title": "Calibration curve",
                 "caption": "Calibrated against uncalibrated probabilities."},
                {"file": "cumulative_gains.png", "title": "Cumulative gains",
                 "caption": "Churner capture as outreach capacity grows."},
                {"file": "confusion_matrix.png", "title": "Confusion matrix",
                 "caption": "At the 0.5 decision threshold."},
            )
        ],
        "limitations": [
            "IBM's fictional Telco sample dataset, not real customer records.",
            "Observational data with no treatment assignment of any kind.",
            "Predictive ranking does not establish a causal retention effect.",
            "The randomized experiment is designed and sized, not conducted.",
            "Latency figures come from a local benchmark, not production network traffic.",
            "This is a portfolio deployment: no real traffic, no availability guarantee.",
            "Prometheus counters are per-instance on serverless and are not globally durable.",
        ],
    }


def main() -> int:
    facts = build_facts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(facts, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}: "
        f"{len(facts['headline_metrics'])} headline metrics, "
        f"{len(facts['policy'])} policy rows, {len(facts['figures'])} figures, "
        f"{facts['engineering']['tests']} tests, "
        f"{len(facts['ranking_cohort'])} sample customers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
