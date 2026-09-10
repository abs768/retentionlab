#!/usr/bin/env python3
"""Audit every numeric claim in README.md against the generated artifacts.

The README hard-codes its results so it reads well on GitHub, which creates a
drift risk: the prose can silently fall out of step with the pipeline. This
script re-reads each claimed number from ``reports/`` and exits non-zero on any
mismatch, so "every number in this README was produced by the pipeline" is a
checkable statement rather than a promise.

This audit runs in CI against committed evidence and the installed service contract.
Run it after regenerating artifacts; scientific dependencies are pinned for serving.

Usage:
    python scripts/verify_readme_claims.py
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


class ClaimChecker:
    """Collects README-claim vs artifact-value comparisons."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str, object, object]] = []

    def check(self, label: str, claimed: float, actual: float) -> None:
        """Compare a README figure against a value read back from an artifact.

        Both sides are rounded to the precision the README prints, so this is an
        exact comparison at the claimed number of digits.
        """
        ok = float(claimed) == float(actual)
        self.results.append((ok, label, claimed, actual))

    @property
    def failures(self) -> list[tuple[bool, str, object, object]]:
        return [row for row in self.results if not row[0]]


def _round_to_claim(value: float, claimed: float) -> float:
    """Round ``value`` to the number of decimals the README shows for ``claimed``."""
    text = str(claimed)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return round(float(value), decimals)


def check_dataset(checker: ClaimChecker, summary: dict, report: dict) -> None:
    checker.check("rows = 7,043", 7043, summary["n_rows"])
    checker.check("raw columns = 21", 21, summary["n_raw_columns"])
    checker.check("model features = 19", 19, summary["n_model_features"])
    checker.check("prevalence = 26.5%", 26.5, round(summary["churn_prevalence"] * 100, 1))
    checker.check(
        "churner count = 1,869", 1869, round(summary["churn_prevalence"] * summary["n_rows"])
    )
    checker.check(
        "duplicate customerIDs = 0",
        0,
        report["checks"]["duplicate_customer_ids"]["count"],
    )
    checker.check(
        "unparseable TotalCharges = 11",
        11,
        report["checks"]["numeric_coercion"]["TotalCharges"]["values_failing_numeric_parse"],
    )
    for split, expected in (("train", 4929), ("validation", 1057), ("test", 1057)):
        checker.check(f"{split} split size", expected, summary["split_sizes"][split])


def check_validation_table(checker: ClaimChecker, validation: pd.DataFrame) -> None:
    claims = {
        "logistic_regression": dict(
            pr_auc=0.628, roc_auc=0.830, brier_score=0.1408, log_loss=0.4358,
            accuracy=0.794, precision=0.639, recall=0.516, f1=0.571,
        ),
        "hist_gradient_boosting": dict(
            pr_auc=0.643, roc_auc=0.833, brier_score=0.1392, log_loss=0.4312,
            accuracy=0.800, precision=0.670, recall=0.491, f1=0.567,
        ),
    }
    for model, metrics in claims.items():
        row = validation[validation["model"] == model].iloc[0]
        for metric, claimed in metrics.items():
            checker.check(
                f"validation {model} {metric}", claimed, _round_to_claim(row[metric], claimed)
            )

    gap = (
        validation[validation["model"] == "hist_gradient_boosting"]["pr_auc"].iloc[0]
        - validation[validation["model"] == "logistic_regression"]["pr_auc"].iloc[0]
    )
    checker.check("validation PR-AUC gap = 0.014", 0.014, round(float(gap), 3))


def check_test_metrics(checker: ClaimChecker, metrics: dict) -> None:
    claims = dict(
        roc_auc=0.855, pr_auc=0.670, brier_score=0.1324, log_loss=0.4058,
        accuracy=0.803, precision=0.671, recall=0.504, f1=0.576,
    )
    for metric, claimed in claims.items():
        checker.check(f"test {metric}", claimed, _round_to_claim(metrics[metric], claimed))
    checker.check("test prevalence = 26.5%", 26.5, round(metrics["positive_rate"] * 100, 1))
    checker.check(
        "PR-AUC / prevalence = 2.5x", 2.5, round(metrics["pr_auc"] / metrics["positive_rate"], 1)
    )


def check_calibration(checker: ClaimChecker, calibration: pd.DataFrame) -> None:
    rows = {row["variant"]: row for _, row in calibration.iterrows()}
    uncalibrated, sigmoid, isotonic = rows["uncalibrated"], rows["sigmoid"], rows["isotonic"]

    checker.check("uncalibrated Brier", 0.13919, round(float(uncalibrated["brier_score"]), 5))
    checker.check("sigmoid Brier", 0.13809, round(float(sigmoid["brier_score"]), 5))
    checker.check("isotonic Brier", 0.13819, round(float(isotonic["brier_score"]), 5))
    checker.check("uncalibrated log loss", 0.4312, round(float(uncalibrated["log_loss"]), 4))
    checker.check("sigmoid log loss", 0.4274, round(float(sigmoid["log_loss"]), 4))
    checker.check("isotonic log loss", 0.4296, round(float(isotonic["log_loss"]), 4))
    checker.check("uncalibrated PR-AUC", 0.643, round(float(uncalibrated["pr_auc"]), 3))
    checker.check("sigmoid PR-AUC", 0.639, round(float(sigmoid["pr_auc"]), 3))
    checker.check("isotonic PR-AUC", 0.641, round(float(isotonic["pr_auc"]), 3))
    checker.check("uncalibrated ROC-AUC", 0.833, round(float(uncalibrated["roc_auc"]), 3))
    checker.check("sigmoid ROC-AUC", 0.836, round(float(sigmoid["roc_auc"]), 3))
    checker.check("isotonic ROC-AUC", 0.836, round(float(isotonic["roc_auc"]), 3))

    brier_delta = float(uncalibrated["brier_score"] - sigmoid["brier_score"])
    checker.check("Brier improvement = 0.0011", 0.0011, round(brier_delta, 4))
    checker.check(
        "Brier improvement = 0.8% relative",
        0.8,
        round(brier_delta / float(uncalibrated["brier_score"]) * 100, 1),
    )
    checker.check(
        "log-loss improvement = 0.0039",
        0.0039,
        round(float(uncalibrated["log_loss"] - sigmoid["log_loss"]), 4),
    )


def check_policy(checker: ClaimChecker, policy: pd.DataFrame) -> None:
    claims = [
        # capacity, contacted, captured, capture %, precision %, lift
        (0.10, 106, 80, 28.6, 75.5, 2.85),
        (0.20, 212, 143, 51.1, 67.5, 2.55),
        (0.30, 318, 185, 66.1, 58.2, 2.20),
    ]
    for capacity, contacted, captured, capture_pct, precision_pct, lift in claims:
        row = policy[policy["capacity_fraction"] == capacity].iloc[0]
        checker.check(f"top {capacity:.0%} contacted", contacted, int(row["n_contacted"]))
        checker.check(f"top {capacity:.0%} captured", captured, int(row["churners_captured"]))
        checker.check(
            f"top {capacity:.0%} capture rate",
            capture_pct,
            round(float(row["churner_capture_rate"]) * 100, 1),
        )
        checker.check(
            f"top {capacity:.0%} precision",
            precision_pct,
            round(float(row["precision_in_contacted"]) * 100, 1),
        )
        checker.check(f"top {capacity:.0%} lift", lift, round(float(row["lift_vs_random"]), 2))

    at_20 = policy[policy["capacity_fraction"] == 0.20].iloc[0]
    checker.check("total test churners = 280", 280, int(policy.iloc[0]["total_churners"]))
    checker.check(
        "random outreach at 20% reaches ~56 churners",
        56,
        round(float(at_20["expected_random_churners"])),
    )


def check_experiment(checker: ClaimChecker, experiment: pd.DataFrame, summary: dict) -> None:
    checker.check(
        "RCT baseline = 67.5%", 67.5, round(summary["rct_baseline_churn_rate_high_risk"] * 100, 1)
    )

    sizes = experiment[experiment["analysis"] == "required_sample_size"]
    for reduction, n80, n90, treatment_pct in (
        (0.02, 8747, 11710, 65.5),
        (0.03, 3915, 5241, 64.5),
        (0.05, 1428, 1912, 62.5),
    ):
        at_80 = sizes[
            (sizes["assumed_absolute_reduction"] == reduction) & (sizes["power"] == 0.80)
        ].iloc[0]
        at_90 = sizes[
            (sizes["assumed_absolute_reduction"] == reduction) & (sizes["power"] == 0.90)
        ].iloc[0]
        checker.check(f"{reduction:.0%} reduction, 80% power", n80, int(at_80["n_per_arm"]))
        checker.check(f"{reduction:.0%} reduction, 90% power", n90, int(at_90["n_per_arm"]))
        checker.check(
            f"{reduction:.0%} reduction implied treatment rate",
            treatment_pct,
            round(float(at_80["assumed_treatment_churn_rate"]) * 100, 1),
        )

    mde = experiment[experiment["analysis"] == "minimum_detectable_effect"]
    for n_per_arm, mde_80, mde_90 in (
        (1_000, 6.0, 7.0),
        (2_500, 3.8, 4.4),
        (5_000, 2.7, 3.1),
        (10_000, 1.9, 2.2),
        (25_000, 1.2, 1.4),
    ):
        at_80 = mde[(mde["n_per_arm"] == n_per_arm) & (mde["power"] == 0.80)].iloc[0]
        at_90 = mde[(mde["n_per_arm"] == n_per_arm) & (mde["power"] == 0.90)].iloc[0]
        checker.check(
            f"MDE at n={n_per_arm:,}, 80% power",
            mde_80,
            round(float(at_80["mde_absolute_reduction"]) * 100, 1),
        )
        checker.check(
            f"MDE at n={n_per_arm:,}, 90% power",
            mde_90,
            round(float(at_90["mde_absolute_reduction"]) * 100, 1),
        )


def check_association_tests(checker: ClaimChecker, chi_square: pd.DataFrame, details: dict) -> None:
    for feature, statistic, dof, cramers_v in (
        ("Contract", 1184.6, 2, 0.410),
        ("InternetService", 732.3, 2, 0.322),
        ("PaperlessBilling", 258.3, 1, 0.191),
    ):
        row = chi_square[chi_square["feature"] == feature].iloc[0]
        checker.check(f"{feature} chi-square", statistic, round(float(row["chi2_statistic"]), 1))
        checker.check(f"{feature} df", dof, int(row["degrees_of_freedom"]))
        checker.check(f"{feature} Cramer's V", cramers_v, round(float(row["cramers_v"]), 3))

    rates = {
        "Contract": {"Month-to-month": 42.7, "One year": 11.3, "Two year": 2.8},
        "InternetService": {"Fiber optic": 41.9, "DSL": 19.0, "No": 7.4},
        "PaperlessBilling": {"Yes": 33.6, "No": 16.3},
    }
    for feature, levels in rates.items():
        for level, claimed in levels.items():
            actual = details[feature]["churn_rate_by_level"][level]
            checker.check(f"{feature}={level} churn rate", claimed, round(actual * 100, 1))


def check_test_count(checker: ClaimChecker) -> None:
    """The README and repository tree both claim a test count; confirm it."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    collected = int(match.group(1)) if match else -1
    readme = (ROOT / "README.md").read_text()
    claimed = re.search(r"Automated tests: \*\*(\d+)\*\*", readme)
    expected = int(claimed.group(1)) if claimed else -2
    checker.check("README automated test count", expected, collected)
    checker.check("pytest collection succeeded", 0, completed.returncode)


def check_serving(checker: ClaimChecker) -> None:
    """Parse new README numbers rather than maintaining a second hand-typed truth."""
    from app.config import MAX_BATCH_SIZE, Settings
    from app.main import create_app
    from app.model_service import artifact_sha256
    from app.monitoring import load_reference
    from fastapi.routing import APIRoute

    readme = (ROOT / "README.md").read_text()
    settings = Settings()
    metadata = json.loads(settings.metadata_path.read_text())
    reference = load_reference(settings.reference_path)
    api = create_app()
    actuals = {
        "Service endpoints": sum(isinstance(route, APIRoute) for route in api.routes),
        "Maximum batch size": MAX_BATCH_SIZE,
        "Drift features": len(reference["numeric"]) + len(reference["categorical"]),
        "Training reference rows": reference["row_count"],
        "Model artifact size": settings.model_path.stat().st_size,
    }
    for label, actual in actuals.items():
        match = re.search(re.escape(label) + r": \*\*(\d+)\*\*", readme)
        checker.check(label, int(match.group(1)) if match else -1, actual)
    sha_match = re.search(r"Artifact SHA256: `([0-9a-f]{64})`", readme)
    sha = artifact_sha256(settings.model_path.read_bytes())
    checker.check("README artifact SHA matches trusted metadata and bytes", 1,
                  int(sha_match is not None and sha_match.group(1) == metadata["sha256"] == sha))
    benchmark = json.loads((REPORTS / "serving_benchmark.json").read_text())
    checker.check("Benchmark artifact SHA matches served model", 1,
                  int(benchmark["artifact_sha256"] == sha))
    block = readme.split("<!-- BENCHMARK_START -->")[1].split("<!-- BENCHMARK_END -->")[0]
    lines = [line for line in block.splitlines() if re.match(r"\| \d+ \|", line)]
    checker.check("Benchmark row count", len(benchmark["results"]), len(lines))
    by_size = {int(line.split("|")[1].strip()): line for line in lines}
    for row in benchmark["results"]:
        cells = by_size.get(row["batch_size"], "").split("|")[2:-1]
        claimed = [float(cell.strip()) for cell in cells]
        actual = [row[mode][stat] for mode in ("direct_model", "fastapi_testclient")
                  for stat in ("mean_ms", "p50_ms", "p95_ms")]
        for index, value in enumerate(actual):
            checker.check(f"Benchmark batch {row['batch_size']} statistic {index}",
                          claimed[index] if index < len(claimed) else -1, round(value, 2))
    for key, pattern in (("iterations", r"(\d+) timed iterations"),
                         ("warmup_iterations", r"(\d+) warm-up iterations")):
        match = re.search(pattern, block)
        checker.check(f"Benchmark {key}", int(match.group(1)) if match else -1, benchmark[key])



def check_landing_facts(checker: ClaimChecker) -> None:
    """Every number the public landing page renders must match a canonical artifact.

    The page reads ``artifacts/landing/landing_facts.json`` and ``GET /project-summary``
    serves that same file, so validating the payload validates both surfaces.
    """
    facts = json.loads((ROOT / "artifacts" / "landing" / "landing_facts.json").read_text())
    summary = json.loads((REPORTS / "run_summary.json").read_text())
    verification = json.loads((REPORTS / "serving_verification.json").read_text())
    benchmark = json.loads((REPORTS / "serving_benchmark.json").read_text())
    metrics = summary["test_metrics"]
    top20 = next(r for r in summary["policy"] if abs(r["capacity_fraction"] - 0.20) < 1e-9)

    headline = {item["label"]: item["value"] for item in facts["headline_metrics"]}
    expected_headline = {
        "Customer Records": f"{summary['n_rows']:,}",
        "ROC-AUC": f"{metrics['roc_auc']:.3f}",
        "PR-AUC": f"{metrics['pr_auc']:.3f}",
        "Churners Captured": f"{top20['churner_capture_rate'] * 100:.1f}%",
        "Lift vs Random Targeting": f"{top20['lift_vs_random']:.2f}x",
    }
    checker.check("landing headline metric count", len(expected_headline), len(headline))
    for label, want in expected_headline.items():
        checker.check(f"landing headline {label}", 1, int(headline.get(label) == want))

    performance = facts["performance"]
    for key in ("roc_auc", "pr_auc", "brier_score", "log_loss"):
        checker.check(f"landing performance {key}", performance[key], round(metrics[key], 4))
    checker.check("landing test-split size", performance["test_rows"], metrics["n_observations"])

    checker.check("landing policy row count", len(facts["policy"]), len(summary["policy"]))
    for row in facts["policy"]:
        source = next(
            r for r in summary["policy"]
            if int(round(r["capacity_fraction"] * 100)) == row["capacity_pct"]
        )
        checker.check(f"landing policy {row['capacity_pct']}% capture",
                      row["capture_pct"], round(source["churner_capture_rate"] * 100, 1))
        checker.check(f"landing policy {row['capacity_pct']}% lift",
                      row["lift"], round(source["lift_vs_random"], 2))
        checker.check(f"landing policy {row['capacity_pct']}% contacted",
                      row["contacted"], source["n_contacted"])

    experiment = pd.read_csv(TABLES / "experiment_design.csv")
    design = facts["experiment"]
    sizing = experiment[
        (experiment["analysis"] == "required_sample_size")
        & (experiment["assumed_absolute_reduction"] == design["reduction_pp"] / 100)
        & (experiment["power"] == design["power_pct"] / 100)
    ].iloc[0]
    checker.check("landing experiment per-arm size", design["per_arm"], int(sizing["n_per_arm"]))
    checker.check("landing experiment total size", design["total_n"], int(sizing["total_n"]))
    checker.check("landing experiment baseline", design["baseline_pct"],
                  round(float(sizing["baseline_churn_rate"]) * 100, 1))

    engineering = facts["engineering"]
    checker.check("landing feature count", engineering["features"], summary["n_model_features"])
    checker.check("landing max batch", engineering["max_batch"], verification["max_batch_size"])
    checker.check("landing metric families", engineering["metric_families"],
                  len(verification["metrics"]))
    checker.check("landing endpoint count", engineering["endpoint_count"],
                  verification["endpoint_count"])
    checker.check("landing reference rows", engineering["reference_rows"],
                  verification["reference_row_count"])

    checker.check("landing benchmark row count",
                  len(facts["benchmark"]["rows"]), len(benchmark["results"]))
    for row, source in zip(
        facts["benchmark"]["rows"], benchmark["results"], strict=True
    ):
        checker.check(f"landing benchmark batch {row['batch_size']} p95",
                      row["http_p95"], round(source["fastapi_testclient"]["p95_ms"], 2))

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    checker.check("landing automated test count", engineering["tests"],
                  int(match.group(1)) if match else -1)

    # Demo records must never carry a churn label.
    demo_records = [e["customer"] for e in facts["example_customers"]]
    demo_records += [c["customer"] for c in facts["ranking_cohort"]]
    checker.check("landing demo records carry no target", 0,
                  sum(1 for record in demo_records if "Churn" in record))


def check_deployment(checker: ClaimChecker) -> None:
    """README deployment claims must match the recorded public verification."""
    path = REPORTS / "deployment.json"
    if not path.exists():
        return
    deployment = json.loads(path.read_text())
    readme = (ROOT / "README.md").read_text()

    verified = sum(1 for row in deployment["verified_endpoints"] if row["passed"])
    checker.check("deployment.json self-consistent verified count",
                  deployment["endpoints_verified"], verified)

    block = readme.split("<!-- DEPLOYMENT_START -->")[1].split("<!-- DEPLOYMENT_END -->")[0]
    claimed = re.findall(r"\*\*(\d+)\*\*", block)
    checker.check("README verified endpoint count", int(claimed[0]), verified)
    checker.check("README total endpoint count", int(claimed[1]), deployment["endpoints_total"])
    checker.check("README deployment URL matches evidence", 1,
                  int(deployment["public_url"] in block))
    checker.check("README live demo link matches evidence", 1,
                  int(f"**Live Demo:** {deployment['public_url']}" in readme))
    checker.check("every recorded endpoint passed", deployment["endpoints_total"], verified)

def main() -> int:
    summary = json.loads((REPORTS / "run_summary.json").read_text())
    validation_report = json.loads((REPORTS / "validation_report.json").read_text())
    chi_square_details = json.loads((TABLES / "chi_square_details.json").read_text())

    checker = ClaimChecker()
    check_dataset(checker, summary, validation_report)
    check_validation_table(checker, pd.read_csv(TABLES / "validation_model_comparison.csv"))
    check_test_metrics(checker, summary["test_metrics"])
    check_calibration(checker, pd.read_csv(TABLES / "calibration_comparison.csv"))
    check_policy(checker, pd.read_csv(TABLES / "retention_policy.csv"))
    check_experiment(checker, pd.read_csv(TABLES / "experiment_design.csv"), summary)
    check_association_tests(
        checker, pd.read_csv(TABLES / "chi_square_tests.csv"), chi_square_details
    )
    check_test_count(checker)
    check_landing_facts(checker)
    check_deployment(checker)
    check_serving(checker)

    for _, label, claimed, actual in checker.failures:
        print(f"MISMATCH  {label}: README={claimed}  artifact={actual}")

    total = len(checker.results)
    passed = total - len(checker.failures)

    # Record the outcome so other generators quote this number instead of
    # re-deriving it (or scraping this script's stdout).
    (REPORTS / "claim_verification.json").write_text(
        json.dumps(
            {
                "total_claims": total,
                "claims_passed": passed,
                "all_passed": not checker.failures,
                "failures": [row[1] for row in checker.failures],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n{passed}/{total} README numeric claims verified against artifacts.")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
