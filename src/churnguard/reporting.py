"""Generate ``reports/resume_facts.md`` from artifacts the pipeline just wrote.

Every number in the generated file is read back out of the run summary and the
generated tables, so the document cannot drift from -- or invent claims beyond
-- what the executed code actually produced.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def count_collected_tests(project_root: Path = config.PROJECT_ROOT) -> int | None:
    """Return the number of tests pytest collects, or None if collection fails."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"(\d+) tests? collected", completed.stdout)
    return int(match.group(1)) if match else None


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_resume_facts(
    summary: dict[str, Any],
    validation_metrics: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    experiment_table: pd.DataFrame,
    chi_square_summary: pd.DataFrame,
    validation_report: dict[str, Any],
    path: Path,
) -> Path:
    """Render the verified-claims file used as raw material for a resume."""
    test_metrics = summary["test_metrics"]
    policy_rows = {round(row["capacity_fraction"], 2): row for row in summary["policy"]}
    uncalibrated_brier = float(
        calibration_comparison.loc[
            calibration_comparison["variant"] == "uncalibrated", "brier_score"
        ].iloc[0]
    )
    chosen = summary["calibration_choice"]
    chosen_brier = float(
        calibration_comparison.loc[
            calibration_comparison["variant"] == chosen, "brier_score"
        ].iloc[0]
    )
    design = experiment_table[experiment_table["analysis"] == "required_sample_size"]
    n_tests = count_collected_tests()

    lines: list[str] = []
    add = lines.append

    add("# RetentionLab - verified project facts")
    add("")
    add(
        "Every figure below was read back from artifacts produced by "
        "the analytical pipeline and the serving verification/benchmark scripts. "
        "Experiment effects are assumptions; serving measurements are local."
    )
    add("")
    add("## DATA SCIENCE")
    add("")
    add("DATA:")
    add(f"- {summary['n_rows']:,} customer rows, {summary['n_raw_columns']} raw columns "
        "(IBM Telco Customer Churn, fictional sample dataset)")
    add(f"- {summary['n_model_features']} model features after excluding customerID and the target")
    add(f"- churn prevalence {_pct(summary['churn_prevalence'])}")
    add(
        f"- stratified split: {summary['split_sizes']['train']:,} train / "
        f"{summary['split_sizes']['validation']:,} validation / "
        f"{summary['split_sizes']['test']:,} test (seed {summary['random_seed']})"
    )
    for warning in validation_report["warnings"]:
        add(f"- data-quality finding: {warning}")
    add(
        f"- duplicate customerIDs found: "
        f"{validation_report['checks']['duplicate_customer_ids']['count']}"
    )
    add("")

    add("MODEL:")
    for _, row in validation_metrics.iterrows():
        add(
            f"- validation {row['model']}: PR-AUC {row['pr_auc']:.3f}, "
            f"ROC-AUC {row['roc_auc']:.3f}, Brier {row['brier_score']:.4f}, "
            f"log loss {row['log_loss']:.4f}"
        )
    add(
        f"- selected on validation PR-AUC: {summary['selected_model']} "
        "(test set untouched during selection)"
    )
    add(f"- final test ROC-AUC: {test_metrics['roc_auc']:.3f}")
    add(f"- final test PR-AUC: {test_metrics['pr_auc']:.3f} "
        f"(test churn prevalence {_pct(test_metrics['positive_rate'])})")
    add(f"- final test Brier score: {test_metrics['brier_score']:.4f}")
    add(f"- final test log loss: {test_metrics['log_loss']:.4f}")
    add(
        f"- final test accuracy {test_metrics['accuracy']:.3f}, "
        f"precision {test_metrics['precision']:.3f}, "
        f"recall {test_metrics['recall']:.3f}, "
        f"F1 {test_metrics['f1']:.3f} at threshold {test_metrics['threshold']:g}"
    )
    add("")

    add("CALIBRATION:")
    add(
        f"- {config.CALIBRATION_CV_FOLDS}-fold cross-validated calibration compared against "
        "the uncalibrated model on the validation split"
    )
    add(
        f"- validation Brier: uncalibrated {uncalibrated_brier:.5f} -> {chosen} "
        f"{chosen_brier:.5f} "
        f"({'improvement' if chosen_brier < uncalibrated_brier else 'no improvement'} of "
        f"{abs(uncalibrated_brier - chosen_brier):.5f})"
    )
    add("")

    add("POLICY (retrospective prioritisation on the held-out test set, not causal):")
    for fraction in sorted(policy_rows):
        row = policy_rows[fraction]
        add(
            f"- top {fraction:.0%} outreach ({row['n_contacted']} of {row['n_customers']} "
            f"customers): captured {row['churners_captured']} of {row['total_churners']} "
            f"actual churners = {_pct(row['churner_capture_rate'])} capture, "
            f"{_pct(row['precision_in_contacted'])} precision among contacted, "
            f"{row['lift_vs_random']:.2f}x lift vs random outreach"
        )
    add("")

    add("EXPERIMENT DESIGN (prospective; no treatment exists in this dataset):")
    add(
        f"- assumed baseline churn rate {_pct(summary['rct_baseline_churn_rate_high_risk'])}, "
        "the observed churn rate among the customers a top-20% policy would contact"
    )
    for _, row in design.iterrows():
        add(
            f"- detecting a {row['assumed_absolute_reduction'] * 100:.0f} pp absolute "
            f"reduction at {row['power']:.0%} power (alpha {row['alpha']}): "
            f"{int(row['n_per_arm']):,} customers per arm ({int(row['total_n']):,} total)"
        )
    add("")

    add("ASSOCIATION TESTS (observational, pre-specified, not causal):")
    for _, row in chi_square_summary.iterrows():
        add(
            f"- churn vs {row['feature']}: chi-square {row['chi2_statistic']:.1f} "
            f"(df {int(row['degrees_of_freedom'])}), p = {row['p_value']:.3g}, "
            f"Cramer's V {row['cramers_v']:.3f}"
        )
    add("")

    add("## ML ENGINEERING")
    add("")
    verification_path = config.REPORTS_DIR / "serving_verification.json"
    if verification_path.exists():
        verified = json.loads(verification_path.read_text())
        results = verified["test_results"]
        add(f"- {verified['endpoint_count']} service endpoints; "
            f"maximum vectorized batch: {verified['max_batch_size']} customers")
        add(f"- checksum, feature-contract and startup smoke verification: "
            f"{verified['model_integrity']}; model {verified['model_version']}")
        add(f"- PSI on {', '.join(verified['numeric_drift_features'])}; "
            f"categorical total variation on {', '.join(verified['categorical_drift_features'])}")
        add(f"- training-only drift reference: {verified['reference_row_count']:,} rows")
        add(f"- {len(verified['metrics'])} Prometheus metric families; instance-local counters")
        add(f"- executed automated tests: {results['tests']}; "
            f"failures {results['failures']}, errors {results['errors']}, "
            f"skipped {results['skipped']}")
        add(f"- CI: {verified['ci']}")
    elif n_tests is not None:
        add(f"- {n_tests} tests collected; serving verification has not been recorded")
    benchmark_path = config.REPORTS_DIR / "serving_benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text())
        for row in benchmark["results"]:
            direct, http = row["direct_model"], row["fastapi_testclient"]
            add(f"- local batch {row['batch_size']}: direct p50/p95 "
                f"{direct['p50_ms']:.2f}/{direct['p95_ms']:.2f} ms; FastAPI TestClient "
                f"{http['p50_ms']:.2f}/{http['p95_ms']:.2f} ms (not network latency)")
    runtime_path = config.REPORTS_DIR / "runtime_validation.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        add(f"- Docker: {runtime['docker_status']}")
        add(f"- local Uvicorn HTTP endpoints verified: {len(runtime['verified_endpoints'])}")
    add("- no production traffic, causal retention uplift, "
        "or globally aggregated telemetry measured")
    add("")

    deployment_path = config.REPORTS_DIR / "deployment.json"
    if deployment_path.exists():
        deployment = json.loads(deployment_path.read_text())
        add("## DEPLOYMENT")
        add("")
        if deployment.get("status") == "verified" and deployment.get("public_url"):
            # Only claim a deployment once its public endpoints have actually answered.
            add(f"- deployed on {deployment['provider']} "
                f"(project {deployment['vercel_project']}): {deployment['public_url']}")
            add(f"- {deployment['endpoints_verified']} of "
                f"{deployment['endpoints_total']} public HTTPS endpoints exercised and verified "
                f"on {deployment['verification_timestamp_utc']}")
            add(f"- serving model {deployment['model_version']} with startup checksum "
                "verification, from the same artifact bytes verified locally")
            add("- public response times include cold starts and are not a latency benchmark; "
                "/metrics counters are instance-local, not globally aggregated")
        else:
            add(f"- deployment: {deployment.get('status', 'unknown')}")
        add("")

    while lines and not lines[-1]:
        lines.pop()  # no trailing blank line at end of file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", path.name)
    return path
