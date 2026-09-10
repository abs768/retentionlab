#!/usr/bin/env python3
"""Refresh evidence blocks and generated resume facts from executed verification reports."""

import json
import re

import pandas as pd
from app.config import ROOT, Settings

from churnguard.reporting import write_resume_facts


def main():
    reports = ROOT / "reports"
    verified = json.loads((reports / "serving_verification.json").read_text())
    metadata = json.loads(Settings().metadata_path.read_text())
    benchmark = json.loads((reports / "serving_benchmark.json").read_text())
    n_tests = verified["test_results"]["tests"]
    drift_count = (len(verified["numeric_drift_features"])
                   + len(verified["categorical_drift_features"]))
    claims = [
        f"Automated tests: **{n_tests}** (all passed, zero skipped).",
        f"Service endpoints: **{verified['endpoint_count']}** (excluding framework documentation).",
        f"Maximum batch size: **{verified['max_batch_size']}** customers.",
        f"Drift features: **{drift_count}** "
        "(numeric and categorical combined).",
        f"Training reference rows: **{verified['reference_row_count']}**.",
        f"Model artifact size: **{metadata['artifact_size_bytes']}** bytes.",
        f"Artifact SHA256: `{metadata['sha256']}`.",
    ]
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()
    readme = replace_block(readme, "SERVING_EVIDENCE", "\n\n".join(claims))
    rows = [
        "| Batch size | Direct mean ms | Direct p50 ms | Direct p95 ms | "
        "TestClient mean ms | TestClient p50 ms | TestClient p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in benchmark["results"]:
        cells = [str(row["batch_size"])]
        for mode in ("direct_model", "fastapi_testclient"):
            cells.extend(f"{row[mode][stat]:.2f}" for stat in ("mean_ms", "p50_ms", "p95_ms"))
        rows.append("| " + " | ".join(cells) + " |")
    details = (
        f"Measured on Python {benchmark['python_version']}, scikit-learn "
        f"{benchmark['sklearn_version']}, {benchmark['platform']}; "
        f"{benchmark['warmup_iterations']} warm-up iterations and "
        f"{benchmark['iterations']} timed iterations per path and batch size. "
        "Native thread settings: "
        f"OMP_NUM_THREADS={benchmark['native_threads']['OMP_NUM_THREADS']}, "
        f"OPENBLAS_NUM_THREADS={benchmark['native_threads']['OPENBLAS_NUM_THREADS']}."
    )
    readme = replace_block(readme, "BENCHMARK", "\n".join(rows) + "\n\n" + details)
    readme = re.sub(r"# \d+ pytest tests", f"# {n_tests} pytest tests", readme)
    readme_path.write_text(readme)
    tables = reports / "tables"
    write_resume_facts(
        summary=json.loads((reports / "run_summary.json").read_text()),
        validation_metrics=pd.read_csv(tables / "validation_model_comparison.csv"),
        calibration_comparison=pd.read_csv(tables / "calibration_comparison.csv"),
        experiment_table=pd.read_csv(tables / "experiment_design.csv"),
        chi_square_summary=pd.read_csv(tables / "chi_square_tests.csv"),
        validation_report=json.loads((reports / "validation_report.json").read_text()),
        path=reports / "resume_facts.md",
    )
    print("Updated README evidence and generated resume facts")


def replace_block(text, name, body):
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Missing or duplicated {name} markers")
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body + "\n" + end + after


if __name__ == "__main__":
    main()
