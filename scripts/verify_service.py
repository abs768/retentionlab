#!/usr/bin/env python3
"""Run tests, lint and artifact checks; record executed evidence without local paths."""

import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import MAX_BATCH_SIZE, ROOT, Settings
from app.main import create_app
from app.model_service import ModelService
from app.monitoring import load_reference
from fastapi.routing import APIRoute


def main():
    subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="retentionlab-tests-") as directory:
        junit = Path(directory) / "results.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=ROOT, check=True
        )
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    if counts["skipped"] or counts["failures"] or counts["errors"]:
        raise RuntimeError("Verification requires a complete passing suite without skips")
    service = ModelService.load(Settings())
    reference = load_reference(Settings().reference_path)
    api = create_app()
    endpoints = [
        {"method": next(iter(route.methods)), "path": route.path}
        for route in api.routes
        if isinstance(route, APIRoute)
    ]
    report = {
        "python_version": platform.python_version(),
        "test_results": counts,
        "lint": "passed",
        "model_integrity": "passed",
        "model_version": service.metadata["model_version"],
        "model_sha256": service.metadata["sha256"],
        "endpoints": endpoints,
        "endpoint_count": len(endpoints),
        "max_batch_size": MAX_BATCH_SIZE,
        "numeric_drift_features": list(reference["numeric"]),
        "categorical_drift_features": list(reference["categorical"]),
        "reference_row_count": reference["row_count"],
        "metrics": [
            "retentionlab_inference_requests_total",
            "retentionlab_inference_errors_total",
            "retentionlab_predictions_total",
            "retentionlab_inference_latency_seconds",
            "retentionlab_batch_size",
            "retentionlab_drift_checks_total",
        ],
        "ci": "Configured for Python 3.10/3.12 and Docker; remote run not observed",
    }
    (ROOT / "reports/serving_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Verified {counts['tests']} passing tests, {len(endpoints)} endpoints and frozen artifact"
    )


if __name__ == "__main__":
    main()
