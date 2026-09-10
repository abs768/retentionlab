#!/usr/bin/env python3
"""Exercise the public RetentionLab deployment and record the evidence.

A successful Vercel build is not evidence that the service works. This script
sends real requests to the public production URL, checks the response bodies
(not just status codes), and writes ``reports/deployment.json``.

Usage:
    python scripts/verify_deployment.py https://<production-url>
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "artifacts" / "model" / "model_metadata.json"
OUTPUT = ROOT / "reports" / "deployment.json"
PROJECT_NAME = "retentionlab"
TIMEOUT = 60


def request(url: str, method: str = "GET", payload: dict | None = None) -> dict[str, Any]:
    """Perform one HTTP request and return status, elapsed body, and parsed JSON."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw, status, content_type = exc.read(), exc.code, exc.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        return {"status": None, "error": str(exc.reason), "json": None, "bytes": 0}

    parsed = None
    if "application/json" in content_type:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
    return {
        "status": status,
        "content_type": content_type.split(";")[0],
        "bytes": len(raw),
        "json": parsed,
        "text_head": raw[:160].decode("utf-8", "replace") if parsed is None else None,
    }



def current_commit() -> str | None:
    """HEAD when the verification ran (the release commit's parent, if any)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def sample_customer() -> dict[str, Any]:
    return json.loads(METADATA.read_text())["smoke_input"]


def build_checks(base: str) -> list[dict[str, Any]]:
    customer = sample_customer()
    second = dict(customer, Contract="Two year", tenure=60)
    batch = {"customers": [customer, second]}
    return [
        {"name": "GET /", "method": "GET", "path": "/"},
        {"name": "GET /health", "method": "GET", "path": "/health"},
        {"name": "GET /ready", "method": "GET", "path": "/ready"},
        {"name": "GET /model-info", "method": "GET", "path": "/model-info"},
        {"name": "POST /predict", "method": "POST", "path": "/predict", "payload": customer},
        {
            "name": "POST /batch-predict",
            "method": "POST",
            "path": "/batch-predict",
            "payload": batch,
        },
        {
            "name": "POST /rank",
            "method": "POST",
            "path": "/rank",
            "payload": {"customers": [customer, second], "capacity_fraction": 0.5},
        },
        {
            "name": "POST /monitor/drift",
            "method": "POST",
            "path": "/monitor/drift",
            "payload": batch,
        },
        {"name": "GET /metrics", "method": "GET", "path": "/metrics"},
        {"name": "GET /docs", "method": "GET", "path": "/docs"},
        {"name": "GET /project-summary", "method": "GET", "path": "/project-summary"},
        {"name": "GET /static/landing.css", "method": "GET", "path": "/static/landing.css"},
        {"name": "GET /static/landing.js", "method": "GET", "path": "/static/landing.js"},
        {"name": "GET /figures/roc_curve.png", "method": "GET", "path": "/figures/roc_curve.png"},
    ]


def evaluate(name: str, result: dict[str, Any]) -> tuple[bool, str]:
    """Assert something meaningful about each endpoint's body, not just its status."""
    if result["status"] != 200:
        return False, f"HTTP {result['status']}"
    body = result["json"]
    if name == "GET /":
        ok = result["content_type"] == "text/html" and result["bytes"] > 2000
        return ok, "landing page HTML served"
    if name == "GET /health":
        return body == {"status": "ok"}, "liveness ok"
    if name == "GET /ready":
        ok = bool(body) and body.get("model_loaded") and body.get("checksum_verified")
        return ok, f"model {body.get('model_version')} loaded, checksum verified"
    if name == "GET /model-info":
        return bool(body) and "model_version" in body, f"version {body.get('model_version')}"
    if name == "POST /predict":
        p = (body or {}).get("churn_probability")
        return isinstance(p, float) and 0.0 <= p <= 1.0, f"probability {p:.6f}"
    if name == "POST /batch-predict":
        preds = (body or {}).get("predictions") or []
        ok = len(preds) == 2 and all(0.0 <= r["churn_probability"] <= 1.0 for r in preds)
        return ok, f"{len(preds)} predictions returned"
    if name == "POST /rank":
        rows = (body or {}).get("customers") or []
        selected = (body or {}).get("selected_count")
        probabilities = [row["churn_probability"] for row in rows]
        descending = probabilities == sorted(probabilities, reverse=True)
        ranks_ok = [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
        ok = bool(rows) and descending and ranks_ok
        return ok, f"{len(rows)} rows ranked by descending risk, selected_count={selected}"
    if name == "POST /monitor/drift":
        return bool(body) and "overall_status" in body, f"status {body.get('overall_status')}"
    if name == "GET /metrics":
        return result["bytes"] > 0, f"{result['bytes']} bytes of exposition"
    if name == "GET /docs":
        return result["bytes"] > 500, "OpenAPI docs page served"
    if name == "GET /project-summary":
        ok = bool(body) and bool(body.get("headline_metrics"))
        count = len((body or {}).get("headline_metrics", []))
        return ok, f"{count} headline metrics served as JSON"
    if name.startswith("GET /static/"):
        expected = "text/css" if name.endswith(".css") else "application/javascript"
        ok = result["content_type"] == expected and result["bytes"] > 500
        return ok, f"{result['content_type']}, {result['bytes']} bytes"
    if name.startswith("GET /figures/"):
        ok = result["content_type"] == "image/png" and result["bytes"] > 1000
        return ok, f"generated PNG, {result['bytes']} bytes"
    return True, "ok"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    base = sys.argv[1].rstrip("/")

    verified: list[dict[str, Any]] = []
    failures: list[str] = []
    model_version = None

    for check in build_checks(base):
        result = request(base + check["path"], check["method"], check.get("payload"))
        ok, note = evaluate(check["name"], result)
        if check["name"] == "GET /ready" and result["json"]:
            model_version = result["json"].get("model_version")
        verified.append(
            {
                "endpoint": check["name"],
                "http_status": result["status"],
                "passed": ok,
                "observation": note,
            }
        )
        print(f"{'PASS' if ok else 'FAIL'}  {check['name']:<22} {result['status']}  {note}")
        if not ok:
            failures.append(check["name"])

    report = {
        "provider": "Vercel",
        "vercel_project": PROJECT_NAME,
        "public_url": base,
        "docs_url": f"{base}/docs",
        "status": "verified" if not failures else "degraded",
        "model_version": model_version,
        "endpoints_verified": sum(1 for row in verified if row["passed"]),
        "endpoints_total": len(verified),
        "verified_endpoints": verified,
        "failures": failures,
        "verification_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verified_against_commit": current_commit(),
        "notes": [
            "Serverless instances are ephemeral and horizontally scaled: /metrics reflects "
            "one instance's in-process counters, not globally aggregated telemetry.",
            "Response times are not a production latency benchmark; they include cold starts.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{report['endpoints_verified']}/{report['endpoints_total']} endpoints verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
