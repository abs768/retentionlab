#!/usr/bin/env python3
"""Local model and FastAPI TestClient timings; these are NOT network latency."""

import argparse
import json
import os
import platform
from time import perf_counter

# Bound native thread pools before importing NumPy/sklearn; record these settings below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from app.config import ROOT, Settings  # noqa: E402
from app.inference import customers_to_frame  # noqa: E402
from app.main import create_app  # noqa: E402
from app.model_service import ModelService  # noqa: E402
from app.schemas import Customer  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from churnguard import data  # noqa: E402


def measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    elapsed = []
    for _ in range(iterations):
        start = perf_counter()
        operation()
        elapsed.append((perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(elapsed)),
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 20 or args.warmup < 1:
        parser.error("Use at least 20 iterations and one warm-up")
    service = ModelService.load(Settings())
    training = data.make_splits(data.prepare_dataset(data.load_raw_data())).X_train
    result = {
        "benchmark_method": "Local direct inference and FastAPI TestClient /batch-predict",
        "direct_scope": "predict_proba on a prebuilt DataFrame; no loading or adapter timing",
        "testclient_scope": "JSON, validation, adapter, inference, response; no network",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "native_threads": {
            key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "model_version": service.metadata["model_version"],
        "artifact_sha256": service.metadata["sha256"],
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "results": [],
    }
    with TestClient(create_app()) as client:
        for size in (1, 10, 100, 500):
            records = json.loads(training.head(size).to_json(orient="records"))
            frame = customers_to_frame([Customer.model_validate(record) for record in records])

            def http_request(records=records, size=size):
                response = client.post("/batch-predict", json={"customers": records})
                response.raise_for_status()
                assert response.json()["total_customers"] == size

            direct = measure(
                lambda frame=frame: service.model.predict_proba(frame), args.warmup, args.iterations
            )
            http = measure(http_request, args.warmup, args.iterations)
            result["results"].append(
                {"batch_size": size, "direct_model": direct, "fastapi_testclient": http}
            )
            print(
                f"Batch {size}: direct p50={direct['p50_ms']:.2f} ms, "
                f"TestClient p50={http['p50_ms']:.2f} ms"
            )
    (ROOT / "reports/serving_benchmark.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
