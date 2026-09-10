# ChurnGuard — ML Engineering Completion Audit

1. **Baseline:** clean repository at `fe855b81bc6397d8bc375418be011689111ff83e`; 37 tests passed, Ruff clean, 107/107 numeric checks passed.
2. **Analytical preservation:** original modeling, training, calibration, evaluation, policy and experiment code is unchanged. The clean Python 3.10 pipeline rerun reproduced the complete run summary and rankings. Its maximum probability difference was floating-point roundoff (`3.33e-16`). The original model and analytical evidence were retained byte-for-byte; only resume facts were extended.
3. **Implementation:** added `app/` modules for configuration, typed schemas, model service, the shared adapter, ranking, monitoring, metrics and FastAPI lifespan/routes. Added metadata/reference artifacts; export, download, verification, documentation and benchmark scripts; serving and monitoring tests; Docker files; CI checks; and source/data licensing documentation.
4. **Endpoints:** 8 implemented and tested through both TestClient and local Uvicorn HTTP:

   - `GET /health`
   - `GET /ready`
   - `GET /model-info`
   - `POST /predict`
   - `POST /batch-predict`
   - `POST /rank`
   - `POST /monitor/drift`
   - `GET /metrics`

5. **Batch limit:** 1000 customers for batch prediction, ranking and drift. Empty and oversized batches are rejected. Predictions use one vectorized call and preserve input order.
6. **Artifact:** `churnguard-hgb-calibrated-v1`, 1,149,743 bytes; SHA256 `cf3addc6f6c93474c8bf7fe1f5f875bacd815df2c6d87a061170c1d76c7af581`. Startup verifies checksum before trusted deserialization, runtime version, feature names/count, identifier exclusion, class order and finite smoke probabilities. SHA256 does not make untrusted pickle safe.
7. **Drift:** PSI on tenure, MonthlyCharges and TotalCharges; categorical total variation on Contract, InternetService and PaymentMethod. The reference uses 4,929 deterministic training rows only, with fixed bins and separate missingness checks. Thresholds are practical heuristics; synthetic shifts are test fixtures only.
8. **Observability:** standard startup/error/drift logging without customer payloads. Six Prometheus families:

   - `churnguard_inference_requests_total`
   - `churnguard_inference_errors_total`
   - `churnguard_predictions_total`
   - `churnguard_inference_latency_seconds`
   - `churnguard_batch_size`
   - `churnguard_drift_checks_total`

9. **Tests:** 72 passed, zero failed/errors/skipped: original 37 plus 35 serving/monitoring cases. Verified in a fresh Python 3.10.20 environment after `pip install -e ".[dev]"`. `pip check` and Ruff pass.
10. **README verification:** 142/142 checks pass, including the retained analytical checks and parsed serving/benchmark claims.
11. **Benchmark:** local sequential measurements on Python 3.10.20, sklearn 1.6.1, Darwin 25.5.0 arm64; 5 warm-ups and 50 timed iterations per configuration. Native thread counts: {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}. These are not production-network latency.

| Batch | Direct p50 / p95 (ms) | FastAPI TestClient p50 / p95 (ms) |
| --- | --- | --- |
| 1 | 14.06 / 14.39 | 15.66 / 16.32 |
| 10 | 14.44 / 16.19 | 16.11 / 16.68 |
| 100 | 16.58 / 17.17 | 19.55 / 20.89 |
| 500 | 27.27 / 29.91 | 34.60 / 36.15 |

12. **Docker:** Build attempted; Docker CLI 28.4.0 present but daemon unavailable. Build/run untested locally. Dockerfile uses Python 3.10 slim, a non-root user, packaged artifacts, readiness healthcheck and Uvicorn without reload.
13. **CI:** Python 3.10/3.12 matrix installs/imports, downloads/checks data, lints, tests, verifies the model and audits README claims. A separate job builds Docker and checks container endpoints. YAML syntax parsed locally. Remote Actions were not run; Python 3.12 is not locally tested.
14. **Deployment:** Not deployed: Vercel CLI unavailable; authenticated CLI session could not be checked. No public URL or deployed endpoint is claimed.
15. **Dataset/license:** conservative removal of the raw CSV from the current tracked tree; the local CSV is retained and a pinned upstream downloader verifies its original checksum. IBM’s [upstream notice](https://github.com/IBM/telco-customer-churn-on-icp4d#license) licenses the code pattern; this review did not establish a separate dataset redistribution grant. Original project code has an MIT license. The initial commit still contains the CSV, so rights must be resolved before publishing that history.
16. **Security/portability:** no credentials or personal absolute paths found in the proposed tracked files, including the binary model. Only benign ignore/environment-variable and negative-test matches occurred. Artifact paths derive from repository location. Customer payloads are not logged, and identifiers/raw values are not metric labels.
17. **Git:** one new commit is the final operation; its hash is recorded in the completion message. The original commit is preserved. No push was requested or performed.
18. **Remaining limits:** Docker build/run and remote CI still need execution; Python 3.12 is not locally tested; no Vercel deployment. The demo is stateless with process-local counters, no authentication/rate limiting, no production load or availability evidence, no temporal/outcome drift validation and no measured causal retention uplift. Publishing the original dataset-containing history requires resolution of data rights.

Generated evidence is retained in `reports/serving_verification.json`, `reports/serving_benchmark.json`, `reports/runtime_validation.json`, `reports/deployment.json`, `reports/analytical_preservation.json`, and `reports/security_portability_audit.json`.
