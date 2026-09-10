# RetentionLab - verified project facts

Every figure below was read back from artifacts produced by the analytical pipeline and the serving verification/benchmark scripts. Experiment effects are assumptions; serving measurements are local.

## DATA SCIENCE

DATA:
- 7,043 customer rows, 21 raw columns (IBM Telco Customer Churn, fictional sample dataset)
- 19 model features after excluding customerID and the target
- churn prevalence 26.5%
- stratified split: 4,929 train / 1,057 validation / 1,057 test (seed 42)
- data-quality finding: TotalCharges: 11 value(s) could not be parsed as numeric and were coerced to NaN (imputed inside the training pipeline).
- duplicate customerIDs found: 0

MODEL:
- validation logistic_regression: PR-AUC 0.628, ROC-AUC 0.830, Brier 0.1408, log loss 0.4358
- validation hist_gradient_boosting: PR-AUC 0.643, ROC-AUC 0.833, Brier 0.1392, log loss 0.4312
- selected on validation PR-AUC: hist_gradient_boosting (test set untouched during selection)
- final test ROC-AUC: 0.855
- final test PR-AUC: 0.670 (test churn prevalence 26.5%)
- final test Brier score: 0.1324
- final test log loss: 0.4058
- final test accuracy 0.803, precision 0.671, recall 0.504, F1 0.576 at threshold 0.5

CALIBRATION:
- 5-fold cross-validated calibration compared against the uncalibrated model on the validation split
- validation Brier: uncalibrated 0.13919 -> sigmoid 0.13809 (improvement of 0.00110)

POLICY (retrospective prioritisation on the held-out test set, not causal):
- top 10% outreach (106 of 1057 customers): captured 80 of 280 actual churners = 28.6% capture, 75.5% precision among contacted, 2.85x lift vs random outreach
- top 20% outreach (212 of 1057 customers): captured 143 of 280 actual churners = 51.1% capture, 67.5% precision among contacted, 2.55x lift vs random outreach
- top 30% outreach (318 of 1057 customers): captured 185 of 280 actual churners = 66.1% capture, 58.2% precision among contacted, 2.20x lift vs random outreach

EXPERIMENT DESIGN (prospective; no treatment exists in this dataset):
- assumed baseline churn rate 67.5%, the observed churn rate among the customers a top-20% policy would contact
- detecting a 2 pp absolute reduction at 80% power (alpha 0.05): 8,747 customers per arm (17,494 total)
- detecting a 3 pp absolute reduction at 80% power (alpha 0.05): 3,915 customers per arm (7,830 total)
- detecting a 5 pp absolute reduction at 80% power (alpha 0.05): 1,428 customers per arm (2,856 total)
- detecting a 2 pp absolute reduction at 90% power (alpha 0.05): 11,710 customers per arm (23,420 total)
- detecting a 3 pp absolute reduction at 90% power (alpha 0.05): 5,241 customers per arm (10,482 total)
- detecting a 5 pp absolute reduction at 90% power (alpha 0.05): 1,912 customers per arm (3,824 total)

ASSOCIATION TESTS (observational, pre-specified, not causal):
- churn vs Contract: chi-square 1184.6 (df 2), p = 5.86e-258, Cramer's V 0.410
- churn vs InternetService: chi-square 732.3 (df 2), p = 9.57e-160, Cramer's V 0.322
- churn vs PaperlessBilling: chi-square 258.3 (df 1), p = 4.07e-58, Cramer's V 0.191

## ML ENGINEERING

- 12 service endpoints; maximum vectorized batch: 1000 customers
- checksum, feature-contract and startup smoke verification: passed; model churnguard-hgb-calibrated-v1
- PSI on tenure, MonthlyCharges, TotalCharges; categorical total variation on Contract, InternetService, PaymentMethod
- training-only drift reference: 4,929 rows
- 6 Prometheus metric families; instance-local counters
- executed automated tests: 89; failures 0, errors 0, skipped 0
- CI: Configured for Python 3.10/3.12 and Docker; remote run not observed
- local batch 1: direct p50/p95 14.06/14.39 ms; FastAPI TestClient 15.66/16.32 ms (not network latency)
- local batch 10: direct p50/p95 14.44/16.19 ms; FastAPI TestClient 16.11/16.68 ms (not network latency)
- local batch 100: direct p50/p95 16.58/17.17 ms; FastAPI TestClient 19.55/20.89 ms (not network latency)
- local batch 500: direct p50/p95 27.27/29.91 ms; FastAPI TestClient 34.60/36.15 ms (not network latency)
- Docker: Build attempted; Docker CLI 28.4.0 present but daemon unavailable. Build/run untested locally.
- local Uvicorn HTTP endpoints verified: 8
- no production traffic, causal retention uplift, or globally aggregated telemetry measured

## DEPLOYMENT

- deployed on Vercel (project retentionlab): https://retentionlab-rho.vercel.app
- 14 of 14 public HTTPS endpoints exercised and verified on 2026-09-10T09:16:59Z
- serving model churnguard-hgb-calibrated-v1 with startup checksum verification, from the same artifact bytes verified locally
- public response times include cold starts and are not a latency benchmark; /metrics counters are instance-local, not globally aggregated
