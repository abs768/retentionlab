# ChurnGuard - verified project facts

Every figure below was read back from artifacts produced by `python scripts/run_pipeline.py`. Nothing here is estimated or illustrative.

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

ENGINEERING:
- 37 pytest tests, all passing
- single reproducibility command: `python scripts/run_pipeline.py`
- GitHub Actions CI installs dependencies and runs the test suite
- artifacts produced by one run: 20
  - data/processed/test.csv
  - data/processed/train.csv
  - data/processed/validation.csv
  - models/final_model.joblib
  - reports/figures/calibration_curve.png
  - reports/figures/confusion_matrix.png
  - reports/figures/cumulative_gains.png
  - reports/figures/precision_recall_curve.png
  - reports/figures/roc_curve.png
  - reports/resume_facts.md
  - reports/run_summary.json
  - reports/tables/calibration_comparison.csv
  - reports/tables/chi_square_details.json
  - reports/tables/chi_square_tests.csv
  - reports/tables/experiment_design.csv
  - reports/tables/retention_policy.csv
  - reports/tables/test_metrics.csv
  - reports/tables/test_metrics.json
  - reports/tables/validation_model_comparison.csv
  - reports/validation_report.json
