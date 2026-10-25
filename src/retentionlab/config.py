"""Central configuration: paths, seeds, and analysis constants.

Every tunable constant used by the pipeline lives here so that a reviewer can
see the full experimental setup in one screen.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_DATA_PATH: Path = RAW_DATA_DIR / "telco_customer_churn.csv"
DATA_SOURCE_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --- Schema --------------------------------------------------------------
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"
TARGET_LABEL_COLUMN: str = "churn"  # 0/1 encoded target produced by data.py
POSITIVE_LABEL: str = "Yes"
NEGATIVE_LABEL: str = "No"

# Columns that are stored as text in the raw CSV but are numeric in meaning.
NUMERIC_COERCION_COLUMNS: tuple[str, ...] = ("TotalCharges",)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)
EXPECTED_ROW_COUNT: int = 7043

# --- Splitting -----------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.15
VALIDATION_SIZE: float = 0.15  # fraction of the *full* dataset

# --- Modeling ------------------------------------------------------------
CLASSIFICATION_THRESHOLD: float = 0.5
CALIBRATION_CV_FOLDS: int = 5
CALIBRATION_METHODS: tuple[str, ...] = ("sigmoid", "isotonic")

# --- Retention policy ----------------------------------------------------
POLICY_CAPACITY_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.30)

# --- Experiment design ---------------------------------------------------
EXPERIMENT_ALPHA: float = 0.05
EXPERIMENT_POWERS: tuple[float, ...] = (0.80, 0.90)
EXPERIMENT_ABSOLUTE_REDUCTIONS: tuple[float, ...] = (0.02, 0.03, 0.05)
MDE_SAMPLE_SIZES: tuple[int, ...] = (1_000, 2_500, 5_000, 10_000, 25_000)

# --- Pre-specified inferential comparisons (section 11) ------------------
CHI_SQUARE_FEATURES: tuple[str, ...] = (
    "Contract",
    "InternetService",
    "PaperlessBilling",
)
