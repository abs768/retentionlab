"""Repository-relative serving paths and public contract constants."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = "RetentionLab"
PUBLIC_TAGLINE = "End-to-End Customer Retention Analytics Platform"
MODEL_VERSION = "churnguard-hgb-calibrated-v1"
MAX_BATCH_SIZE = 1000
NUMERIC_DRIFT_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_DRIFT_FEATURES = ("Contract", "InternetService", "PaymentMethod")


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT / "models/final_model.joblib"
    metadata_path: Path = ROOT / "artifacts/model/model_metadata.json"
    reference_path: Path = ROOT / "artifacts/reference/reference_profile.json"
    landing_facts_path: Path = ROOT / "artifacts/landing/landing_facts.json"
