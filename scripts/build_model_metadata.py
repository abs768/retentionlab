#!/usr/bin/env python3
"""Export metadata for the EXISTING trusted final model; does not fit any model."""

import io
import json
import platform
import warnings

import joblib
import sklearn
from app.config import MODEL_VERSION, ROOT, Settings
from app.model_service import ModelService, artifact_sha256
from app.schemas import RAW_FEATURES
from sklearn.exceptions import InconsistentVersionWarning

from retentionlab import data


def main():
    settings = Settings()
    content = settings.model_path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(io.BytesIO(content))  # Only the repository's trusted pipeline output.
    if list(model.feature_names_in_) != list(RAW_FEATURES):
        raise ValueError("Artifact feature contract mismatch")
    summary = json.loads((ROOT / "reports/run_summary.json").read_text())
    training = data.make_splits(data.prepare_dataset(data.load_raw_data())).X_train
    # Use an actual training row without its identifier or target for startup smoke inference.
    smoke = json.loads(training.head(1).to_json(orient="records"))[0]
    metadata = {
        "model_name": summary["selected_model"],
        "model_version": MODEL_VERSION,
        "model_type": type(model).__name__,
        "estimator_type": type(model.estimator.named_steps["model"]).__name__,
        "calibration_method": model.method,
        "expected_raw_features": list(model.feature_names_in_),
        "expected_feature_count": int(model.n_features_in_),
        "excluded_identifier_fields": ["customerID"],
        "target_name": "Churn",
        "sha256": artifact_sha256(content),
        "artifact_size_bytes": len(content),
        "python_version": platform.python_version(),
        "python_version_note": "Metadata export runtime; joblib does not record training Python",
        "sklearn_version": sklearn.__version__,
        "source_roc_auc": summary["test_metrics"]["roc_auc"],
        "source_pr_auc": summary["test_metrics"]["pr_auc"],
        "source_brier_score": summary["test_metrics"]["brier_score"],
        "source_metrics": "reports/run_summary.json",
        "smoke_input": smoke,
    }
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    ModelService.load(settings)
    print(f"Verified metadata for {MODEL_VERSION}: {metadata['sha256']}")


if __name__ == "__main__":
    main()
