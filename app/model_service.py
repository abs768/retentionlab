"""Load only a trusted local artifact, after checking its integrity and contract."""

import hashlib
import io
import json
import logging
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from app.config import Settings
from app.inference import checked_probabilities, customers_to_frame
from app.schemas import RAW_FEATURES, Customer

logger = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """A safe startup error without artifact paths or unpickling details."""


def artifact_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class ModelService:
    model: object
    metadata: dict
    checksum_verified: bool = True

    @classmethod
    def load(cls, settings: Settings):
        try:
            metadata = json.loads(settings.metadata_path.read_text())
            content = settings.model_path.read_bytes()
            if artifact_sha256(content) != metadata["sha256"]:
                raise ArtifactError("Model checksum mismatch")
            if len(content) != metadata["artifact_size_bytes"]:
                raise ArtifactError("Model artifact size mismatch")
            if metadata["sklearn_version"] != sklearn.__version__:
                raise ArtifactError("Model and runtime scikit-learn versions differ")
            logger.info("Model artifact integrity verified")
            # A checksum is NOT a sandbox for pickle. Only locally trusted artifacts belong here.
            # Deserialize the same bytes that were hashed (avoids a path replacement race).
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                model = joblib.load(io.BytesIO(content))
            expected = list(RAW_FEATURES)
            if metadata["expected_raw_features"] != expected:
                raise ArtifactError("Metadata feature contract mismatch")
            if list(model.feature_names_in_) != expected or model.n_features_in_ != len(expected):
                raise ArtifactError("Model feature contract mismatch")
            if metadata["expected_feature_count"] != len(expected):
                raise ArtifactError("Metadata feature count mismatch")
            if metadata["excluded_identifier_fields"] != ["customerID"]:
                raise ArtifactError("Identifier exclusion contract mismatch")
            if metadata["target_name"] != "Churn" or "customerID" in expected:
                raise ArtifactError("Target or identifier contract mismatch")
            if not np.array_equal(model.classes_, [0, 1]):
                raise ArtifactError("Model class order mismatch")
            estimator_type = type(model.estimator.named_steps["model"]).__name__
            if metadata["estimator_type"] != estimator_type:
                raise ArtifactError("Estimator type mismatch")
            if metadata["model_type"] != type(model).__name__:
                raise ArtifactError("Model type mismatch")
            if not metadata["model_version"] or not metadata["model_name"]:
                raise ArtifactError("Missing model identity")
            for name in ("source_roc_auc", "source_pr_auc", "source_brier_score"):
                if not 0 <= float(metadata[name]) <= 1:
                    raise ArtifactError("Invalid source metric")
            smoke = Customer.model_validate(metadata["smoke_input"])
            checked_probabilities(model, customers_to_frame([smoke]))
            logger.info("Loaded model version %s", metadata["model_version"])
            return cls(model=model, metadata=metadata)
        except ArtifactError:
            raise
        except Exception:
            raise ArtifactError("Unable to load trusted model artifact and metadata") from None

    def public_info(self) -> dict:
        keys = (
            "model_version",
            "model_type",
            "estimator_type",
            "expected_feature_count",
            "sklearn_version",
            "sha256",
            "source_roc_auc",
            "source_pr_auc",
            "source_brier_score",
        )
        return {key: self.metadata[key] for key in keys}
