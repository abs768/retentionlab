"""Synthetic shifts below are TEST FIXTURES, never evidence of production drift."""

import copy
import json

import numpy as np
from app.config import Settings
from app.main import create_app
from app.monitoring import build_profile, check_drift, load_reference, population_stability_index
from fastapi.testclient import TestClient


def test_reference_is_exact_deterministic_training_split(splits):
    stored = load_reference(Settings().reference_path)
    generated = build_profile(splits.X_train)
    assert stored["source_split"] == "train"
    assert stored["row_count"] == 4929
    assert stored["numeric"] == generated["numeric"]
    assert stored["categorical"] == generated["categorical"]
    assert stored["numeric"] != build_profile(splits.X_test)["numeric"]


def test_reference_like_data_has_zero_or_low_psi(splits):
    profile = load_reference(Settings().reference_path)
    exact = check_drift(splits.X_train, profile)
    assert all(row["psi"] == 0 for row in exact["numeric_drift"].values())
    assert exact["overall_status"] == "little_apparent_shift"
    subset = check_drift(splits.X_train.iloc[:1000], profile)
    assert all(row["psi"] < 0.1 for row in subset["numeric_drift"].values())


def test_synthetic_numeric_and_categorical_shift_warns_with_fixed_bins(splits):
    profile = load_reference(Settings().reference_path)
    original = copy.deepcopy(profile)
    shifted = splits.X_train.iloc[:500].copy()
    shifted["tenure"] = 1000
    shifted["MonthlyCharges"] = 10000.0
    shifted["Contract"] = "Two year"
    result = check_drift(shifted, profile)
    assert result["numeric_drift"]["tenure"]["psi"] >= 0.25
    assert result["categorical_drift"]["Contract"]["total_variation_distance"] >= 0.25
    assert result["overall_status"] == "substantial_apparent_shift"
    assert result["warnings"]
    assert profile == original
    assert (
        result["numeric_drift"]["tenure"]["interior_bin_edges"]
        == (original["numeric"]["tenure"]["interior_bin_edges"])
    )


def test_all_missing_total_charges_and_missingness_shift(splits):
    profile = load_reference(Settings().reference_path)
    recent = splits.X_train.iloc[:100].copy()
    recent["TotalCharges"] = np.nan
    result = check_drift(recent, profile)
    assert result["numeric_drift"]["TotalCharges"]["psi"] is None
    assert result["missingness_changes"]["TotalCharges"]["change"] > 0.99
    assert any("missingness" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_psi_zero_bins_finite_and_identity():
    assert population_stability_index([0, 1], [0, 1]) == 0
    assert np.isfinite(population_stability_index([0, 1], [1, 0]))
    assert population_stability_index([0, 1], [1, 0]) > 0.25


def test_monitoring_warning_is_not_validation_rejection(splits):
    records = json.loads(splits.X_train.iloc[:200].to_json(orient="records"))
    for record in records:
        record["tenure"] = 1000
    with TestClient(create_app()) as client:
        response = client.post("/monitor/drift", json={"customers": records})
        assert response.status_code == 200
        assert response.json()["overall_status"] == "substantial_apparent_shift"
        assert client.post("/batch-predict", json={"customers": records}).status_code == 200
        records[0]["tenure"] = -1
        assert client.post("/monitor/drift", json={"customers": records}).status_code == 422
