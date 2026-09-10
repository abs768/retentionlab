"""Serving contracts tested against the trusted frozen model and actual data."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from app.config import Settings
from app.inference import customers_to_frame
from app.main import create_app
from app.model_service import ArtifactError, ModelService, artifact_sha256
from app.schemas import RAW_FEATURES, Customer
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def sample():
    payload = json.loads(Settings().metadata_path.read_text())["smoke_input"]
    return {"customerID": "test-customer", **payload}


@pytest.fixture
def client():
    with TestClient(create_app()) as active:
        yield active


def test_health_readiness_and_public_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json()["checksum_verified"] is True
    info = client.get("/model-info").json()
    assert info["model_version"] == readiness.json()["model_version"]
    assert info["expected_feature_count"] == 19
    assert info["source_roc_auc"] == pytest.approx(0.855, abs=0.0005)
    assert "/Users/" not in json.dumps(info)
    assert "model_path" not in info


def test_artifact_load_checksum_and_feature_contract():
    settings = Settings()
    service = ModelService.load(settings)
    assert artifact_sha256(settings.model_path.read_bytes()) == service.metadata["sha256"]
    assert list(service.model.feature_names_in_) == list(RAW_FEATURES)
    assert "customerID" not in service.model.feature_names_in_


@pytest.mark.parametrize("failure", ["missing", "checksum", "corrupt", "features", "version"])
def test_unready_artifacts_fail_safely(tmp_path, failure, sample):
    settings = Settings()
    content = settings.model_path.read_bytes()
    metadata = json.loads(settings.metadata_path.read_text())
    if failure == "checksum":
        metadata["sha256"] = "0" * 64
    if failure == "corrupt":
        content = b"invalid joblib"
        metadata.update(sha256=artifact_sha256(content), artifact_size_bytes=len(content))
    if failure == "features":
        metadata["expected_raw_features"][0] = "customerID"
    if failure == "version":
        metadata["sklearn_version"] = "0.0.0"
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    if failure != "missing":
        model_path.write_bytes(content)
    metadata_path.write_text(json.dumps(metadata))
    broken = replace(settings, model_path=model_path, metadata_path=metadata_path)
    with pytest.raises(ArtifactError):
        ModelService.load(broken)
    with TestClient(create_app(broken)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json=sample)
        assert response.status_code == 503
        assert str(tmp_path) not in response.text


def test_checksum_failure_does_not_deserialize(tmp_path):
    metadata = json.loads(Settings().metadata_path.read_text())
    metadata["sha256"] = "0" * 64
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with patch("app.model_service.joblib.load") as load:
        with pytest.raises(ArtifactError, match="checksum"):
            ModelService.load(replace(Settings(), metadata_path=path))
        load.assert_not_called()


def test_load_once_at_startup(sample):
    import joblib

    with patch("app.model_service.joblib.load", wraps=joblib.load) as load:
        with TestClient(create_app()) as client:
            for _ in range(3):
                assert client.post("/predict", json=sample).status_code == 200
            assert load.call_count == 1


def test_adapter_order_exclusion_and_missing_dtype(sample):
    customer = Customer.model_validate({**sample, "TotalCharges": None})
    frame = customers_to_frame([customer])
    assert list(frame.columns) == list(RAW_FEATURES)
    assert frame.TotalCharges.dtype.kind == "f"
    assert frame.TotalCharges.isna().all()
    assert {"customerID", "Churn", "churn"}.isdisjoint(frame.columns)


def test_prediction_deterministic_id_independent_and_offline_equal(client, sample):
    first = client.post("/predict", json=sample).json()
    repeat = client.post("/predict", json=sample).json()
    changed = client.post("/predict", json={**sample, "customerID": "another-id"}).json()
    assert first == repeat
    assert first["churn_probability"] == changed["churn_probability"]
    assert 0 <= first["churn_probability"] <= 1
    offline = client.app.state.service.model.predict_proba(
        customers_to_frame([Customer.model_validate(sample)])
    )[0, 1]
    assert first["churn_probability"] == offline


@pytest.mark.parametrize(
    "changes",
    [
        {"tenure": -1},
        {"MonthlyCharges": -1},
        {"TotalCharges": -1},
        {"SeniorCitizen": 2},
        {"SeniorCitizen": True},
        {"Contract": "unknown"},
        {"Churn": "Yes"},
        {"tenure": "10"},
        {"MonthlyCharges": None},
    ],
)
def test_invalid_inputs_rejected(client, sample, changes):
    response = client.post("/predict", json={**sample, **changes})
    assert response.status_code == 422
    assert "test-customer" not in response.text


def test_nonfinite_and_malformed_json_rejected(client, sample):
    for value in (float("nan"), float("inf"), -float("inf")):
        for field in ("TotalCharges", "MonthlyCharges", "tenure"):
            response = client.post(
                "/predict",
                content=json.dumps({**sample, field: value}),
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422
    assert client.post("/predict", content="{invalid").status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_raw_blank_numeric_text_and_unusual_valid_values(client, sample):
    for charges in (None, " ", "42.50", 1e9):
        response = client.post("/predict", json={**sample, "TotalCharges": charges})
        assert response.status_code == 200
    response = client.post("/predict", json={**sample, "tenure": 1000, "MonthlyCharges": 1e9})
    assert response.status_code == 200


def test_batch_vectorization_order_and_single_record(client, sample):
    customers = [{**sample, "customerID": str(i), "tenure": i} for i in range(10)]
    model = client.app.state.service.model
    with patch.object(model, "predict_proba", wraps=model.predict_proba) as predict:
        response = client.post("/batch-predict", json={"customers": customers})
        assert response.status_code == 200
        assert predict.call_count == 1
        assert len(predict.call_args.args[0]) == 10
    result = response.json()
    assert [p["customer_id"] for p in result["predictions"]] == [str(i) for i in range(10)]
    single = client.post("/batch-predict", json={"customers": [sample]}).json()
    assert single["predictions"][0] == client.post("/predict", json=sample).json()


def test_batch_boundaries(client, sample):
    for size in (0, 1001):
        for endpoint in ("/batch-predict", "/monitor/drift", "/rank"):
            body = {"customers": [sample] * size}
            if endpoint == "/rank":
                body["capacity_fraction"] = 0.2
            assert client.post(endpoint, json=body).status_code == 422
    assert client.post("/batch-predict", json={"customers": [sample] * 1000}).status_code == 200


def test_rank_descending_capacity_and_ties(client, sample):
    customers = [{**sample, "customerID": str(i), "tenure": i} for i in range(11)]
    for capacity, count in (
        ({"top_n": 4}, 4),
        ({"capacity_fraction": 0.2}, 3),
        ({"top_n": 0}, 0),
        ({"capacity_fraction": 1.0}, 11),
    ):
        result = client.post("/rank", json={"customers": customers, **capacity}).json()
        scores = [row["churn_probability"] for row in result["customers"]]
        assert scores == sorted(scores, reverse=True)
        assert sum(row["selected_for_outreach"] for row in result["customers"]) == count
        assert result["selected_count"] == count
        assert [row["rank"] for row in result["customers"]] == list(range(1, 12))
    tied = [{**sample, "customerID": str(i)} for i in range(100)]
    result = client.post("/rank", json={"customers": tied, "capacity_fraction": 0.07}).json()
    assert result["selected_count"] == 7
    assert [row["customer_id"] for row in result["customers"]] == [str(i) for i in range(100)]
    assert (
        client.post("/rank", json={"customers": tied, "capacity_fraction": 0.2}).json()[
            "selected_count"
        ]
        == 20
    )


def test_bad_capacity_rejected(client, sample):
    for capacity in (
        {},
        {"top_n": 1, "capacity_fraction": 0.2},
        {"top_n": -1},
        {"top_n": 2},
        {"top_n": 1.2},
        {"top_n": True},
        {"capacity_fraction": -0.1},
        {"capacity_fraction": 1.1},
        {"capacity_fraction": "NaN"},
    ):
        assert client.post("/rank", json={"customers": [sample], **capacity}).status_code == 422


def test_inference_failure_bad_probabilities_and_metrics(client, sample, caplog):
    model = client.app.state.service.model
    for bad in (
        RuntimeError("private payload"),
        np.array([[np.nan, 0.2]]),
        np.array([[-0.1, 1.1]]),
        np.array([[0.5, 0.6]]),
    ):
        kwargs = {"side_effect": bad} if isinstance(bad, Exception) else {"return_value": bad}
        with patch.object(model, "predict_proba", **kwargs):
            response = client.post("/predict", json=sample)
            assert response.status_code == 503
            assert "private payload" not in response.text
    assert "private payload" not in caplog.text
    assert sample["customerID"] not in caplog.text
    assert client.app.state.metrics.errors.labels(endpoint="/predict")._value.get() == 4


def test_metrics_increment_and_no_customer_labels(client, sample):
    client.post("/predict", json=sample)
    client.post("/predict", json={**sample, "tenure": -1})
    client.post("/batch-predict", json={"customers": [sample] * 3})
    client.post("/monitor/drift", json={"customers": [sample]})
    text = client.get("/metrics").text
    assert 'churnguard_inference_requests_total{endpoint="/predict"} 2.0' in text
    assert 'churnguard_inference_errors_total{endpoint="/predict"} 1.0' in text
    assert "churnguard_predictions_total 4.0" in text
    assert "churnguard_drift_checks_total 1.0" in text
    assert "churnguard_inference_latency_seconds_count" in text
    assert "churnguard_batch_size_count 2.0" in text
    assert sample["customerID"] not in text
    for metric in client.app.state.metrics.registry.collect():
        for row in metric.samples:
            assert set(row.labels).issubset({"endpoint", "le"})


def test_frozen_test_metrics_unchanged(splits, client):
    from churnguard.config import REPORTS_DIR
    from churnguard.evaluation import compute_metrics

    actual = compute_metrics(
        splits.y_test.to_numpy(), client.app.state.service.model.predict_proba(splits.X_test)[:, 1]
    )
    expected = json.loads((REPORTS_DIR / "tables/test_metrics.json").read_text())
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value, abs=1e-12)
