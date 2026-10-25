"""Serving contracts tested against the trusted frozen model and actual data."""

import json
import math
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
    assert 'retentionlab_inference_requests_total{endpoint="/predict"} 2.0' in text
    assert 'retentionlab_inference_errors_total{endpoint="/predict"} 1.0' in text
    assert "retentionlab_predictions_total 4.0" in text
    assert "retentionlab_drift_checks_total 1.0" in text
    assert "retentionlab_inference_latency_seconds_count" in text
    assert "retentionlab_batch_size_count 2.0" in text
    assert sample["customerID"] not in text
    for metric in client.app.state.metrics.registry.collect():
        for row in metric.samples:
            assert set(row.labels).issubset({"endpoint", "le"})


def _facts():
    return json.loads(Settings().landing_facts_path.read_text())


def test_landing_page_renders_all_sections(client):
    """GET / serves the recruiter page with every documented section anchor."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "RetentionLab" in body
    assert "End-to-End Customer Retention Analytics Platform" in body
    for anchor in (
        "overview", "problem", "performance", "targeting", "demo",
        "experiment", "architecture", "engineering", "monitoring", "limitations",
    ):
        assert f'id="{anchor}"' in body, anchor


def test_landing_metrics_match_canonical_run_summary(client):
    """Every headline number on the page traces back to reports/run_summary.json."""
    root = Settings().landing_facts_path.parents[2]
    summary = json.loads((root / "reports" / "run_summary.json").read_text())
    top20 = next(r for r in summary["policy"] if abs(r["capacity_fraction"] - 0.20) < 1e-9)
    body = client.get("/").text

    assert f"{summary['n_rows']:,}" in body
    assert f"{summary['test_metrics']['roc_auc']:.3f}" in body
    assert f"{summary['test_metrics']['pr_auc']:.3f}" in body
    assert f"{top20['churner_capture_rate'] * 100:.1f}%" in body
    assert f"{top20['lift_vs_random']:.2f}x" in body


def test_landing_form_covers_the_full_model_contract(client):
    """The demo form must expose every raw feature except the derived TotalCharges."""
    body = client.get("/").text
    for feature in RAW_FEATURES:
        if feature == "TotalCharges":
            assert "derived-total" in body
            continue
        assert f'data-field="{feature}"' in body, feature


def test_landing_form_options_come_from_the_pydantic_schema(client):
    """Categorical dropdowns must offer exactly the values the API accepts."""
    body = client.get("/").text
    for value in ("Month-to-month", "Two year", "Fiber optic", "No internet service",
                  "Bank transfer (automatic)", "No phone service"):
        assert f'<option value="{value}"' in body, value


def test_landing_page_does_not_leak_local_paths(client):
    """The public page must never expose developer filesystem paths."""
    body = client.get("/").text
    assert "/Users/" not in body
    assert "bhavanishankar" not in body


def test_static_assets_are_served_with_correct_types(client):
    for path, content_type in (
        ("/static/landing.css", "text/css"),
        ("/static/landing.js", "application/javascript"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert len(response.content) > 500


def test_generated_figures_are_served(client):
    """Figures shown on the page are the pipeline's own PNG artifacts."""
    for figure in _facts()["figures"]:
        response = client.get(f"/figures/{figure['file']}")
        assert response.status_code == 200, figure["file"]
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    "path",
    [
        "/static/../config.py",
        "/static/../../README.md",
        "/figures/../../README.md",
        "/static/nope.txt",
        "/figures/nope.png",
    ],
)
def test_asset_routes_reject_traversal_and_unknown_files(client, path):
    assert client.get(path).status_code == 404


def test_project_summary_matches_the_rendered_landing_facts(client):
    """The JSON endpoint and the page must quote the same generated evidence."""
    response = client.get("/project-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload == _facts()
    assert payload["headline_metrics"]
    assert payload["engineering"]["tests"] > 0


def test_landing_sample_customers_are_valid_model_inputs(client):
    """Every demo record must satisfy the real schema and score successfully."""
    facts = _facts()
    records = [e["customer"] for e in facts["example_customers"]]
    records += [c["customer"] for c in facts["ranking_cohort"]]
    assert len(records) >= 12
    for record in records:
        # No target may ever appear in a demo input.
        assert "Churn" not in record
        assert set(record) == set(RAW_FEATURES)
        Customer.model_validate(record)
    response = client.post("/batch-predict", json={"customers": records})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == len(records)


def test_ranking_cohort_ranks_end_to_end(client):
    """The ranking demo's payload works against the live /rank contract."""
    facts = _facts()
    payload = {
        "customers": [
            dict(entry["customer"], customerID=entry["id"])
            for entry in facts["ranking_cohort"]
        ],
        "capacity_fraction": 0.20,
    }
    response = client.post("/rank", json=payload)
    assert response.status_code == 200
    body = response.json()
    probabilities = [row["churn_probability"] for row in body["customers"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["selected_count"] == math.ceil(0.20 * len(payload["customers"]))


def test_landing_claim_count_matches_verifier_artifact():
    """The documentation-integrity card must quote the verifier's recorded result."""
    root = Settings().landing_facts_path.parents[2]
    record_path = root / "reports" / "claim_verification.json"
    if not record_path.exists():
        pytest.skip("claim verification has not been recorded yet")
    record = json.loads(record_path.read_text())
    claimed = _facts()["engineering"]["readme_claims"]
    if record["all_passed"]:
        assert claimed == record["total_claims"]
    else:
        assert claimed is None


REPOSITORY_URL = "https://github.com/abs768/retentionlab"


def test_landing_links_to_the_public_repository(client):
    """The GitHub entry points must target the real repository, not a placeholder."""
    facts = _facts()
    assert facts["repository_url"] == REPOSITORY_URL

    body = client.get("/").text
    assert body.count(f'href="{REPOSITORY_URL}"') >= 2, "expect nav/hero and footer links"
    # External links must not leak referrer or expose the opener.
    for fragment in body.split(f'href="{REPOSITORY_URL}"')[1:]:
        attrs = fragment[:120]
        assert 'target="_blank"' in attrs
        assert "noopener" in attrs and "noreferrer" in attrs


def test_project_summary_exposes_the_repository_url(client):
    assert client.get("/project-summary").json()["repository_url"] == REPOSITORY_URL
