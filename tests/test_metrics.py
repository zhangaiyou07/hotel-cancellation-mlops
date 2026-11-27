"""/metrics tests: Prometheus can read the endpoint, and requests and predictions are counted."""
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from api.main import app
from api.metrics import endpoint_label
from tests.conftest import VALID_BOOKING

client = TestClient(app)


def count(name: str, **labels) -> float:
    """Current value of one metric (0 if it has not been recorded yet)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_is_readable():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_request_duration_seconds" in response.text


def test_successful_prediction_is_counted(small_model):
    labels = {"method": "POST", "endpoint": "/predict", "status": "200"}
    before = count("api_requests_total", **labels)
    timed_before = count("api_request_duration_seconds_count", method="POST", endpoint="/predict")
    predictions_before = sum(count("predictions_total", risk_level=r) for r in ("low", "medium", "high"))

    assert client.post("/predict", json=VALID_BOOKING).status_code == 200

    assert count("api_requests_total", **labels) == before + 1
    assert count("api_request_duration_seconds_count", method="POST", endpoint="/predict") == timed_before + 1
    predictions_after = sum(count("predictions_total", risk_level=r) for r in ("low", "medium", "high"))
    assert predictions_after == predictions_before + 1


def test_bad_input_is_counted_as_422(small_model):
    labels = {"method": "POST", "endpoint": "/predict", "status": "422"}
    before = count("api_requests_total", **labels)
    client.post("/predict", json={**VALID_BOOKING, "lead_time": -1})
    assert count("api_requests_total", **labels) == before + 1


def test_unknown_paths_share_one_label():
    assert endpoint_label("/predict") == "/predict"
    assert endpoint_label("/wp-admin") == "other"
    assert endpoint_label("/random/123") == "other"
