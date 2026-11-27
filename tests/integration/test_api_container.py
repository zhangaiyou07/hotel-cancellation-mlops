"""Integration test: sends real HTTP requests to the API running in its Docker container, with the
real champion model loaded from MLflow. Skipped unless API_URL is set, for example:
    API_URL=http://localhost:8000 pytest tests/integration
"""
import os

import httpx
import pytest

from tests.conftest import VALID_BOOKING

API_URL = os.getenv("API_URL")
pytestmark = pytest.mark.skipif(not API_URL, reason="set API_URL to test a running API container")


@pytest.fixture(scope="module")
def api():
    # generous timeout: the first prediction also loads the model from MLflow
    with httpx.Client(base_url=API_URL, timeout=60) as client:
        yield client


def test_container_is_healthy(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_real_model_returns_a_prediction(api):
    response = api.post("/predict", json=VALID_BOOKING)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["booking_id"] == "test-001"
    assert 0 <= body["cancel_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"].isdigit()  # a real registered model version, not the test model


def test_container_rejects_bad_input(api):
    response = api.post("/predict", json={**VALID_BOOKING, "lead_time": -1})
    assert response.status_code == 422


def test_container_exposes_metrics(api):
    api.post("/predict", json=VALID_BOOKING)
    response = api.get("/metrics")
    assert response.status_code == 200
    counted = [
        line for line in response.text.splitlines()
        if line.startswith("api_requests_total{") and 'endpoint="/predict"' in line and 'status="200"' in line
    ]
    assert counted, "successful /predict requests are not counted"
