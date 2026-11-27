"""Model API tests: valid requests get a valid answer, bad input is rejected with a clear error."""
import pytest
from fastapi.testclient import TestClient

from api.main import app, risk_level

client = TestClient(app)

VALID_BOOKING = {
    "booking_id": "test-001",
    "hotel": "City Hotel",
    "arrival_date": "2017-06-10",
    "lead_time": 120,
    "adr": 95.5,
    "deposit_type": "No Deposit",
    "market_segment": "Online TA",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=VALID_BOOKING)
    assert response.status_code == 200
    body = response.json()
    assert body["booking_id"] == "test-001"
    assert 0 <= body["cancel_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


@pytest.mark.parametrize(
    "bad_field",
    [
        {"lead_time": -5},          # negative number of days
        {"hotel": "Hilton"},        # unknown hotel
        {"arrival_date": "soon"},   # not a date
        {"unknown_field": 1},       # field the API does not know
    ],
)
def test_predict_rejects_bad_input(bad_field):
    response = client.post("/predict", json={**VALID_BOOKING, **bad_field})
    assert response.status_code == 422


def test_predict_rejects_missing_required_field():
    booking = {k: v for k, v in VALID_BOOKING.items() if k != "hotel"}
    response = client.post("/predict", json=booking)
    assert response.status_code == 422


def test_risk_level_thresholds():
    assert risk_level(0.10) == "low"
    assert risk_level(0.30) == "medium"
    assert risk_level(0.59) == "medium"
    assert risk_level(0.60) == "high"
