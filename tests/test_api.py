"""Model API tests: valid requests get a sensible answer, bad input is rejected with a clear error."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import api.main as main
from api.main import app, get_model, risk_level
from api.model import LoadedModel, booking_to_frame
from api.schemas import BookingRequest
from src.features import TARGET, make_pipeline

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


@pytest.fixture(scope="module", autouse=True)
def small_model(sample_df):
    """A small model trained on a data sample, used instead of the MLflow champion in tests
    (CI has no MLflow server). It goes through exactly the same preprocessing pipeline."""
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    pipeline = make_pipeline(LogisticRegression(max_iter=2000), kind="onehot").fit(X, y)
    model = LoadedModel(pipeline=pipeline, version="test")
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.clear()


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
    assert body["model_version"] == "test"


def test_non_refundable_deposit_raises_cancel_probability():
    """Sensible prediction: in this data, non-refundable deposits are cancelled far more often."""
    no_deposit = client.post("/predict", json={**VALID_BOOKING, "deposit_type": "No Deposit"})
    non_refund = client.post("/predict", json={**VALID_BOOKING, "deposit_type": "Non Refund"})
    assert non_refund.json()["cancel_probability"] > no_deposit.json()["cancel_probability"]


def test_booking_to_frame_has_model_columns():
    frame = booking_to_frame(BookingRequest(**VALID_BOOKING))
    assert len(frame) == 1
    assert "booking_id" not in frame.columns
    assert frame.loc[0, "total_of_special_requests"] == 0  # post-booking column filled with 0
    assert pd.isna(frame.loc[0, "agent"])  # no agent -> missing, as in the raw data
    assert frame.loc[0, "arrival_date"] == pd.Timestamp("2017-06-10")


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


def test_predict_returns_503_when_model_cannot_load(monkeypatch):
    def broken_loader():
        raise RuntimeError("MLflow is not reachable")

    monkeypatch.setattr(main, "load_champion", broken_loader)
    main._cached_model.cache_clear()
    monkeypatch.delitem(app.dependency_overrides, get_model)  # use the real loader for this test
    response = client.post("/predict", json=VALID_BOOKING)
    assert response.status_code == 503
    assert "Model not available" in response.json()["detail"]


def test_risk_level_thresholds():
    assert risk_level(0.10) == "low"
    assert risk_level(0.30) == "medium"
    assert risk_level(0.59) == "medium"
    assert risk_level(0.60) == "high"
