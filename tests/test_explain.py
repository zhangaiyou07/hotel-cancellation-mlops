"""Explanation tests: /predict returns the top factors behind each prediction, they add up to the
model's output, and a model we cannot explain still gets a prediction."""
import math

from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api.explain import feature_contributions, top_factors
from api.main import app, get_model
from api.model import LoadedModel, booking_to_frame
from api.schemas import BookingRequest
from src.features import TARGET, make_pipeline
from tests.conftest import VALID_BOOKING

client = TestClient(app)


def test_predict_returns_top_factors(small_model):
    factors = client.post("/predict", json=VALID_BOOKING).json()["top_factors"]
    assert 1 <= len(factors) <= 5
    assert set(factors[0]) == {"feature", "value", "impact"}
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)  # biggest impact first


def test_non_refundable_deposit_is_a_top_reason_to_cancel(small_model):
    booking = {**VALID_BOOKING, "deposit_type": "Non Refund"}
    factors = client.post("/predict", json=booking).json()["top_factors"]
    deposit = next(f for f in factors if f["feature"] == "deposit_type")
    assert deposit["value"] == "Non Refund"
    assert deposit["impact"] > 0  # pushes towards cancellation


def test_contributions_add_up_to_the_prediction(small_model):
    """Like SHAP values: intercept + all contributions = the model's log-odds."""
    frame = booking_to_frame(BookingRequest(**VALID_BOOKING))
    probability = small_model.pipeline.predict_proba(frame)[0, 1]
    contributions = feature_contributions(small_model.pipeline, frame)
    total = small_model.pipeline[-1].intercept_[0] + sum(contributions.values())
    assert math.isclose(total, math.log(probability / (1 - probability)), abs_tol=1e-6)


def test_one_hot_columns_are_grouped_back_to_their_feature(small_model):
    frame = booking_to_frame(BookingRequest(**VALID_BOOKING))
    features = [f["feature"] for f in top_factors(small_model.pipeline, frame, k=100)]
    assert "hotel" in features
    assert not any(name.startswith("hotel_") for name in features)
    assert len(features) == len(set(features))


def test_unexplainable_model_still_predicts(sample_df, monkeypatch):
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    dummy = LoadedModel(pipeline=make_pipeline(DummyClassifier(), kind="onehot").fit(X, y), version="dummy")
    monkeypatch.setitem(app.dependency_overrides, get_model, lambda: dummy)

    response = client.post("/predict", json=VALID_BOOKING)

    assert response.status_code == 200
    assert response.json()["top_factors"] == []
