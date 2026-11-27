"""Prediction logging tests: every prediction is saved with its booking_id, and a database problem
never stops the API from answering."""
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import select

import api.main as main
from api.main import app
from api.store import predictions
from tests.conftest import VALID_BOOKING

client = TestClient(app)


def saved_rows(store) -> list[dict]:
    with store.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(predictions)).mappings()]


def test_prediction_is_saved_with_booking_id(small_model, prediction_store):
    body = client.post("/predict", json=VALID_BOOKING).json()

    rows = saved_rows(prediction_store)
    assert len(rows) == 1
    row = rows[0]
    assert row["booking_id"] == "test-001"
    assert row["model_version"] == "test"
    assert row["cancel_probability"] == body["cancel_probability"]
    assert row["risk_level"] == body["risk_level"]
    assert row["features"]["hotel"] == "City Hotel"
    assert row["features"]["arrival_date"] == "2017-06-10"
    assert "booking_id" not in row["features"]


def test_missing_booking_id_is_generated(small_model, prediction_store):
    booking = {k: v for k, v in VALID_BOOKING.items() if k != "booking_id"}
    body = client.post("/predict", json=booking).json()

    assert body["booking_id"]  # the API made one up
    assert saved_rows(prediction_store)[0]["booking_id"] == body["booking_id"]


def test_database_failure_does_not_break_prediction(small_model, monkeypatch):
    class BrokenStore:
        def save(self, *args, **kwargs):
            raise ConnectionError("database is down")

    monkeypatch.setattr(main, "get_store", lambda: BrokenStore())
    errors_before = REGISTRY.get_sample_value("prediction_log_errors_total") or 0.0

    response = client.post("/predict", json=VALID_BOOKING)

    assert response.status_code == 200
    assert REGISTRY.get_sample_value("prediction_log_errors_total") == errors_before + 1
