"""Loads the champion model from the MLflow Model Registry and turns a booking into model input."""
import os
from dataclasses import dataclass

import pandas as pd

from api.explain import top_factors
from api.schemas import BookingRequest

MODEL_NAME = os.getenv("MODEL_NAME", "hotel-cancellation-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Columns that are only known after the booking is created. The model drops them, but its
# feature-engineering step still expects them to exist, so we fill them with 0.
POST_BOOKING_DEFAULTS = {
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "booking_changes": 0,
}


def booking_to_frame(booking: BookingRequest) -> pd.DataFrame:
    """Turn one validated booking into the one-row table the model was trained on."""
    row = booking.model_dump(exclude={"booking_id"})
    row["arrival_date"] = pd.Timestamp(row["arrival_date"])
    row["country"] = row["country"].upper()
    for col in ("agent", "company"):
        if row[col] is None:
            row[col] = float("nan")  # missing in the raw data means "no agent / no company"
    row.update(POST_BOOKING_DEFAULTS)
    return pd.DataFrame([row])


@dataclass
class LoadedModel:
    """The trained scikit-learn pipeline plus the registry version it came from."""

    pipeline: object
    version: str

    def predict_proba(self, booking: BookingRequest) -> float:
        return float(self.pipeline.predict_proba(booking_to_frame(booking))[0, 1])

    def explain(self, booking: BookingRequest) -> list[dict]:
        """The features that pushed this booking's probability up or down the most."""
        return top_factors(self.pipeline, booking_to_frame(booking))


def load_champion() -> LoadedModel:
    """Load the model version that the 'champion' alias points to."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    return LoadedModel(pipeline=pipeline, version=str(version))
