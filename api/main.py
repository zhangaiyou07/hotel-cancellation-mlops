"""Model API: receives one hotel booking and returns the probability that it will be cancelled."""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from api.model import LoadedModel, load_champion
from api.schemas import BookingRequest, PredictionResponse

app = FastAPI(title="Hotel Cancellation Model API", version="0.2.0")


@lru_cache(maxsize=1)
def _cached_model() -> LoadedModel:
    return load_champion()


def get_model() -> LoadedModel:
    """Load the champion model on the first request and reuse it afterwards."""
    try:
        return _cached_model()
    except Exception as exc:
        # e.g. MLflow is not reachable or no model is registered yet: answer clearly instead of crashing
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc


def risk_level(probability: float) -> str:
    """Turn a probability into a label the hotel staff can act on."""
    if probability >= 0.6:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


@app.get("/health")
def health() -> dict:
    """Lets docker-compose and the tests check that the service is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    booking: BookingRequest,
    model: Annotated[LoadedModel, Depends(get_model)],
) -> PredictionResponse:
    """Predict the cancellation probability of one booking with the champion model."""
    probability = model.predict_proba(booking)
    return PredictionResponse(
        booking_id=booking.booking_id,
        cancel_probability=round(probability, 4),
        risk_level=risk_level(probability),
        model_version=model.version,
    )
