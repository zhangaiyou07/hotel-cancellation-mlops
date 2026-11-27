"""Model API: receives one hotel booking and returns the probability that it will be cancelled."""
from fastapi import FastAPI

from api.schemas import BookingRequest, PredictionResponse

app = FastAPI(title="Hotel Cancellation Model API", version="0.1.0")


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
def predict(booking: BookingRequest) -> PredictionResponse:
    """Predict the cancellation probability of one booking."""
    # TEMPORARY: a fixed value until we load the real model from MLflow (next step)
    probability = 0.5
    return PredictionResponse(
        booking_id=booking.booking_id,
        cancel_probability=probability,
        risk_level=risk_level(probability),
        model_version="placeholder",
    )
