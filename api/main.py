"""Model API: receives one hotel booking and returns the probability that it will be cancelled."""
import time
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.metrics import CANCEL_PROBABILITY, LATENCY, PREDICTIONS, REQUESTS, endpoint_label
from api.model import LoadedModel, load_champion
from api.schemas import BookingRequest, PredictionResponse

app = FastAPI(title="Hotel Cancellation Model API", version="0.3.0")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Count every request and time it, whatever the outcome (success, bad input or error)."""
    if request.url.path == "/metrics":
        return await call_next(request)  # do not count Prometheus' own visits
    endpoint = endpoint_label(request.url.path)
    start = time.perf_counter()
    status = 500  # if the endpoint crashes, the request is counted as a server error
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(request.method, endpoint, str(status)).inc()


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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Current values of all metrics, in the text format Prometheus reads."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    booking: BookingRequest,
    model: Annotated[LoadedModel, Depends(get_model)],
) -> PredictionResponse:
    """Predict the cancellation probability of one booking with the champion model."""
    probability = model.predict_proba(booking)
    level = risk_level(probability)
    PREDICTIONS.labels(level).inc()
    CANCEL_PROBABILITY.observe(probability)
    return PredictionResponse(
        booking_id=booking.booking_id,
        cancel_probability=round(probability, 4),
        risk_level=level,
        model_version=model.version,
    )
