"""Model API: receives one hotel booking and returns the probability that it will be cancelled."""
import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.metrics import (
    CANCEL_PROBABILITY,
    LATENCY,
    PREDICTION_LOG_ERRORS,
    PREDICTIONS,
    REQUESTS,
    endpoint_label,
)
from api.model import LoadedModel, load_champion
from api.schemas import BookingRequest, PredictionResponse
from api.store import PredictionStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Hotel Cancellation Model API", version="0.5.0")


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


@lru_cache(maxsize=1)
def get_store() -> PredictionStore:
    """Connect to the prediction database once and reuse the connection pool afterwards."""
    return PredictionStore()


def risk_level(probability: float) -> str:
    """Turn a probability into a label the hotel staff can act on."""
    if probability >= 0.6:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def save_prediction(booking: BookingRequest, booking_id: str, probability: float, level: str, version: str):
    """Write the prediction to the database. If the database is down, log it and carry on:
    the hotel still gets its answer, and the failure shows up in the prediction_log_errors metric."""
    try:
        features = booking.model_dump(mode="json", exclude={"booking_id"})
        get_store().save(booking_id, features, probability, level, version)
    except Exception:
        PREDICTION_LOG_ERRORS.inc()
        logger.exception("Could not save prediction %s", booking_id)


def explain_prediction(model: LoadedModel, booking: BookingRequest) -> list[dict]:
    """Top factors for the frontend. A problem with the explanation never blocks the prediction."""
    try:
        return model.explain(booking)
    except Exception:
        logger.exception("Could not explain prediction")
        return []


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
    booking_id = booking.booking_id or str(uuid.uuid4())  # every prediction gets an id
    probability = round(model.predict_proba(booking), 4)
    level = risk_level(probability)
    PREDICTIONS.labels(level).inc()
    CANCEL_PROBABILITY.observe(probability)
    save_prediction(booking, booking_id, probability, level, model.version)
    return PredictionResponse(
        booking_id=booking_id,
        cancel_probability=probability,
        risk_level=level,
        model_version=model.version,
        top_factors=explain_prediction(model, booking),
    )
