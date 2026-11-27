"""Prometheus metrics of the model API, exposed at GET /metrics for Prometheus to collect."""
from prometheus_client import Counter, Histogram

# Request count by method, endpoint and HTTP status (200 ok, 422 bad input, 503 model unavailable, ...)
REQUESTS = Counter(
    "api_requests",
    "HTTP requests handled by the model API",
    ["method", "endpoint", "status"],
)

# How long each request takes (Grafana shows e.g. the 95th percentile latency)
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Time spent handling a request",
    ["method", "endpoint"],
)

# What the model predicts: a shift here is a first sign of drift (prediction drift)
PREDICTIONS = Counter(
    "predictions",
    "Predictions returned, by risk level",
    ["risk_level"],
)
CANCEL_PROBABILITY = Histogram(
    "prediction_cancel_probability",
    "Distribution of predicted cancellation probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Only these paths get their own label; anything else is grouped as "other"
# (so random URLs cannot create thousands of different time series)
KNOWN_ENDPOINTS = {"/predict", "/health"}


def endpoint_label(path: str) -> str:
    return path if path in KNOWN_ENDPOINTS else "other"
