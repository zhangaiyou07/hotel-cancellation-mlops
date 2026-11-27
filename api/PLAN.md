# Model API plan (feature/model-api)

- **What:** A FastAPI service that loads `hotel-cancellation-model@champion` from the MLflow Model Registry at startup and serves predictions. Runs as its own container in docker-compose.
- **Endpoints:** `POST /predict` (booking details in JSON -> `cancel_probability`, `risk_level`, `model_version`), `GET /health`, and `GET /metrics` for Prometheus (request count, latency, errors).
- **Input validation:** a Pydantic schema with types and valid ranges (e.g. adults >= 0, lead_time >= 0, known deposit types), so bad input returns a clear 422 error instead of crashing.
- **Logging:** every request and prediction is written to PostgreSQL (timestamp, input features, probability, model version) so the monitoring team can compute drift with Evidently.
- **Tools / tests:** FastAPI, Pydantic, MLflow, prometheus-client, SQLAlchemy/psycopg, Docker. Unit tests for validation and response format, plus an integration test that starts the API container and sends a real request.