# Monitoring plan (feature/monitoring)

- **Traffic simulator:** a Python script (`monitoring/simulate_traffic.py`) that sends unseen test-set rows to the model API's `/predict`; a `--drift` flag shifts key features (e.g. longer `lead_time`, different `market_segment` mix) to inject drift.
- **Evidently:** a job that reads logged predictions from Postgres, compares them with the training reference data, and computes data drift and model performance metrics.
- **Output:** metrics are written to a Postgres table (`drift_metrics`) that Grafana reads; drift above a threshold raises an alert / retraining trigger.
- **Tools:** Python, requests (or Locust), Evidently, PostgreSQL, pytest.
- **Tests:** injected drift is flagged; normal traffic is not.
