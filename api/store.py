"""Saves every prediction to a database, so the monitoring job can compare live traffic with the
training data (drift) and join predictions with the true outcome by booking_id."""
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine

# Locally: a SQLite file next to the code. In docker-compose: the shared PostgreSQL, e.g.
# DATABASE_URL=postgresql+psycopg://user:password@postgres:5432/mlops
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predictions.db")

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("booking_id", String(64), nullable=False, index=True),
    Column("model_version", String(32), nullable=False),
    Column("cancel_probability", Float, nullable=False),
    Column("risk_level", String(8), nullable=False),
    Column("features", JSON, nullable=False),  # the booking exactly as the API received it
)


class PredictionStore:
    """Writes one row per prediction into the `predictions` table (created if it does not exist)."""

    def __init__(self, url: str = DATABASE_URL):
        self.engine = create_engine(url)
        metadata.create_all(self.engine)

    def save(self, booking_id: str, features: dict, probability: float, risk_level: str, version: str):
        row = {
            "created_at": datetime.now(timezone.utc),
            "booking_id": booking_id,
            "model_version": version,
            "cancel_probability": probability,
            "risk_level": risk_level,
            "features": features,
        }
        with self.engine.begin() as connection:  # begin() commits automatically at the end
            connection.execute(predictions.insert().values(**row))
