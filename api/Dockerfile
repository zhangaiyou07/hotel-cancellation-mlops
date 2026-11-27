# Image of the model API. Build from the repository root (it needs both api/ and src/):
#   docker build -f api/Dockerfile -t hotel-api .
FROM python:3.11-slim

# XGBoost needs the OpenMP library (the Linux version of the libomp fix on macOS)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Packages first: Docker caches this step, so a code change does not reinstall everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src/ is needed because the saved pipeline uses src.features.FeatureEngineer
COPY src/ src/
COPY api/ api/

# Run as a normal user instead of root
RUN useradd --create-home appuser
USER appuser

# Defaults for running the container alone; docker-compose overrides them
# (MLFLOW_TRACKING_URI -> the MLflow server, DATABASE_URL -> the shared PostgreSQL)
ENV PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    DATABASE_URL=sqlite:////tmp/predictions.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
