import pytest
from sklearn.linear_model import LogisticRegression

import api.main as main
from api.main import app, get_model
from api.model import LoadedModel
from api.store import PredictionStore
from src.features import TARGET, basic_clean, load_raw, make_pipeline

# One valid booking that all API tests can reuse
VALID_BOOKING = {
    "booking_id": "test-001",
    "hotel": "City Hotel",
    "arrival_date": "2017-06-10",
    "lead_time": 120,
    "adr": 95.5,
    "deposit_type": "No Deposit",
    "market_segment": "Online TA",
}


@pytest.fixture(scope="session")
def raw_df():
    """The full raw dataset (loaded once per test session)."""
    return load_raw()


@pytest.fixture(scope="session")
def sample_df(raw_df):
    """A small, cleaned random sample so model tests run in seconds."""
    return basic_clean(raw_df).sample(4000, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def small_model(sample_df):
    """A small model trained on a data sample, used instead of the MLflow champion in API tests
    (CI has no MLflow server). It goes through exactly the same preprocessing pipeline."""
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    pipeline = make_pipeline(LogisticRegression(max_iter=2000), kind="onehot").fit(X, y)
    model = LoadedModel(pipeline=pipeline, version="test")
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def prediction_store(tmp_path, monkeypatch):
    """Every test writes predictions to its own throw-away SQLite file instead of the real database."""
    store = PredictionStore(f"sqlite:///{tmp_path / 'predictions.db'}")
    monkeypatch.setattr(main, "get_store", lambda: store)
    return store
