import pytest

from src.features import basic_clean, load_raw


@pytest.fixture(scope="session")
def raw_df():
    """The full raw dataset (loaded once per test session)."""
    return load_raw()


@pytest.fixture(scope="session")
def sample_df(raw_df):
    """A small, cleaned random sample so model tests run in seconds."""
    return basic_clean(raw_df).sample(4000, random_state=42).reset_index(drop=True)
