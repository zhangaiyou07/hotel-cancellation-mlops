"""Data validation and preprocessing tests."""
import pandas as pd

from src.features import LEAKY_COLUMNS, TARGET, FeatureEngineer, basic_clean

REQUIRED_COLUMNS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "adults", "children", "babies", "market_segment", "deposit_type", "adr", TARGET,
]


def test_required_columns_present(raw_df):
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    assert not missing, f"missing columns: {missing}"


def test_target_is_binary_and_not_missing(raw_df):
    assert raw_df[TARGET].notna().all()
    assert set(raw_df[TARGET].unique()) <= {0, 1}


def test_basic_clean_removes_invalid_rows(raw_df):
    clean = basic_clean(raw_df)
    guests = clean["adults"] + clean["children"].fillna(0) + clean["babies"]
    assert (guests > 0).all(), "bookings with zero guests should be dropped"
    assert clean["adr"].between(0, 1000, inclusive="left").all(), "absurd prices should be dropped"
    assert len(clean) < len(raw_df)


def test_feature_engineer_known_input(sample_df):
    """A hand-made booking produces the expected engineered features."""
    row = sample_df.drop(columns=[TARGET]).head(1).copy()
    row["stays_in_weekend_nights"] = 2
    row["stays_in_week_nights"] = 3
    row["adults"], row["children"], row["babies"] = 2, 1, 0
    row["adr"] = 120.0
    row["arrival_date"] = pd.Timestamp("2017-06-10")
    row["lead_time"] = 10

    out = FeatureEngineer().transform(row).iloc[0]
    assert out["total_nights"] == 5
    assert out["total_guests"] == 3
    assert out["has_kids"] == 1
    assert out["adr_per_guest"] == 40.0
    assert out["total_revenue"] == 600.0
    assert out["booking_month"] == 5  # booked 10 days before 10 June -> May


def test_no_leaky_columns_after_feature_engineering(sample_df):
    out = FeatureEngineer().transform(sample_df)
    leaked = [c for c in LEAKY_COLUMNS + [TARGET] if c in out.columns]
    assert not leaked, f"leaky columns reached the model: {leaked}"
