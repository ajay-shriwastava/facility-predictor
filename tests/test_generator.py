"""Smoke tests for the synthetic data generator."""

from facility_predictor.generator import FACILITIES, generate


def test_generate_row_count():
    df = generate()
    assert len(df) >= 10_000, f"Expected at least 10,000 bookings, got {len(df)}"


def test_generate_schema():
    df = generate()
    expected_cols = {"booking_id", "resident_id", "facility_id", "booking_timestamp", "usage_timestamp"}
    assert expected_cols == set(df.columns)


def test_generate_resident_count():
    df = generate()
    assert df["resident_id"].nunique() == 200


def test_generate_facilities():
    df = generate()
    assert set(df["facility_id"].unique()).issubset(set(FACILITIES))


def test_booking_before_usage():
    df = generate()
    import pandas as pd
    df["booking_timestamp"] = pd.to_datetime(df["booking_timestamp"])
    df["usage_timestamp"]   = pd.to_datetime(df["usage_timestamp"])
    assert (df["booking_timestamp"] < df["usage_timestamp"]).all(), \
        "Every booking_timestamp must be before its usage_timestamp"


def test_all_in_2025():
    df = generate()
    import pandas as pd
    df["booking_timestamp"] = pd.to_datetime(df["booking_timestamp"])
    df["usage_timestamp"]   = pd.to_datetime(df["usage_timestamp"])
    assert (df["booking_timestamp"].dt.year == 2025).all()
    assert (df["usage_timestamp"].dt.year   == 2025).all()