"""Tests for leakage-safe feature engineering."""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from facility_predictor.features import (
    FEATURE_COLS,
    MISSING_FAC,
    build_feature_matrix,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal booking DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


def _booking(resident_id, facility, booking_dt, usage_dt):
    return {
        "booking_id":        f"BK-{resident_id}",
        "resident_id":       resident_id,
        "facility_id":       facility,
        "booking_timestamp": booking_dt,
        "usage_timestamp":   usage_dt,
    }


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def single_resident_df():
    """5 bookings for one resident, all at the Gym, spaced 3 days apart."""
    base = datetime(2025, 2, 1, 8, 0)
    rows = []
    for i in range(5):
        bt = base + timedelta(days=i * 3)
        ut = bt + timedelta(hours=1)
        rows.append(_booking("R-001", "Gym", bt, ut))
    return _make_df(rows)


# ── Shape & columns ────────────────────────────────────────────────────────────

def test_output_shape(single_resident_df):
    features, targets = build_feature_matrix(single_resident_df)
    assert features.shape == (5, 22)
    assert targets.shape  == (5, 4)


def test_feature_columns(single_resident_df):
    features, _ = build_feature_matrix(single_resident_df)
    assert list(features.columns) == FEATURE_COLS


def test_target_columns(single_resident_df):
    _, targets = build_feature_matrix(single_resident_df)
    assert set(targets.columns) == {"facility_id", "usage_dow", "usage_hour", "lead_time_hours"}


# ── Cold-start (first booking has no history) ──────────────────────────────────

def test_cold_start_facility_is_missing(single_resident_df):
    features, _ = build_feature_matrix(single_resident_df)
    first = features.iloc[0]
    assert first["res_facility_prefer_7d"]  == MISSING_FAC
    assert first["res_facility_prefer_30d"] == MISSING_FAC
    assert first["res_booking_count_7d"]    == 0
    assert first["res_booking_count_30d"]   == 0


def test_cold_start_hour_dow_are_nan(single_resident_df):
    features, _ = build_feature_matrix(single_resident_df)
    first = features.iloc[0]
    assert math.isnan(first["res_preferred_hour_7d"])
    assert math.isnan(first["res_preferred_dow_7d"])
    assert math.isnan(first["res_avg_lead_time_hours"])


# ── Leakage: features at row i must not see row i's targets ───────────────────

def test_no_leakage(single_resident_df):
    """Booking count at row 0 must be 0; at row k must equal k."""
    features, _ = build_feature_matrix(single_resident_df)
    # All 5 bookings are within 30 days of each other
    for k, count in enumerate(features["res_booking_count_30d"]):
        assert count == k, f"Row {k}: expected {k} prior bookings, got {count}"


# ── Drift flag ─────────────────────────────────────────────────────────────────

def test_drift_flag_set_when_facilities_differ():
    """3 Gym bookings spread over a month, then 2 Pool bookings in the last week.
    At the final row: 7d mode=Pool, 30d mode=Gym → drift=1."""
    base = datetime(2025, 3, 1, 9, 0)
    rows = [
        _booking("R-001", "Gym",          base,                       base + timedelta(hours=1)),
        _booking("R-001", "Gym",          base + timedelta(days=5),   base + timedelta(days=5,  hours=1)),
        _booking("R-001", "Gym",          base + timedelta(days=10),  base + timedelta(days=10, hours=1)),
        _booking("R-001", "Swimming Pool", base + timedelta(days=25), base + timedelta(days=25, hours=1)),
        _booking("R-001", "Swimming Pool", base + timedelta(days=27), base + timedelta(days=27, hours=1)),
        # current row at day 30: sees 7d=Pool, 30d=Gym
        _booking("R-001", "Gym",          base + timedelta(days=30),  base + timedelta(days=30, hours=1)),
    ]
    features, _ = build_feature_matrix(_make_df(rows))
    assert features.iloc[-1]["res_facility_drift_7_30"] == 1


def test_drift_flag_clear_when_same():
    """All bookings at Gym — drift must always be 0."""
    base = datetime(2025, 3, 1, 9, 0)
    rows = [
        _booking("R-001", "Gym", base + timedelta(days=i * 3), base + timedelta(days=i * 3, hours=1))
        for i in range(6)
    ]
    features, _ = build_feature_matrix(_make_df(rows))
    assert (features["res_facility_drift_7_30"] == 0).all()


# ── Window boundary ────────────────────────────────────────────────────────────

def test_7d_window_excludes_older_records():
    """A booking older than 7 days must not count in the 7d window."""
    base = datetime(2025, 4, 1, 10, 0)
    rows = [
        _booking("R-001", "Gym",         base,                      base + timedelta(hours=1)),
        _booking("R-001", "Tennis Court", base + timedelta(days=10), base + timedelta(days=10, hours=1)),
    ]
    features, _ = build_feature_matrix(_make_df(rows))
    # Row 1 (Tennis Court): the Gym booking is 10 days ago, outside 7d window
    assert features.iloc[1]["res_booking_count_7d"] == 0
    assert features.iloc[1]["res_booking_count_30d"] == 1