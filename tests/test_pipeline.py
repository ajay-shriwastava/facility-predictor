"""Sanity checks for pipeline constants, cascade helpers, and data loading."""

import numpy as np
import pandas as pd
import pytest

from facility_predictor.generator import FACILITIES
from facility_predictor.pipeline import (
    DAY_NAMES,
    FEATURES_CACHE,
    FAC_TO_IDX,
    IDX_TO_FAC,
    TARGETS_CACHE,
    TRAIN_CUTOFF,
    _add_dow,
    _add_facility,
    _add_hour,
    _load_and_prepare,
)


# ── Constants ──────────────────────────────────────────────────────────────────

def test_fac_to_idx_covers_all_facilities():
    assert set(FAC_TO_IDX.keys()) == set(FACILITIES)


def test_idx_to_fac_is_inverse_of_fac_to_idx():
    for fac, idx in FAC_TO_IDX.items():
        assert IDX_TO_FAC[idx] == fac


def test_day_names_has_seven_entries():
    assert len(DAY_NAMES) == 7


def test_train_cutoff_is_october():
    assert TRAIN_CUTOFF == pd.Timestamp("2025-10-01")


# ── Cascade column adders ──────────────────────────────────────────────────────

@pytest.fixture
def dummy_X():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_add_facility_appends_column(dummy_X):
    vals = np.array([0, 1, 2])
    out = _add_facility(dummy_X, vals)
    assert "predicted_facility" in out.columns
    np.testing.assert_array_equal(out["predicted_facility"].values, vals)


def test_add_dow_appends_column(dummy_X):
    vals = np.array([3, 4, 5])
    out = _add_dow(dummy_X, vals)
    assert "predicted_dow" in out.columns
    np.testing.assert_array_equal(out["predicted_dow"].values, vals)


def test_add_hour_appends_column(dummy_X):
    vals = np.array([8.0, 9.0, 10.0])
    out = _add_hour(dummy_X, vals)
    assert "predicted_hour" in out.columns
    np.testing.assert_array_equal(out["predicted_hour"].values, vals)


def test_cascade_adders_do_not_mutate_input(dummy_X):
    original_cols = list(dummy_X.columns)
    _add_facility(dummy_X, np.array([0, 1, 2]))
    _add_dow(dummy_X, np.array([0, 1, 2]))
    _add_hour(dummy_X, np.array([0.0, 1.0, 2.0]))
    assert list(dummy_X.columns) == original_cols


# ── Data loading (skipped if CSV not present) ──────────────────────────────────

@pytest.mark.skipif(
    not (pd.io.common.file_exists("data/synthetic_bookings.csv")),  # type: ignore[attr-defined]
    reason="synthetic_bookings.csv not generated yet",
)
def test_load_and_prepare_shapes():
    X_train, X_test, y_train, y_test = _load_and_prepare()

    assert len(X_train) == len(y_train)
    assert len(X_test)  == len(y_test)
    assert len(X_train) > len(X_test), "train set should be larger than test set"

    assert set(y_train.columns) == {"facility_id", "usage_dow", "usage_hour", "lead_time_hours"}
    assert set(y_test.columns)  == {"facility_id", "usage_dow", "usage_hour", "lead_time_hours"}


@pytest.mark.skipif(
    not (pd.io.common.file_exists("data/synthetic_bookings.csv")),  # type: ignore[attr-defined]
    reason="synthetic_bookings.csv not generated yet",
)
def test_load_and_prepare_writes_cache():
    _load_and_prepare()
    assert FEATURES_CACHE.exists()
    assert TARGETS_CACHE.exists()