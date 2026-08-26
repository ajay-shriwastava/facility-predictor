"""Business-critical sanity checks for evaluation match logic and reporting."""

from __future__ import annotations

import pandas as pd
import pytest

from facility_predictor.evaluation import (
    HOUR_TOLERANCE,
    NUDGE_TOLERANCE,
    compute_summary,
    error_analysis,
    load_review,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_review():
    """
    Three rows with known match outcomes:
        row 0 — all 4 match               → score 4
        row 1 — fac miss, dow miss         → score 2  (hour + nudge match)
        row 2 — hour miss, nudge miss      → score 2  (fac + dow match)
    """
    return pd.DataFrame({
        "pred_facility":          ["Gym",    "Pool",      "Gym"],
        "pred_day":               ["Monday", "Tuesday",   "Wednesday"],
        "pred_hour":              ["08:00",  "10:00",     "14:00"],
        "pred_lead_time_hours":   [2.0,      5.0,         1.0],
        "actual_facility":        ["Gym",    "Gym",       "Gym"],
        "actual_day":             ["Monday", "Wednesday", "Wednesday"],
        "actual_hour":            ["08:00",  "10:00",     "17:00"],
        "actual_lead_time_hours": [2.5,      5.0,         4.0],
    })


@pytest.fixture
def loaded_df(tmp_path, raw_review):
    p = tmp_path / "review.csv"
    raw_review.to_csv(p, index=False)
    return load_review(p)


def _single_row(pred_hour: str, actual_hour: str,
                pred_lead: float, actual_lead: float) -> pd.DataFrame:
    return pd.DataFrame({
        "pred_facility":          ["Gym"],
        "pred_day":               ["Monday"],
        "pred_hour":              [pred_hour],
        "pred_lead_time_hours":   [pred_lead],
        "actual_facility":        ["Gym"],
        "actual_day":             ["Monday"],
        "actual_hour":            [actual_hour],
        "actual_lead_time_hours": [actual_lead],
    })


# ── Operational guard ──────────────────────────────────────────────────────────

def test_load_review_raises_if_file_missing(tmp_path):
    """System must fail loudly when pipeline.evaluate() has not been run."""
    with pytest.raises(FileNotFoundError):
        load_review(tmp_path / "nonexistent.csv")


# ── Core match logic ───────────────────────────────────────────────────────────

def test_facility_match_logic(loaded_df):
    """Facility is the root of the cascade — wrong match logic corrupts all reporting."""
    assert loaded_df.loc[0, "fac_match"]      # Gym == Gym
    assert not loaded_df.loc[1, "fac_match"]  # Pool != Gym


def test_score_is_sum_of_match_columns(loaded_df):
    """Score is the headline KPI; must equal the count of individual matches."""
    assert loaded_df.loc[0, "score"] == 4  # all match
    assert loaded_df.loc[1, "score"] == 2  # hour + nudge match only
    assert loaded_df.loc[2, "score"] == 2  # fac + dow match only


# ── Hour tolerance boundary ────────────────────────────────────────────────────

def test_hour_match_at_exact_tolerance_is_accepted(tmp_path):
    """A prediction exactly at the hour tolerance must count as correct (business SLA)."""
    df = _single_row(f"{8:02d}:00", f"{8 + HOUR_TOLERANCE:02d}:00", 2.0, 2.0)
    df.to_csv(tmp_path / "r.csv", index=False)
    assert load_review(tmp_path / "r.csv").loc[0, "hour_match"]


def test_hour_match_one_over_tolerance_is_rejected(tmp_path):
    """A prediction one hour past the tolerance must not be counted as a hit."""
    df = _single_row(f"{8:02d}:00", f"{8 + HOUR_TOLERANCE + 1:02d}:00", 2.0, 2.0)
    df.to_csv(tmp_path / "r.csv", index=False)
    assert not load_review(tmp_path / "r.csv").loc[0, "hour_match"]


# ── Nudge tolerance boundary ───────────────────────────────────────────────────

def test_nudge_match_at_exact_tolerance_is_accepted(tmp_path):
    """A notification prediction exactly at the lead-time tolerance must count as correct."""
    df = _single_row("08:00", "08:00", 2.0, 2.0 + NUDGE_TOLERANCE)
    df.to_csv(tmp_path / "r.csv", index=False)
    assert load_review(tmp_path / "r.csv").loc[0, "nudge_match"]


def test_nudge_match_one_over_tolerance_is_rejected(tmp_path):
    """A notification prediction beyond the lead-time tolerance must not count as a hit."""
    df = _single_row("08:00", "08:00", 2.0, 2.0 + NUDGE_TOLERANCE + 0.1)
    df.to_csv(tmp_path / "r.csv", index=False)
    assert not load_review(tmp_path / "r.csv").loc[0, "nudge_match"]


# ── Business reporting ─────────────────────────────────────────────────────────

def test_summary_full_partial_no_match_counts(loaded_df):
    """Full / partial / no-match breakdown is the business report card."""
    summary = compute_summary(loaded_df)
    assert summary["total_predictions"] == 3
    assert summary["full_matches"]      == 1  # row 0
    assert summary["partial_matches"]   == 2  # rows 1, 2
    assert summary["no_matches"]        == 0


def test_summary_exact_match_rate(loaded_df):
    """Exact match rate is the primary headline KPI."""
    summary = compute_summary(loaded_df)
    assert abs(summary["exact_match_rate"] - 1 / 3) < 1e-9


# ── Error diagnosis ────────────────────────────────────────────────────────────

def test_facility_mismatches_pivot_on_actual_facility(loaded_df):
    """Error analysis must identify which facilities are predicted incorrectly — actionable insight."""
    errors = error_analysis(loaded_df)
    # row 1: pred=Pool, actual=Gym → mismatch counted under "Gym"
    assert "Gym" in errors["facility"].index