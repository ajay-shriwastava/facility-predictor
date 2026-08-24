"""
Leakage-safe feature engineering.

Every feature for a given row is anchored to its booking_timestamp (T_inference).
Only records where booking_timestamp < T_inference are used — no exceptions.

Feature set (22 features):
    Rolling windows (10)  : facility preference, booking count, preferred hour,
                            preferred dow — each for 7d and 30d; facility preference
                            also for 60d and 90d.
    Drift (1)             : 7d vs 30d facility preference mismatch
    Lead time (2)         : average and std dev of historical lead times
    Community context (2) : total community bookings (7d), most popular facility (30d)
    Temporal/cyclical (7) : sin/cos encoding of hour, dow, month + is_weekend
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from facility_predictor.generator import FACILITIES

FAC_ENCODE: dict[str, int] = {f: i for i, f in enumerate(FACILITIES)}
MISSING_FAC = -1
TWO_PI = 2 * np.pi

FEATURE_COLS = [
    "res_facility_prefer_7d",
    "res_facility_prefer_30d",
    "res_facility_prefer_60d",
    "res_facility_prefer_90d",
    "res_booking_count_7d",
    "res_booking_count_30d",
    "res_preferred_hour_7d",
    "res_preferred_hour_30d",
    "res_preferred_dow_7d",
    "res_preferred_dow_30d",
    "res_facility_drift_7_30",
    "res_avg_lead_time_hours",
    "res_lead_time_std",
    "community_booking_count_7d",
    "community_most_popular_fac_30d",
    "booking_hour_sin",
    "booking_hour_cos",
    "booking_dow_sin",
    "booking_dow_cos",
    "booking_month_sin",
    "booking_month_cos",
    "is_weekend",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mode_fac_encoded(series: pd.Series) -> int:
    if series.empty:
        return MISSING_FAC
    return FAC_ENCODE.get(str(series.mode().iloc[0]), MISSING_FAC)


def _mode_float(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mode().iloc[0])


def _window(records: list[dict], t: pd.Timestamp, days: int) -> list[dict]:
    """Trailing `days` window from accumulated history (already bounded to < t)."""
    cutoff = t - pd.Timedelta(days=days)
    return [r for r in records if r["booking_timestamp"] >= cutoff]


# ── Per-row feature extraction ─────────────────────────────────────────────────

def _extract(row: pd.Series, res_hist: list[dict], comm_hist: list[dict]) -> dict:
    t = pd.Timestamp(row["booking_timestamp"])

    # Resident rolling windows
    h7  = _window(res_hist, t, 7)
    h30 = _window(res_hist, t, 30)
    h60 = _window(res_hist, t, 60)
    h90 = _window(res_hist, t, 90)

    fac_7  = _mode_fac_encoded(pd.Series([r["facility_id"] for r in h7]))
    fac_30 = _mode_fac_encoded(pd.Series([r["facility_id"] for r in h30]))
    fac_60 = _mode_fac_encoded(pd.Series([r["facility_id"] for r in h60]))
    fac_90 = _mode_fac_encoded(pd.Series([r["facility_id"] for r in h90]))

    bk_7  = len(h7)
    bk_30 = len(h30)

    hr_7  = _mode_float(pd.Series([r["usage_hour"] for r in h7]))
    hr_30 = _mode_float(pd.Series([r["usage_hour"] for r in h30]))

    dow_7  = _mode_float(pd.Series([r["usage_dow"] for r in h7]))
    dow_30 = _mode_float(pd.Series([r["usage_dow"] for r in h30]))

    drift = int(
        fac_7 != MISSING_FAC
        and fac_30 != MISSING_FAC
        and fac_7 != fac_30
    )

    if not res_hist:
        avg_lead = float("nan")
        std_lead = float("nan")
    else:
        leads = [r["lead_time_hours"] for r in res_hist]
        avg_lead = float(np.mean(leads))
        std_lead = float(np.std(leads, ddof=1)) if len(leads) > 1 else 0.0

    # Community context
    c7  = _window(comm_hist, t, 7)
    c30 = _window(comm_hist, t, 30)
    comm_count_7  = len(c7)
    comm_pop_30   = _mode_fac_encoded(pd.Series([r["facility_id"] for r in c30]))

    # Temporal / cyclical
    bk_hour  = t.hour
    bk_dow   = t.dayofweek
    bk_month = t.month

    return {
        "res_facility_prefer_7d":       fac_7,
        "res_facility_prefer_30d":      fac_30,
        "res_facility_prefer_60d":      fac_60,
        "res_facility_prefer_90d":      fac_90,
        "res_booking_count_7d":         bk_7,
        "res_booking_count_30d":        bk_30,
        "res_preferred_hour_7d":        hr_7,
        "res_preferred_hour_30d":       hr_30,
        "res_preferred_dow_7d":         dow_7,
        "res_preferred_dow_30d":        dow_30,
        "res_facility_drift_7_30":      drift,
        "res_avg_lead_time_hours":      avg_lead,
        "res_lead_time_std":            std_lead,
        "community_booking_count_7d":   comm_count_7,
        "community_most_popular_fac_30d": comm_pop_30,
        "booking_hour_sin":   np.sin(TWO_PI * bk_hour  / 24),
        "booking_hour_cos":   np.cos(TWO_PI * bk_hour  / 24),
        "booking_dow_sin":    np.sin(TWO_PI * bk_dow   / 7),
        "booking_dow_cos":    np.cos(TWO_PI * bk_dow   / 7),
        "booking_month_sin":  np.sin(TWO_PI * bk_month / 12),
        "booking_month_cos":  np.cos(TWO_PI * bk_month / 12),
        "is_weekend":         int(bk_dow >= 5),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute leakage-safe features for every row in df.

    Rows are processed in chronological order. For row i, only rows 0..i-1
    (strictly earlier booking_timestamps) are visible.

    Returns
    -------
    features : pd.DataFrame  shape (n, 22) — aligned with df after sort
    targets  : pd.DataFrame  columns: facility_id, usage_dow, usage_hour, lead_time_hours
    """
    df = df.copy()
    df["booking_timestamp"] = pd.to_datetime(df["booking_timestamp"])
    df["usage_timestamp"]   = pd.to_datetime(df["usage_timestamp"])

    df["usage_hour"]      = df["usage_timestamp"].dt.hour
    df["usage_dow"]       = df["usage_timestamp"].dt.dayofweek
    df["lead_time_hours"] = (
        (df["usage_timestamp"] - df["booking_timestamp"]).dt.total_seconds() / 3600
    )

    df = df.sort_values("booking_timestamp").reset_index(drop=True)

    comm_records: list[dict] = []
    res_records: dict[str, list[dict]] = defaultdict(list)
    feature_rows: list[dict] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        rid = row["resident_id"]

        feature_rows.append(_extract(row, res_records[rid], comm_records))

        rec = {
            "booking_timestamp": row["booking_timestamp"],
            "facility_id":       row["facility_id"],
            "usage_hour":        int(row["usage_hour"]),
            "usage_dow":         int(row["usage_dow"]),
            "lead_time_hours":   float(row["lead_time_hours"]),
        }
        comm_records.append(rec)
        res_records[rid].append(rec)

    features = pd.DataFrame(feature_rows, index=df.index)
    targets  = df[["facility_id", "usage_dow", "usage_hour", "lead_time_hours"]].copy()

    return features, targets