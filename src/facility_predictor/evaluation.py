"""
Evaluation metrics and summary reporting.

Reads prediction_review.csv produced by pipeline.evaluate()
and computes per-output metrics plus an overall composite score.

Threshold constants here are the single source of truth for match tolerances:
    HOUR_TOLERANCE  — max absolute hour difference to count as a hit (default ±1)
    NUDGE_TOLERANCE — max absolute lead-time difference in hours (default ±2)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from facility_predictor.pipeline import MODELS_DIR, REVIEW_PATH

HOUR_TOLERANCE  = 1   # hours, for usage-hour match
NUDGE_TOLERANCE = 2   # hours, for notification lead-time match


def load_review(path: Path = REVIEW_PATH) -> pd.DataFrame:
    """
    Load the review CSV and compute match columns + score from raw predictions.

    Match logic (owned here, not in pipeline):
        fac_match   — exact string match on facility name
        dow_match   — exact string match on day name
        hour_match  — |pred_hour - actual_hour| <= HOUR_TOLERANCE
        nudge_match — |pred_lead_time_hours - actual_lead_time_hours| <= NUDGE_TOLERANCE
        score       — integer sum of the four boolean match columns (0–4)
    """
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}. Run pipeline.evaluate() first.")
    df = pd.read_csv(path)

    df["fac_match"]   = df["pred_facility"] == df["actual_facility"]
    df["dow_match"]   = df["pred_day"] == df["actual_day"]
    df["hour_match"]  = (
        df["pred_hour"].str[:2].astype(int)
        .sub(df["actual_hour"].str[:2].astype(int))
        .abs()
        <= HOUR_TOLERANCE
    )
    df["nudge_match"] = (
        (df["pred_lead_time_hours"] - df["actual_lead_time_hours"]).abs()
        <= NUDGE_TOLERANCE
    )
    df["score"] = (
        df["fac_match"].astype(int)
        + df["dow_match"].astype(int)
        + df["hour_match"].astype(int)
        + df["nudge_match"].astype(int)
    )
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute per-output match rates and exact match rate from the review DataFrame.

    Returns
    -------
    dict with keys:
        facility_match_rate, day_match_rate, hour_match_rate, nudge_match_rate,
        exact_match_rate, total_predictions, full_matches, partial_matches, no_matches
    """
    n = len(df)
    return {
        "facility_match_rate": float(df["fac_match"].mean()),
        "day_match_rate":      float(df["dow_match"].mean()),
        "hour_match_rate":     float(df["hour_match"].mean()),
        "nudge_match_rate":    float(df["nudge_match"].mean()),
        "exact_match_rate":    float((df["score"] == 4).mean()),
        "total_predictions":   n,
        "full_matches":        int((df["score"] == 4).sum()),
        "partial_matches":     int(((df["score"] > 0) & (df["score"] < 4)).sum()),
        "no_matches":          int((df["score"] == 0).sum()),
    }


def error_analysis(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Break down mismatches by a meaningful pivot column per output.

    Returns
    -------
    dict mapping output name → Series of mismatch counts:
        facility → by actual_facility
        day      → by actual_day
        hour     → by actual_hour
        nudge    → by actual_day
    """
    def _mismatches(match_col: str, pivot_col: str) -> pd.Series:
        miss = df[~df[match_col]]
        return miss[pivot_col].value_counts() if not miss.empty else pd.Series(dtype=int)

    return {
        "facility": _mismatches("fac_match",   "actual_facility"),
        "day":      _mismatches("dow_match",    "actual_day"),
        "hour":     _mismatches("hour_match",   "actual_hour"),
        "nudge":    _mismatches("nudge_match",  "actual_day"),
    }


def get_feature_importance(model_name: str, models_dir: Path = MODELS_DIR) -> list[tuple[str, float]]:
    """
    Return top-10 (feature, importance) pairs for a given model.
    model_name: 'facility' | 'day' | 'hour' | 'notification'
    """
    path = models_dir / f"{model_name}_model.json"
    if model_name in ("facility", "day"):
        m = xgb.XGBClassifier()
    else:
        m = xgb.XGBRegressor()
    m.load_model(path)

    names  = m.get_booster().feature_names
    scores = m.feature_importances_
    pairs  = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return pairs[:10]


def print_report(path: Path = REVIEW_PATH) -> None:
    df      = load_review(path)
    summary = compute_summary(df)
    errors  = error_analysis(df)

    print("\n" + "=" * 55)
    print("  PREDICTION REVIEW REPORT")
    print("=" * 55)
    print(f"  Total predictions : {summary['total_predictions']:,}")
    print(f"  Full matches (4/4): {summary['full_matches']:,}  ({summary['exact_match_rate']:.1%})")
    print(f"  Partial matches   : {summary['partial_matches']:,}")
    print(f"  No matches (0/4)  : {summary['no_matches']:,}")
    print("-" * 55)
    print(f"  Facility match          : {summary['facility_match_rate']:.1%}")
    print(f"  Day match               : {summary['day_match_rate']:.1%}")
    print(f"  Hour match  (±{HOUR_TOLERANCE}hr)    : {summary['hour_match_rate']:.1%}")
    print(f"  Nudge match (±{NUDGE_TOLERANCE}hr)    : {summary['nudge_match_rate']:.1%}")
    print("-" * 55)
    print("  Mismatches by actual value:")
    for output, pivot_col, pivot_label in [
        ("facility", "actual_facility", "facility"),
        ("day",      "actual_day",      "day"),
        ("hour",     "actual_hour",     "hour"),
        ("nudge",    "actual_day",      "day (nudge)"),
    ]:
        series = errors.get(output, pd.Series(dtype=int))
        if not series.empty:
            print(f"    [{output} — by {pivot_label}]")
            for val, cnt in series.items():
                print(f"      {str(val):<22} {cnt}")
    print("=" * 55 + "\n")