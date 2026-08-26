"""Business-critical sanity checks for the agent explanation layer."""

from __future__ import annotations

import facility_predictor.agent as agent_module
from facility_predictor.agent import _format_features, explain_prediction


def test_format_features_known_label():
    """Property managers must see readable text, not raw column names."""
    result = _format_features([("res_facility_prefer_7d", 0.25)])
    assert "7-day facility preference" in result
    assert "0.250" in result


def test_format_features_unknown_label_falls_back_to_raw_name():
    """If features.py adds a new column not yet in FEATURE_LABELS, it must not silently disappear."""
    result = _format_features([("some_unlabelled_feature", 0.1)])
    assert "some_unlabelled_feature" in result


def test_explain_prediction_returns_none_without_api_key(monkeypatch):
    """Graceful degradation: Streamlit must not crash when ANTHROPIC_API_KEY is absent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(agent_module, "_chain", None)  # reset singleton so key check runs fresh

    result = explain_prediction(
        resident_id="R-001",
        recent_bookings="No history",
        prediction={"facility": "Gym", "day": "Monday", "hour": "08:00", "nudge": "Sunday 06:00"},
        top_features=[("res_facility_prefer_7d", 0.25)],
    )
    assert result is None