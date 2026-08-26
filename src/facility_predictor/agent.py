"""
LangChain + Claude explanation layer.

Takes a prediction and the XGBoost feature importances that drove it,
and generates a concise natural language explanation for the property manager.

Graceful degradation: returns None silently if ANTHROPIC_API_KEY is not set.
"""

from __future__ import annotations

import os

_MODEL      = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 300

FEATURE_LABELS: dict[str, str] = {
    "res_facility_prefer_7d":        "7-day facility preference",
    "res_facility_prefer_30d":       "30-day facility preference",
    "res_facility_prefer_60d":       "60-day facility preference",
    "res_facility_prefer_90d":       "90-day facility preference",
    "res_booking_count_7d":          "bookings in last 7 days",
    "res_booking_count_30d":         "bookings in last 30 days",
    "res_preferred_hour_7d":         "preferred usage hour (7d)",
    "res_preferred_hour_30d":        "preferred usage hour (30d)",
    "res_preferred_dow_7d":          "preferred day of week (7d)",
    "res_preferred_dow_30d":         "preferred day of week (30d)",
    "res_facility_drift_7_30":       "recent facility preference drift",
    "res_avg_lead_time_hours":       "average booking lead time (hours)",
    "res_lead_time_std":             "lead time consistency",
    "community_booking_count_7d":    "community activity level (7d)",
    "community_most_popular_fac_30d":"most popular community facility (30d)",
    "booking_hour_sin":              "booking time of day (cyclical)",
    "booking_hour_cos":              "booking time of day (cyclical)",
    "booking_dow_sin":               "booking day of week (cyclical)",
    "booking_dow_cos":               "booking day of week (cyclical)",
    "booking_month_sin":             "booking month / season (cyclical)",
    "booking_month_cos":             "booking month / season (cyclical)",
    "is_weekend":                    "weekend booking flag",
    "predicted_facility":            "predicted facility (cascade)",
    "predicted_dow":                 "predicted day (cascade)",
    "predicted_hour":                "predicted hour (cascade)",
}


_chain = None  # lazy singleton — built on first successful call


def _format_features(top_features: list[tuple[str, float]]) -> str:
    lines = []
    for name, score in top_features:
        label = FEATURE_LABELS.get(name, name)
        lines.append(f"- {label}: importance {score:.3f}")
    return "\n".join(lines)


def _get_chain():
    """Build (once) and return the LangChain chain, or None if unavailable."""
    global _chain
    if _chain is not None:
        return _chain
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        return None

    llm = ChatAnthropic(model=_MODEL, max_tokens=_MAX_TOKENS)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an assistant explaining ML facility booking predictions "
                "to a residential property manager. Be concise, factual, and "
                "grounded only in the data provided. Do not speculate."
            ),
        ),
        (
            "human",
            (
                "Resident {resident_id} recent booking history:\n{recent_bookings}\n\n"
                "Model prediction:\n"
                "  Facility : {facility}\n"
                "  Day      : {day}\n"
                "  Time     : {hour}\n"
                "  Nudge at : {nudge}\n\n"
                "Top signals driving this prediction:\n{top_features}\n\n"
                "In 2-3 sentences, explain why the model made this prediction "
                "based only on the signals above."
            ),
        ),
    ])
    _chain = prompt | llm | StrOutputParser()
    return _chain


def explain_prediction(
    resident_id: str,
    recent_bookings: str,
    prediction: dict[str, str],
    top_features: list[tuple[str, float]],
) -> str | None:
    """
    Generate a natural language explanation for a prediction using Claude via LangChain.

    Parameters
    ----------
    resident_id     : e.g. "R-104"
    recent_bookings : formatted string of last 5 bookings
    prediction      : dict with keys facility, day, hour, nudge
    top_features    : list of (feature_name, importance_score) tuples — from
                      evaluation.get_feature_importance(model_name)

    Returns
    -------
    Explanation string, or None if ANTHROPIC_API_KEY is not set or the API call fails.
    """
    chain = _get_chain()
    if chain is None:
        return None

    try:
        return chain.invoke({
            "resident_id":     resident_id,
            "recent_bookings": recent_bookings,
            "facility":        prediction.get("facility", ""),
            "day":             prediction.get("day", ""),
            "hour":            prediction.get("hour", ""),
            "nudge":           prediction.get("nudge", ""),
            "top_features":    _format_features(top_features),
        })
    except Exception:
        return None