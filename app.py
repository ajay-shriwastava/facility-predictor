"""
Streamlit UI — Facility Usage Prediction System.

Three tabs:
  1. Summary Dashboard  — per-output metrics and feature importance
  2. Prediction Review  — full test-set table with match indicators + CSV download
  3. Resident Explorer  — per-resident deep dive with optional LLM explanation
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Facility Usage Prediction System",
    layout="wide",
)

from facility_predictor.pipeline import MODELS_DIR, REVIEW_PATH

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("Facility Predictor")
st.sidebar.markdown("---")

if st.sidebar.button("Generate Data + Train Models", use_container_width=True):
    from facility_predictor.generator import generate
    from facility_predictor.pipeline import train

    with st.spinner("Generating synthetic dataset..."):
        generate("data/synthetic_bookings.csv")
    st.sidebar.success("Dataset generated.")

    with st.spinner("Training cascade models (this takes a few minutes)..."):
        metrics = train()
    st.sidebar.success("Training complete.")
    st.rerun()

if st.sidebar.button("Run Evaluation", use_container_width=True):
    from facility_predictor.pipeline import evaluate

    with st.spinner("Evaluating on test set..."):
        evaluate()
    st.sidebar.success("Evaluation complete.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Train window : Jan–Sep 2025\n\nTest window : Oct–Dec 2025"
)

# ── Require evaluation output ──────────────────────────────────────────────────

st.title("Facility Usage Prediction System")

if not REVIEW_PATH.exists():
    st.info(
        "No evaluation results found.\n\n"
        "Click **Generate Data + Train Models** then **Run Evaluation** in the sidebar."
    )
    st.stop()

@st.cache_data
def load_review() -> pd.DataFrame:
    from facility_predictor.evaluation import load_review as _load_review
    return _load_review()


review = load_review()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Summary Dashboard", "Prediction Review", "Resident Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Summary Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    n = len(review)

    fac_rate   = review["fac_match"].mean()
    dow_rate   = review["dow_match"].mean()
    hour_rate  = review["hour_match"].mean()
    nudge_rate = review["nudge_match"].mean()
    exact_rate = (review["score"] == 4).mean()

    st.subheader("Test Set Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Facility Match",      f"{fac_rate:.1%}")
    c2.metric("Day Match",           f"{dow_rate:.1%}")
    c3.metric("Hour Match (±1hr)",   f"{hour_rate:.1%}")
    c4.metric("Nudge Match (±2hrs)", f"{nudge_rate:.1%}")
    c5.metric("Exact Match (4/4)",   f"{exact_rate:.1%}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Match Rate by Output")
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Facility", "Day", "Hour (±1)", "Nudge (±2)"]
        values = [fac_rate, dow_rate, hour_rate, nudge_rate]
        bars = ax.barh(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Match Rate")
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%}", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Score Distribution")
        score_counts = review["score"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar([f"{i} of 4" for i in score_counts.index], score_counts.values,
                color="#4C72B0")
        ax2.set_xlabel("Outputs Correct")
        ax2.set_ylabel("Count")
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")
    st.subheader("Feature Importance")

    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.json")):
        from facility_predictor.evaluation import get_feature_importance

        model_choice = st.selectbox(
            "Select model", ["facility", "day", "hour", "notification"]
        )
        top10 = get_feature_importance(model_choice)
        names, scores = zip(*top10)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.barh(list(reversed(names)), list(reversed(scores)), color="#4C72B0")
        ax3.set_xlabel("Importance")
        ax3.set_title(f"Top 10 Features — {model_choice.capitalize()} Model")
        ax3.spines[["top", "right"]].set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("Train models to view feature importance.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Prediction Review
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Prediction Review Table")

    filter_opt = st.radio(
        "Filter by score",
        ["All", "Full match (4/4)", "Partial (1-3/4)", "No match (0/4)"],
        horizontal=True,
    )

    filtered = review.copy()
    if filter_opt == "Full match (4/4)":
        filtered = filtered[filtered["score"] == 4]
    elif filter_opt == "Partial (1-3/4)":
        filtered = filtered[(filtered["score"] > 0) & (filtered["score"] < 4)]
    elif filter_opt == "No match (0/4)":
        filtered = filtered[filtered["score"] == 0]

    def _row_color(row: pd.Series) -> list[str]:
        s = row["score"]
        if s == 4:
            bg = "background-color: #d4edda"
        elif s >= 2:
            bg = "background-color: #fff3cd"
        else:
            bg = "background-color: #f8d7da"
        return [bg] * len(row)

    display_cols = [
        "resident_id", "past_bookings",
        "pred_facility", "pred_day", "pred_hour", "pred_nudge",
        "actual_facility", "actual_day", "actual_hour", "actual_booked_at",
        "fac_match", "dow_match", "hour_match", "nudge_match", "score",
    ]

    st.write(f"Showing {len(filtered):,} of {len(review):,} predictions")

    st.dataframe(
        filtered[display_cols].style.apply(_row_color, axis=1),
        use_container_width=True,
        height=500,
    )

    csv_bytes = filtered[display_cols].to_csv(index=False).encode()
    st.download_button(
        label="Download as CSV",
        data=csv_bytes,
        file_name="prediction_review.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Resident Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Resident Explorer")

    residents = sorted(review["resident_id"].unique())
    selected  = st.selectbox("Select a resident", residents)

    res_rows = review[review["resident_id"] == selected]

    if res_rows.empty:
        st.warning("No test predictions for this resident.")
        st.stop()

    row = res_rows.iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Past Bookings (context)**")
        if row["past_bookings"] != "No history":
            past_entries = row["past_bookings"].split(" | ")
            past_rows = []
            for entry in past_entries:
                parts = entry.split("/")
                if len(parts) == 4:
                    past_rows.append({
                        "Facility": parts[0],
                        "Day":      parts[1],
                        "Time":     parts[2],
                        "Booked At": parts[3],
                    })
            if past_rows:
                st.dataframe(pd.DataFrame(past_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No booking history before this test record.")

    with col2:
        st.markdown("**Prediction vs Actual**")
        comparison = pd.DataFrame([
            {
                "Output":    "Facility",
                "Predicted": row["pred_facility"],
                "Actual":    row["actual_facility"],
                "Match":     "YES" if row["fac_match"]   else "NO",
            },
            {
                "Output":    "Day",
                "Predicted": row["pred_day"],
                "Actual":    row["actual_day"],
                "Match":     "YES" if row["dow_match"]   else "NO",
            },
            {
                "Output":    "Hour",
                "Predicted": row["pred_hour"],
                "Actual":    row["actual_hour"],
                "Match":     "YES" if row["hour_match"]  else "NO",
            },
            {
                "Output":    "Nudge / Booked At",
                "Predicted": row["pred_nudge"],
                "Actual":    row["actual_booked_at"],
                "Match":     "YES" if row["nudge_match"] else "NO",
            },
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.markdown(f"**Score: {row['score']} of 4**")

    st.markdown("---")
    st.subheader("AI Explanation")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.info(
            "Add ANTHROPIC_API_KEY to a .env file to enable AI-generated explanations."
        )
    else:
        if st.button("Generate Explanation"):
            from facility_predictor.agent import explain_prediction
            from facility_predictor.evaluation import get_feature_importance

            top_features = get_feature_importance("facility")
            prediction   = {
                "facility": row["pred_facility"],
                "day":      row["pred_day"],
                "hour":     row["pred_hour"],
                "nudge":    row["pred_nudge"],
            }
            with st.spinner("Generating explanation..."):
                explanation = explain_prediction(
                    resident_id=selected,
                    recent_bookings=row["past_bookings"],
                    prediction=prediction,
                    top_features=top_features,
                )
            if explanation:
                st.markdown(f"> {explanation}")
            else:
                st.warning("Could not generate explanation.")