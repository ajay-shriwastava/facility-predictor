"""
Chained XGBoost training and inference pipeline.

Cascade order: Facility → Day → Hour → Notification
Teacher forcing is used during training (actual values flow forward).
Predicted values flow forward during inference.

Entry points
------------
train()    — train all 4 models, log to MLflow, save artifacts
evaluate() — run cascade inference on test set, write prediction_review.csv
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from facility_predictor.features import FEATURE_COLS, build_feature_matrix
from facility_predictor.generator import FACILITIES, generate

# ── Paths & constants ──────────────────────────────────────────────────────────

DATA_PATH      = Path("data/synthetic_bookings.csv")
MODELS_DIR     = Path("models")
REVIEW_PATH    = Path("data/prediction_review.csv")
FEATURES_CACHE = Path("data/features_cache.parquet")
TARGETS_CACHE  = Path("data/targets_cache.parquet")

TRAIN_CUTOFF = pd.Timestamp("2025-10-01")

FAC_TO_IDX: dict[str, int] = {f: i for i, f in enumerate(FACILITIES)}
IDX_TO_FAC: dict[int, str] = {i: f for i, f in enumerate(FACILITIES)}
DAY_NAMES  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# XGBoost shared params
_XGB_COMMON = dict(n_estimators=300, learning_rate=0.1, random_state=42,
                   tree_method="hist", verbosity=0)


# ── Model constructors ─────────────────────────────────────────────────────────

def _facility_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **_XGB_COMMON,
        max_depth=6,
        num_class=len(FACILITIES),
        objective="multi:softprob",
        eval_metric="mlogloss",
    )


def _day_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **_XGB_COMMON,
        max_depth=5,
        num_class=7,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )


def _hour_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        **_XGB_COMMON,
        max_depth=5,
        objective="reg:squarederror",
        eval_metric="rmse",
    )


def _notification_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        **_XGB_COMMON,
        max_depth=5,
        objective="reg:squarederror",
        eval_metric="rmse",
    )


# ── Data loading & splitting ───────────────────────────────────────────────────

def _load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data, build features, split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test  — all aligned DataFrames
    """
    if not DATA_PATH.exists():
        print("Dataset not found — generating...")
        generate(DATA_PATH)

    raw = pd.read_csv(DATA_PATH)
    raw["booking_timestamp"] = pd.to_datetime(raw["booking_timestamp"])
    raw = raw.sort_values("booking_timestamp").reset_index(drop=True)

    print(f"Building feature matrix for {len(raw):,} bookings...")
    features, targets = build_feature_matrix(raw)
    features.to_parquet(FEATURES_CACHE)
    targets.to_parquet(TARGETS_CACHE)

    mask_train = raw["booking_timestamp"] < TRAIN_CUTOFF
    mask_test  = raw["booking_timestamp"] >= TRAIN_CUTOFF

    X_train = features[mask_train].reset_index(drop=True)
    X_test  = features[mask_test].reset_index(drop=True)
    y_train = targets[mask_train].reset_index(drop=True)
    y_test  = targets[mask_test].reset_index(drop=True)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


# ── Cascade feature builders ───────────────────────────────────────────────────

def _add_facility(X: pd.DataFrame, fac_encoded: np.ndarray) -> pd.DataFrame:
    return X.assign(predicted_facility=fac_encoded)


def _add_dow(X: pd.DataFrame, dow: np.ndarray) -> pd.DataFrame:
    return X.assign(predicted_dow=dow)


def _add_hour(X: pd.DataFrame, hour: np.ndarray) -> pd.DataFrame:
    return X.assign(predicted_hour=hour)


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_path: Path = DATA_PATH, models_dir: Path = MODELS_DIR) -> dict:
    """
    Train the 4-model cascade with teacher forcing.

    Logs parameters and metrics to MLflow.
    Saves model artifacts to models_dir.

    Returns
    -------
    dict of training metrics for each model.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = _load_and_prepare()

    y_fac_train  = y_train["facility_id"].map(FAC_TO_IDX).astype(int)
    y_dow_train  = y_train["usage_dow"].astype(int)
    y_hour_train = y_train["usage_hour"].astype(float)
    y_lead_train = y_train["lead_time_hours"].astype(float)

    y_fac_test   = y_test["facility_id"].map(FAC_TO_IDX).astype(int)
    y_dow_test   = y_test["usage_dow"].astype(int)
    y_hour_test  = y_test["usage_hour"].astype(float)
    y_lead_test  = y_test["lead_time_hours"].astype(float)

    metrics: dict = {}

    mlflow.set_experiment("facility-predictor")

    with mlflow.start_run(run_name="cascade_training"):
        mlflow.log_params({
            "train_cutoff":  str(TRAIN_CUTOFF.date()),
            "n_estimators":  _XGB_COMMON["n_estimators"],
            "learning_rate": _XGB_COMMON["learning_rate"],
            "train_rows":    len(X_train),
            "test_rows":     len(X_test),
            "n_facilities":  len(FACILITIES),
        })

        # ── Model 1: Facility ──────────────────────────────────────────────────
        print("Training facility model...")
        m_fac = _facility_model()
        m_fac.fit(X_train[FEATURE_COLS], y_fac_train)

        fac_test_pred  = m_fac.predict(X_test[FEATURE_COLS])

        fac_acc = float(accuracy_score(y_fac_test, fac_test_pred))
        fac_f1  = float(f1_score(y_fac_test, fac_test_pred, average="macro", zero_division=0))
        metrics["facility_accuracy"] = fac_acc
        metrics["facility_f1_macro"] = fac_f1
        mlflow.log_metrics({"facility_accuracy": fac_acc, "facility_f1_macro": fac_f1})
        m_fac.save_model(models_dir / "facility_model.json")
        print(f"  Facility — Accuracy: {fac_acc:.3f}  Macro-F1: {fac_f1:.3f}")

        # ── Model 2: Day (teacher forcing: use actual facility) ────────────────
        print("Training day model...")
        X_train_day = _add_facility(X_train[FEATURE_COLS], y_fac_train.values)
        X_test_day  = _add_facility(X_test[FEATURE_COLS],  fac_test_pred)

        m_day = _day_model()
        m_day.fit(X_train_day, y_dow_train)

        dow_test_pred  = m_day.predict(X_test_day)

        dow_acc = float(accuracy_score(y_dow_test, dow_test_pred))
        dow_mae = float(mean_absolute_error(y_dow_test, dow_test_pred))
        metrics["day_accuracy"] = dow_acc
        metrics["day_mae"]      = dow_mae
        mlflow.log_metrics({"day_accuracy": dow_acc, "day_mae": dow_mae})
        m_day.save_model(models_dir / "day_model.json")
        print(f"  Day — Accuracy: {dow_acc:.3f}  MAE: {dow_mae:.2f} days")

        # ── Model 3: Hour (teacher forcing: use actual facility + actual dow) ──
        print("Training hour model...")
        X_train_hour = _add_dow(_add_facility(X_train[FEATURE_COLS], y_fac_train.values), y_dow_train.values)
        X_test_hour  = _add_dow(_add_facility(X_test[FEATURE_COLS],  fac_test_pred),      dow_test_pred)

        m_hour = _hour_model()
        m_hour.fit(X_train_hour, y_hour_train)

        hour_test_pred  = np.clip(np.round(m_hour.predict(X_test_hour)),  0, 23)

        hour_mae    = float(mean_absolute_error(y_hour_test, hour_test_pred))
        hour_pm1    = float(np.mean(np.abs(y_hour_test.values - hour_test_pred) <= 1))
        metrics["hour_mae"]          = hour_mae
        metrics["hour_within_1hr"]   = hour_pm1
        mlflow.log_metrics({"hour_mae": hour_mae, "hour_within_1hr": hour_pm1})
        m_hour.save_model(models_dir / "hour_model.json")
        print(f"  Hour — MAE: {hour_mae:.2f} hrs  Within ±1hr: {hour_pm1:.3f}")

        # ── Model 4: Notification (teacher forcing: actual fac + dow + hour) ──
        print("Training notification model...")
        X_train_notif = _add_hour(X_train_hour.copy(), y_hour_train.values)
        X_test_notif  = _add_hour(X_test_hour.copy(),  hour_test_pred)

        m_notif = _notification_model()
        m_notif.fit(X_train_notif, y_lead_train)

        lead_test_pred = np.clip(m_notif.predict(X_test_notif), 1, None)
        lead_mae = float(mean_absolute_error(y_lead_test, lead_test_pred))
        metrics["notification_mae"] = lead_mae
        mlflow.log_metric("notification_mae", lead_mae)
        m_notif.save_model(models_dir / "notification_model.json")
        print(f"  Notification — MAE: {lead_mae:.2f} hrs")

        mlflow.log_artifact(str(models_dir / "facility_model.json"))
        mlflow.log_artifact(str(models_dir / "day_model.json"))
        mlflow.log_artifact(str(models_dir / "hour_model.json"))
        mlflow.log_artifact(str(models_dir / "notification_model.json"))

    print("Training complete.")
    return metrics


# ── Inference ──────────────────────────────────────────────────────────────────

def _load_models(models_dir: Path) -> tuple:
    m_fac   = xgb.XGBClassifier();   m_fac.load_model(models_dir / "facility_model.json")
    m_day   = xgb.XGBClassifier();   m_day.load_model(models_dir / "day_model.json")
    m_hour  = xgb.XGBRegressor();    m_hour.load_model(models_dir / "hour_model.json")
    m_notif = xgb.XGBRegressor();    m_notif.load_model(models_dir / "notification_model.json")
    return m_fac, m_day, m_hour, m_notif


def _cascade_predict(
    X: pd.DataFrame,
    m_fac: xgb.XGBClassifier,
    m_day: xgb.XGBClassifier,
    m_hour: xgb.XGBRegressor,
    m_notif: xgb.XGBRegressor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run cascade inference. Returns (fac_idx, dow, hour, lead_hours)."""
    fac_pred  = m_fac.predict(X[FEATURE_COLS])

    X_day     = _add_facility(X[FEATURE_COLS], fac_pred)
    dow_pred  = m_day.predict(X_day)

    X_hour    = _add_dow(X_day, dow_pred)
    hour_pred = np.clip(np.round(m_hour.predict(X_hour)), 0, 23)

    X_notif   = _add_hour(X_hour.copy(), hour_pred)
    lead_pred = np.clip(m_notif.predict(X_notif), 1, None)

    return fac_pred, dow_pred, hour_pred, lead_pred


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    output_path: Path = REVIEW_PATH,
) -> dict:
    """
    Evaluate cascade on the chronological test set and write prediction_review.csv.

    Returns
    -------
    dict of test metrics.
    """
    raw = pd.read_csv(data_path)
    raw["booking_timestamp"] = pd.to_datetime(raw["booking_timestamp"])
    raw = raw.sort_values("booking_timestamp").reset_index(drop=True)

    if FEATURES_CACHE.exists() and TARGETS_CACHE.exists():
        print("Loading cached feature matrix...")
        features = pd.read_parquet(FEATURES_CACHE)
        targets  = pd.read_parquet(TARGETS_CACHE)
    else:
        print(f"Building feature matrix for {len(raw):,} bookings...")
        features, targets = build_feature_matrix(raw)

    mask_test = raw["booking_timestamp"] >= TRAIN_CUTOFF
    X_test    = features[mask_test].reset_index(drop=True)
    y_test    = targets[mask_test].reset_index(drop=True)
    raw_test  = raw[mask_test].reset_index(drop=True)

    m_fac, m_day, m_hour, m_notif = _load_models(models_dir)

    fac_pred, dow_pred, hour_pred, lead_pred = _cascade_predict(
        X_test, m_fac, m_day, m_hour, m_notif
    )

    # ── Decode predictions ────────────────────────────────────────────────────
    pred_facility = [IDX_TO_FAC.get(int(i), "Unknown") for i in fac_pred]
    pred_dow      = [DAY_NAMES[int(d) % 7] for d in dow_pred]
    pred_hour_str = [f"{int(h):02d}:00" for h in hour_pred]

    # Nudge = predicted usage time minus predicted lead time (day + HH:MM)
    pred_nudge: list[str] = []
    for i in range(len(raw_test)):
        usage_dow_idx = int(dow_pred[i]) % 7
        usage_h       = int(hour_pred[i])
        lead_h        = float(lead_pred[i])
        nudge_minutes = usage_dow_idx * 24 * 60 + usage_h * 60 - int(lead_h * 60)
        nudge_dow_idx = (nudge_minutes // (24 * 60)) % 7
        nudge_hour    = (nudge_minutes % (24 * 60)) // 60
        nudge_min     = nudge_minutes % 60
        pred_nudge.append(f"{DAY_NAMES[nudge_dow_idx]} {nudge_hour:02d}:{nudge_min:02d}")

    # ── Actual values ─────────────────────────────────────────────────────────
    act_facility  = y_test["facility_id"].tolist()
    act_dow       = [DAY_NAMES[int(d)] for d in y_test["usage_dow"]]
    act_hour_str  = [f"{int(h):02d}:00" for h in y_test["usage_hour"]]
    act_booked_at = raw_test["booking_timestamp"].dt.strftime("%a %H:%M").tolist()

    # Past bookings: last 5 for each test resident before this booking
    res_history = {rid: grp for rid, grp in raw.groupby("resident_id")}
    past_str: list[str] = []
    for _, row in raw_test.iterrows():
        rid  = row["resident_id"]
        t    = row["booking_timestamp"]
        hist = res_history[rid]
        hist = hist[hist["booking_timestamp"] < t].tail(5)
        lines = []
        for _, h in hist.iterrows():
            fac  = h["facility_id"]
            ud   = pd.Timestamp(h["usage_timestamp"])
            bd   = pd.Timestamp(h["booking_timestamp"])
            lines.append(f"{fac}/{ud.strftime('%a')}/{ud.strftime('%H:%M')}/{bd.strftime('%a %H:%M')}")
        past_str.append(" | ".join(lines) if lines else "No history")

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_fac_true = y_test["facility_id"].map(FAC_TO_IDX).astype(int).values
    metrics = {
        "facility_accuracy": float(accuracy_score(y_fac_true, fac_pred)),
        "facility_f1_macro": float(f1_score(y_fac_true, fac_pred, average="macro", zero_division=0)),
        "day_accuracy":      float(accuracy_score(y_test["usage_dow"].astype(int), dow_pred.astype(int))),
        "day_mae":           float(mean_absolute_error(y_test["usage_dow"], dow_pred)),
        "hour_mae":          float(mean_absolute_error(y_test["usage_hour"], hour_pred)),
        "hour_within_1hr":   float(np.mean(np.abs(y_test["usage_hour"].values - hour_pred) <= 1)),
        "notification_mae":  float(mean_absolute_error(y_test["lead_time_hours"], lead_pred)),
    }

    # ── Write review CSV ──────────────────────────────────────────────────────
    review = pd.DataFrame({
        "booking_id":              raw_test["booking_id"].values,
        "resident_id":             raw_test["resident_id"].values,
        "past_bookings":           past_str,
        "pred_facility":           pred_facility,
        "pred_day":                pred_dow,
        "pred_hour":               pred_hour_str,
        "pred_nudge":              pred_nudge,
        "pred_lead_time_hours":    lead_pred,
        "actual_facility":         act_facility,
        "actual_day":              act_dow,
        "actual_hour":             act_hour_str,
        "actual_booked_at":        act_booked_at,
        "actual_lead_time_hours":  y_test["lead_time_hours"].values,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False)

    print(f"\nEvaluation complete — {len(review):,} test predictions written to {output_path}")
    print(f"  Facility Accuracy : {metrics['facility_accuracy']:.3f}")
    print(f"  Facility Macro-F1 : {metrics['facility_f1_macro']:.3f}")
    print(f"  Day Accuracy      : {metrics['day_accuracy']:.3f}")
    print(f"  Hour MAE          : {metrics['hour_mae']:.2f} hrs")
    print(f"  Notification MAE  : {metrics['notification_mae']:.2f} hrs")

    mlflow.set_experiment("facility-predictor")
    with mlflow.start_run(run_name="cascade_evaluation"):
        mlflow.log_metrics({
            "eval_facility_accuracy": metrics["facility_accuracy"],
            "eval_facility_f1_macro": metrics["facility_f1_macro"],
            "eval_day_accuracy":      metrics["day_accuracy"],
            "eval_day_mae":           metrics["day_mae"],
            "eval_hour_mae":          metrics["hour_mae"],
            "eval_hour_within_1hr":   metrics["hour_within_1hr"],
            "eval_notification_mae":  metrics["notification_mae"],
        })
        mlflow.log_artifact(str(output_path))

    return metrics
