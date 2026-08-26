# Facility Usage Prediction System

A machine learning system that predicts which facility a residential community resident will book next, when they will use it, and when to send them a nudge notification — built on a synthetic dataset of 200 residents across 7 facilities over 12 months.

---

## Quick Start

```bash
git clone <repository-url>
cd facility-predictor
poetry install
make run        # opens at http://localhost:8501
```

The dataset, trained models, and evaluation results are committed to the repository — the app opens fully loaded on first run. Use `make train` only if you want to retrain from scratch.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Business Context](#business-context)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Demo](#demo)
- [Prediction Outputs](#prediction-outputs)
- [Feature Engineering](#feature-engineering)
- [Model Pipeline](#model-pipeline)
- [Evaluation](#evaluation)
- [Limitations](#limitations)

---

## Business Context

Residential communities provide shared facilities — gyms, swimming pools, sports courts, clubhouses, and multipurpose halls. Historical booking behaviour can help predict:

1. **Which facility** a resident is likely to use next
2. **When** (day of week and time of day) they are likely to use it
3. **When to send a notification** to prompt the booking

---

## Architecture

```
synthetic_bookings.csv
        |
        v
+---------------------+
|   generator.py      |  200 residents x 7 facilities x 12 months (2025)
|   Faker + NumPy     |  6 behavioural archetypes, noise, drift, sparsity
+--------+------------+
         |
         v
+---------------------+
|   features.py       |  22 leakage-safe features per booking row
|   Temporal anchor   |  All features strictly anchored to T_inference
+--------+------------+
         |
         v
+----------------------------------------------------------+
|                  pipeline.py - Cascade                   |
|                                                          |
|  [Facility Model] -> [Day Model] -> [Hour Model]         |
|    XGBoost            XGBoost        XGBoost             |
|    Classifier         Classifier     Regressor           |
|         |                 |               |              |
|         +─────────────────+───────────────+-----------+  |
|                                           [Notification]  |
|                                            XGBoost Reg.  |
+--------+-------------------------------------------------+
         |
         v
+---------------------+      +---------------------+
|   evaluation.py     |      |     agent.py        |
|   Metrics + CSV     |      |  LangChain + Claude |
|   prediction_review |      |  Prediction explainer|
+--------+------------+      +----------+----------+
         |                              |
         +--------------+---------------+
                        v
               +-----------------+
               |     app.py      |
               |   Streamlit UI  |
               |  3-tab interface|
               +-----------------+
```

### Data Flow

| Stage | Input | Output |
|---|---|---|
| Generation | Archetype config | `data/synthetic_bookings.csv` |
| Feature engineering | Raw CSV | Feature matrix (22 cols) + target matrix (4 cols) |
| Training | Feature matrix | 4 XGBoost models + MLflow run |
| Evaluation | Test features + saved models | `data/prediction_review.csv` |
| UI | prediction_review.csv | Interactive Streamlit dashboard |
| Explanation | Prediction + feature importances | Natural language via Claude |

### Train / Test Split

```
Jan 2025 ─────────────────── Sep 30  |  Oct 1 ──────── Dec 31
         TRAINING  (274 days)        |    TEST  (92 days)
                                     |
                              cutoff = booking_timestamp < 2025-10-01
```

Test set features are computed using only training-period data — no leakage.

---

## Design Decisions

### 1. Global model over per-resident models

One model trained on all 200 residents. Each row encodes the resident's history as features. This produces better generalisation, handles residents with sparse history, and requires only one training run.

### 2. Chained cascade over independent models

Four models trained in sequence: **Facility -> Day -> Hour -> Notification**. Each model's output becomes an input feature for the next. This captures the natural dependency between predictions (e.g., the hour a resident prefers depends on which facility they are booking).

Teacher forcing is applied during training — actual values flow forward, not predictions — preventing error compounding during fitting.

### 3. XGBoost for all four models

- Handles missing values natively (cold-start residents with no history produce NaN features)
- Works well with mixed feature types (encoded categoricals + floats + cyclical encodings)
- Fast to train and tune
- Feature importances are directly readable — feeds the LLM explanation layer

### 4. Regressor for Hour and Notification, Classifier for Facility and Day

Hour and lead time are continuous and ordinal — a regressor respects that 10:00 is closer to 11:00 than to 23:00. Facility and day of week are categorical with no meaningful order.

### 5. Leakage-safe feature engineering

Every feature for a given row is computed using only records where `booking_timestamp < T_inference`. The implementation sorts all rows chronologically and for row `i`, uses only rows `0..i-1` as history. This applies to both resident-level and community-level aggregations.

### 6. Cyclical encoding for time features

Hour, day of week, and month are encoded as `sin` and `cos` pairs. This ensures the model understands that 23:00 is close to 00:00, Sunday is close to Monday, and December is close to January.

### 7. LangChain + Claude for lightweight explanations

The agent layer takes the top-10 XGBoost feature importances and the prediction, constructs a grounded prompt, and asks Claude to generate a 2-3 sentence explanation. No RAG, no memory, no tools — pure structured prompt engineering. Gracefully disabled if `ANTHROPIC_API_KEY` is absent.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Dependency management | Poetry |
| Data generation | NumPy, Faker |
| Data processing | Pandas |
| Machine learning | XGBoost |
| Preprocessing | scikit-learn |
| Experiment tracking | MLflow (local) |
| LLM / Agent | LangChain + Claude (`claude-haiku-4-5-20251001`) via `langchain-anthropic` |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| Deployment | Streamlit Community Cloud (auto-sync from GitHub) |
| CI | GitHub Actions (lint + test on push) |
| Linting | Ruff |
| Testing | pytest |

---

## Project Structure

```
facility-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Lint + test on every push
│
├── data/                             # Committed — synthetic data included for assignment
│   ├── synthetic_bookings.csv        # Raw synthetic dataset
│   └── prediction_review.csv         # Predictions vs actuals (test set)
│
├── models/                           # Committed — XGBoost artifacts included for assignment
│   ├── facility_model.json
│   ├── day_model.json
│   ├── hour_model.json
│   └── notification_model.json
│
├── mlruns/                           # Gitignored — MLflow local tracking store (regenerated on train)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── src/
│   └── facility_predictor/
│       ├── __init__.py
│       ├── generator.py              # Synthetic data engine
│       ├── features.py               # Leakage-safe feature extraction
│       ├── pipeline.py               # Cascade training + inference + evaluation
│       ├── evaluation.py             # Metrics, error analysis, reporting
│       └── agent.py                  # LangChain + Claude explanation layer
│
├── tests/
│   ├── test_generator.py             # Data generation tests
│   ├── test_features.py              # Feature engineering tests
│   ├── test_pipeline.py              # Pipeline constants and cascade helper tests
│   ├── test_evaluation.py            # Evaluation metric and threshold tests
│   └── test_agent.py                 # Agent graceful degradation tests
│
├── app.py                            # Streamlit UI entry point
├── entrypoint.sh                     # Docker startup script
├── Dockerfile                        # Full app container
├── docker-compose.yml                # Single-command deployment
├── pyproject.toml                    # Dependencies + project metadata
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Git

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd facility-predictor

# 2. Install dependencies
poetry install

# 3. (Optional) Add Claude API key for AI explanations
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

> The system runs fully without an API key. The AI explanation feature in the Resident Explorer tab is the only component that requires one.

---

## How to Run

### Option 1 — Streamlit UI (recommended)

```bash
make run
# or: poetry run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The app opens fully loaded — dataset, models, and evaluation results are all pre-committed. The sidebar buttons (**Generate Data + Train Models**, **Run Evaluation**) are available if you want to regenerate or retrain from scratch.

### Option 2 — Headless (retrain from scratch)

```bash
make train      # train → evaluate → print report
make run        # then launch the UI
```

Or step by step:

```bash
# Generate synthetic dataset (only if you want fresh data)
poetry run python -c "from facility_predictor.generator import generate; generate('data/synthetic_bookings.csv')"

# Train all 4 cascade models
poetry run python -c "from facility_predictor.pipeline import train; train()"

# Evaluate on test set and write prediction_review.csv
poetry run python -c "from facility_predictor.pipeline import evaluate; evaluate()"

# Print terminal evaluation report
poetry run python -c "from facility_predictor.evaluation import print_report; print_report()"
```

### Option 3 — Docker (single command)

```bash
# Build and run — generates data, trains, evaluates, then starts the UI
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

To include your Claude API key for AI explanations:

```bash
ANTHROPIC_API_KEY=your-key-here docker compose up --build
```

Data and models are mounted as volumes so they persist between container restarts.

### Option 4 — MLflow experiment tracking

```bash
poetry shell
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to view all training runs, parameters, and per-model metrics.

### Run tests

```bash
poetry run pytest tests/ -v
```

### Run linter

```bash
poetry run ruff check src/
```

---

## Demo

### Tab 1 — Summary Dashboard

Displays per-output match rates as metric cards, a match-rate bar chart, and a score distribution chart across the full test set. Select any of the 4 models from a dropdown to view its top-10 feature importances.

### Tab 2 — Prediction Review Table

The complete test-set comparison table required by the assignment brief. Each row shows:

| Column | Description |
|---|---|
| `resident_id` | Resident reference |
| `past_bookings` | Last 5 bookings used as context |
| `pred_facility` / `pred_day` / `pred_hour` / `pred_nudge` | Model predictions |
| `actual_facility` / `actual_day` / `actual_hour` / `actual_booked_at` | Ground truth from test set |
| `fac_match` / `dow_match` / `hour_match` / `nudge_match` | Per-output match indicators |
| `score` | Number of outputs correct (0–4) |

Row colours: green = 4/4, yellow = 2-3/4, red = 0-1/4.

Filter by match score and download filtered results as CSV.

### Tab 3 — Resident Explorer

Select any resident from the test set to view:
- Their last 5 bookings as context
- A side-by-side prediction vs actual table with per-output match indicators
- An AI-generated explanation (requires `ANTHROPIC_API_KEY`) grounded in the XGBoost feature importances for that prediction

**Illustrative output:**

| Resident | Past Bookings | Predicted | Actual | Score |
|---|---|---|---|---|
| R-104 | Gym/Mon/07:00, Gym/Wed/07:00 | Gym / Fri / 07:00 / Nudge Thu 18:05 | Gym / Fri / 07:00 / Booked Thu 18:10 | 4 of 4 |
| R-126 | Badminton/Thu/20:00 | Gym / Fri / 19:00 / Nudge Thu 12:30 | Badminton / Fri / 20:00 / Booked Thu 12:30 | 2 of 4 |

---

## Prediction Outputs

| Output | Description | Model type | Evaluation metric |
|---|---|---|---|
| Facility | Which facility the resident will book | XGBoost Classifier (7 classes) | Accuracy + Macro F1 |
| Day | Day of week for the usage | XGBoost Classifier (7 classes) | Accuracy + MAE in days |
| Hour | Hour of day for the usage | XGBoost Regressor (rounded to nearest hour) | MAE + Within-1hr accuracy |
| Notification | Lead time in hours (booking to usage) | XGBoost Regressor | MAE in hours |

---

## Feature Engineering

All 22 features are computed with a strict temporal anchor. Only records with `booking_timestamp < T_inference` are used for any given row.

| Group | Features | Count |
|---|---|---|
| Facility preference | Most booked facility in 7d / 30d / 60d / 90d (encoded 0-6) | 4 |
| Booking count | Total bookings in 7d and 30d | 2 |
| Preferred hour | Most frequent usage hour in 7d and 30d | 2 |
| Preferred day | Most frequent day of week in 7d and 30d | 2 |
| Behavioural drift | Binary flag: 7d preferred facility differs from 30d | 1 |
| Lead time | All-time average and standard deviation of booking lead time (hours) | 2 |
| Community context | Total community bookings (7d), most popular facility community-wide (30d) | 2 |
| Cyclical time | sin/cos encoding of booking hour, day of week, and month | 6 |
| Weekend flag | Binary: booking made on Saturday or Sunday | 1 |
| **Total** | | **22** |

Cascade features added per downstream model:

| Model | Additional input |
|---|---|
| Day | `predicted_facility` |
| Hour | `predicted_facility`, `predicted_dow` |
| Notification | `predicted_facility`, `predicted_dow`, `predicted_hour` |

---

## Model Pipeline

### Cascade with teacher forcing

```
X_base (22 features)
    |
    +---> Facility Model ---> pred_facility (test) / actual_facility (train)
    |                                   |
    +---> Day Model <-------------------+
    |         ---> pred_dow (test) / actual_dow (train)
    |                   |
    +---> Hour Model <--+
    |         ---> pred_hour (test) / actual_hour (train)
    |                   |
    +---> Notification Model <----------+
              ---> pred_lead_time_hours
```

During training, actual target values flow into each subsequent model (teacher forcing). During inference, predicted values flow forward through the cascade.

### Reproducibility

- NumPy random seed: `42`
- XGBoost `random_state`: `42`
- All runs logged to MLflow with parameters, metrics, and model artifacts
- `pyproject.toml` pins all dependency versions exactly

---

## Evaluation

### Match criteria

| Output | Considered a match when |
|---|---|
| Facility | Predicted facility exactly equals actual facility |
| Day | Predicted day of week exactly equals actual day of week |
| Hour | Predicted hour is within ±1 hour of actual |
| Nudge | Predicted lead time is within ±2 hours of actual lead time |

> Tolerances are defined as constants (`HOUR_TOLERANCE`, `NUDGE_TOLERANCE`) in `evaluation.py` — the single source of truth for match criteria.

### Metrics

- **Facility**: Accuracy + Macro F1 (Macro F1 handles the class imbalance — Gym accounts for ~30% of bookings)
- **Day**: Accuracy + MAE in days
- **Hour**: MAE in hours + Within-1hr accuracy rate
- **Notification**: MAE in hours
- **Overall**: Exact match rate (all 4 outputs correct) + full score distribution (0/4 through 4/4)

---

## Limitations

1. **Synthetic data**: The dataset is generated from rule-based archetypes. Real booking behaviour is messier — cancellations, group bookings, seasonal closures, and irregular life events are not modelled.

2. **Cold-start**: Residents with minimal history before the test cutoff will have NaN-heavy feature vectors. XGBoost handles these natively but predictions are less accurate for sparse users.

3. **Fixed facility set**: The model is trained on 7 specific facilities. Adding or renaming a facility requires retraining from scratch.

4. **No capacity modelling**: The model predicts when a resident wants to book, not whether the facility will be available. Peak-hour overcrowding is not accounted for.

5. **Cascade error propagation**: A wrong facility prediction flows into the Day, Hour, and Notification models during inference, pulling downstream predictions off-target.

6. **Static train/test split**: The model is evaluated on Oct-Dec 2025 only. In production, periodic retraining would be required as resident behaviour evolves over time.