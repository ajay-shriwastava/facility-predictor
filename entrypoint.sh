#!/bin/bash
set -e

echo "=== Facility Usage Prediction System ==="

# Generate synthetic dataset if not present
if [ ! -f "data/synthetic_bookings.csv" ]; then
    echo "Generating synthetic dataset..."
    python -c "from facility_predictor.generator import generate; generate('data/synthetic_bookings.csv')"
fi

# Train models if not present
if [ ! -f "models/facility_model.json" ]; then
    echo "Training cascade models (this takes a few minutes)..."
    python -c "from facility_predictor.pipeline import train; train()"
fi

# Run evaluation if review not present
if [ ! -f "data/prediction_review.csv" ]; then
    echo "Running evaluation on test set..."
    python -c "from facility_predictor.pipeline import evaluate; evaluate()"
fi

echo "Starting Streamlit..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true