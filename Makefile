.PHONY: install run train test lint help

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install   Install dependencies via Poetry"
	@echo "  run       Launch the Streamlit app"
	@echo "  train     Run full ML pipeline: train → evaluate → report"
	@echo "  test      Run test suite"
	@echo "  lint      Run Ruff linter"

install:
	poetry install

run:
	streamlit run app.py

train:
	poetry run python -c "from facility_predictor.pipeline import train; train()"
	poetry run python -c "from facility_predictor.pipeline import evaluate; evaluate()"
	poetry run python -c "from facility_predictor.evaluation import print_report; print_report()"

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check src/