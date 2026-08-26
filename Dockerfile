FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==2.4.1

# Copy dependency files first (layer cache — only invalidated when deps change)
COPY pyproject.toml poetry.lock README.md ./

# Increase pip download timeout for large packages (scipy, mlflow, etc.)
ENV PIP_DEFAULT_TIMEOUT=300

# Install production dependencies only (no pytest / ruff)
# --no-root: skip installing the project package itself (src/ not copied yet)
RUN poetry config virtualenvs.create false \
 && poetry config installer.max-workers 4 \
 && poetry install --without dev --no-interaction --no-ansi --no-root

# Copy source
COPY src/ ./src/
COPY app.py ./
COPY entrypoint.sh ./

# Make the local package importable without a pip install step
ENV PYTHONPATH=/app/src

RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]