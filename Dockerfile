# syntax=docker/dockerfile:1.7

# -----------------------------
# Builder stage
# -----------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_REQUESTS_TIMEOUT=1200 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages required for building Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip setuptools wheel && \
    pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./

# Install dependencies with cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry config installer.parallel false && \
    poetry install --no-root

# Copy source after dependencies for better cache reuse
COPY . .

# -----------------------------
# Runtime stage
# -----------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 2424

CMD ["python", "-m", "agent_app.main"]