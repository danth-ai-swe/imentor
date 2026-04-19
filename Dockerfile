# ── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/install/lib/python3.12/site-packages

WORKDIR /app

# curl cần thiết cho HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /install

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

COPY --chown=appuser:appgroup . .

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "main.py"]