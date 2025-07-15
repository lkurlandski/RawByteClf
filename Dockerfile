# ---------- stage 1: base environment ----------
# —— GPU variant (comment out if CPU only) ——
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
# —— CPU variant —
# FROM python:3.10-slim AS base

# 1. Create a non-root user (good hygiene)
ARG USERNAME=appuser
RUN useradd -ms /bin/bash $USERNAME
WORKDIR /home/$USERNAME/app

# ---- ADD THIS BLOCK ----
# Install `tree` (a 120 KB binary) in one layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends tree \
    && rm -rf /var/lib/apt/lists/*
# ------------------------

# 2. Copy python environment and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy only what's needed for the demo
COPY --chown=appuser:appuser demo/ ./demo/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser config/default.json ./config/default.json

# 4. Set environment variables (optional)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

USER $USERNAME

# 5. Default command
ENTRYPOINT []
CMD ["python", "demo/hello.py"]
