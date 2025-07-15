# ---------- stage 1: base environment ----------
# —— GPU variant (comment out if CPU only) ——
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
# —— CPU variant —
# FROM python:3.10-slim AS base

# 1. Create a non-root user (good hygiene)
ARG USERNAME=appuser
RUN useradd -ms /bin/bash $USERNAME
WORKDIR /home/$USERNAME/app
RUN mkdir -p /home/$USERNAME/app && chown -R $USERNAME:$USERNAME /home/$USERNAME/app

# ---- ADD THIS BLOCK ----
# Install `tree` (a 120 KB binary) in one layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends tree \
    && rm -rf /var/lib/apt/lists/*
# ------------------------

# 2. Copy python environment and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser demo/data ./data/
COPY --chown=appuser:appuser demo/datasets ./datasets/
COPY --chown=appuser:appuser demo/tests.sh ./demo/tests.sh
COPY --chown=appuser:appuser demo/hello.py ./demo/hello.py
COPY --chown=appuser:appuser demo/prepare.sh ./demo/prepare.sh
COPY --chown=appuser:appuser demo/preprocess.sh ./demo/preprocess.sh
COPY --chown=appuser:appuser demo/gencaches.sh ./demo/gencaches.sh
COPY --chown=appuser:appuser demo/tokenize.sh ./demo/tokenize.sh
COPY --chown=appuser:appuser demo/create.py ./demo/create.py
COPY --chown=appuser:appuser cache/empty_shas.txt ./cache/empty_shas.txt
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser config/default.json ./config/default.json

# 4. Set environment variables (optional)
ENV PYTHONUNBUFFERED=1 \
    LMLM_GET_MATERIALS_ESP_FAM_MIN_FREQ=2 \
    LMLM_GET_MATERIALS_ESP_FAM_MAX_IMBALANCE_RATIO=1000 \
    LMLM_GET_MATERIALS_ESP_FAM_TOP_K=3 \
    LMLM_GET_MATERIALS_ESP_BEH_MIN_FREQ=2 \
    LMLM_GET_MATERIALS_ESP_BEH_MAX_IMBALANCE_RATIO=1000 \
    LMLM_GET_MATERIALS_ESP_BEH_TOP_K=3 \
    LMLM_GET_MATERIALS_ESP_CLM_VL_SIZE=64 \
    LMLM_GET_MATERIALS_ESP_MLM_VL_SIZE=64

USER $USERNAME

# 5. Default command
ENTRYPOINT []
CMD ["python", "demo/hello.py"]
