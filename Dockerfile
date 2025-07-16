# Set up the base image for the Docker container (GPU/CPU).
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
# FROM python:3.10-slim AS base

# Set up the user and working directory.
ARG USERNAME=appuser
RUN useradd -ms /bin/bash $USERNAME
WORKDIR /home/$USERNAME/app
RUN mkdir -p /home/$USERNAME/app && chown -R $USERNAME:$USERNAME /home/$USERNAME/app

# Install useful tools for debugging.
RUN apt-get update
RUN apt-get install -y --no-install-recommends tree
RUN apt-get install -y --no-install-recommends vim
RUN rm -rf /var/lib/apt/lists/*

# Copy demo components.
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
COPY --chown=appuser:appuser cache/processedShas.txt ./cache/processedShas.txt
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser config/default.json ./config/default.json

# Install Python dependencies.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables.
ENV PYTHONUNBUFFERED=1
ENV LMLM_GET_MATERIALS_ESP_FAM_MIN_FREQ=2
ENV LMLM_GET_MATERIALS_ESP_FAM_MAX_IMBALANCE_RATIO=1000
ENV LMLM_GET_MATERIALS_ESP_FAM_TOP_K=3
ENV LMLM_GET_MATERIALS_ESP_BEH_MIN_FREQ=2
ENV LMLM_GET_MATERIALS_ESP_BEH_MAX_IMBALANCE_RATIO=1000
ENV LMLM_GET_MATERIALS_ESP_BEH_TOP_K=3
ENV LMLM_GET_MATERIALS_ESP_CLM_VL_SIZE=64
ENV LMLM_GET_MATERIALS_ESP_MLM_VL_SIZE=64
ENV LMLM_GET_MATERIALS_ESP_LM_VL_SIZE=64

# Set the default user.
USER $USERNAME

# Default command to run when the container starts.
ENTRYPOINT []
CMD ["python", "demo/hello.py"]
