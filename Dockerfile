# ---------- stage 1: base environment ----------
# —— GPU variant (comment out if CPU only) ——
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS base
# —— CPU variant —
# FROM python:3.10-slim AS base

# 1. Create a non-root user (good hygiene)
ARG USERNAME=appuser
RUN useradd -ms /bin/bash $USERNAME
WORKDIR /home/$USERNAME/app

# 2. Copy python environment and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy only what's needed for the demo
COPY demo/ ./demo/

# 4. Set environment variables (optional)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

USER $USERNAME

# 5. Default command
ENTRYPOINT ["python", "demo/demo.py"]

