FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN mkdir -p /app/outputs /app/.cache/huggingface


CMD ["bash", "-c", "echo 'Available scripts:'; ls scripts/; echo; echo 'See REPRODUCE.md for the full pipeline.'"]
