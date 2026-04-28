FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# Copy models if they exist (optional — can also mount via volume)
COPY models/ ./models/ 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "mlops_canary_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
