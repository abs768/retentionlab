FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /service
RUN useradd --create-home --uid 10001 retentionlab
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir -e .
COPY models/final_model.joblib ./models/final_model.joblib
COPY artifacts ./artifacts
COPY scripts/verify_model_artifact.py ./scripts/verify_model_artifact.py
RUN python scripts/verify_model_artifact.py
USER retentionlab
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
