FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt requirements-serving.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt
COPY src ./src
COPY data/raw ./data/raw
COPY README.md ./README.md
# Training publishes only a completely verified bundle.
RUN python -m src.models.train
RUN python -c "from src.models.artifacts import load_bundle; load_bundle()"

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements-serving.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt -c requirements-lock.txt
COPY src/__init__.py src/config.py ./src/
COPY src/data/__init__.py src/data/validate_data.py ./src/data/
COPY src/features ./src/features
COPY src/models/__init__.py src/models/artifacts.py src/models/predict.py src/models/local_explain.py ./src/models/
COPY app ./app
COPY --from=builder /app/models ./models
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
