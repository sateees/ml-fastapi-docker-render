# Stage 1: Builder image
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model
RUN python model/train_model.py

# Stage 2: Final image
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model.joblib from the builder stage
COPY --from=builder /app/model.joblib /app/model.joblib

# Copy your FastAPI code
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
