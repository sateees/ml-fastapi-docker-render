# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files and the app code
COPY model.joblib ./
COPY app ./app
# If you want to also include the training script, you can COPY model/train_model.py as well

# Expose the port (FastAPI default 8000)
EXPOSE 8000

# Run the API with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
