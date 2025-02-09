from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Define a request body schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

# Load the model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API!"}

@app.post("/predict")
def predict_species(features: IrisFeatures):
    # Convert input features to the format the model expects
    data = [[
        features.sepal_length, 
        features.sepal_width, 
        features.petal_length, 
        features.petal_width
    ]]
    
    prediction = model.predict(data)[0]
    # The Iris dataset typically has target classes 0, 1, 2
    # For clarity, you might map them to species names
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species_name = species_map[prediction]
    return {
        "prediction": int(prediction),
        "species": species_name
    }

