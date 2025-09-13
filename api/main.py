import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# --- 1. SETUP ---
# Create a FastAPI app instance
app = FastAPI(title="Real Estate Price Prediction API", version="1.0")

# --- 2. LOAD ARTIFACTS ---
# Define paths to model and data artifacts
# Adjust these paths if your file structure is different.
MODEL_DIR = "../model"
DATA_DIR = "../data"

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.json")
DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, "zipcode_demographics.csv")

# Load the artifacts at startup to avoid reloading them on every request
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        model_features = json.load(f)
    
    # Load demographics data for backend joining
    zipcode_demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': int}).set_index('zipcode')
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load necessary artifacts. Make sure paths are correct. Error: {e}")

# --- 3. DEFINE INPUT DATA MODEL ---
# Pydantic model for input validation.
# These fields should match the columns in 'future_unseen_examples.csv'
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., example=3, description="Number of bedrooms")
    bathrooms: float = Field(..., example=2.25, description="Number of bathrooms")
    sqft_living: int = Field(..., example=2570, description="Square footage of the living space")
    sqft_lot: int = Field(..., example=7242, description="Square footage of the lot")
    floors: float = Field(..., example=2.0, description="Number of floors")
    waterfront: int = Field(..., example=0, description="Whether the property has a waterfront (0 or 1)")
    view: int = Field(..., example=0, description="Quality of the view (0-4)")
    condition: int = Field(..., example=3, description="Condition of the house (1-5)")
    grade: int = Field(..., example=7, description="Construction grade (1-13)")
    sqft_above: int = Field(..., example=2170, description="Square footage above ground")
    sqft_basement: int = Field(..., example=400, description="Square footage of the basement")
    yr_built: int = Field(..., example=1951, description="Year built")
    yr_renovated: int = Field(..., example=1991, description="Year renovated (0 if never)")
    zipcode: int = Field(..., example=98125, description="Zip code of the property")
    lat: float = Field(..., example=47.7210, description="Latitude")
    long: float = Field(..., example=-122.319, description="Longitude")
    sqft_living15: int = Field(..., example=1690, description="Average living space of 15 nearest neighbors")
    sqft_lot15: int = Field(..., example=7639, description="Average lot size of 15 nearest neighbors")
    
# --- 4. API ENDPOINTS ---
@app.get("/", summary="Health Check")
def read_root():
    """
    Root endpoint for health checking.
    Returns a welcome message if the API is running.
    """
    return {"message": "Welcome to the phData Real Estate Prediction API"}

@app.post("/predict", summary="Predict House Price")
def predict_price(features: HouseFeatures):
    """
    Receives house features, joins demographic data, and returns a price prediction.

    - **Input**: JSON object with house features.
    - **Output**: JSON object with the model's price prediction.
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # --- Backend Data Joining ---
    # Check if the zipcode from the input exists in our demographics data
    if features.zipcode not in zipcode_demographics.index:
        raise HTTPException(
            status_code=404, 
            detail=f"Demographic data not found for zipcode {features.zipcode}"
        )
        
    # Join the demographic data based on zipcode
    full_df = input_df.merge(
        zipcode_demographics, 
        how="left", 
        left_on="zipcode", 
        right_index=True
    )

    # --- Prediction ---
    # Ensure columns are in the same order as during model training
    try:
        prediction_features = full_df[model_features]
    except KeyError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"An expected feature is missing from the input or joined data: {e}"
        )

    # Make the prediction
    prediction = model.predict(prediction_features)

    # Return the result
    return {
        "predicted_price": round(prediction[0], 2),
        "model_version": "1.0-baseline"
    }

# To run this API:
# 1. Make sure you have fastapi and uvicorn installed: pip install fastapi "uvicorn[standard]"
# 2. Navigate to the 'api' directory in your terminal.
# 3. Run the command: uvicorn main:app --reload

