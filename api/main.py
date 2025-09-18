import joblib
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import os
from datetime import datetime
from sklearn.cluster import KMeans

# 1. SETUP & SECURITY
app = FastAPI(title="Real Estate Price Prediction API", version="1.0")

API_KEY = os.getenv("API_KEY", "default_secret_key")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Dependency function to validate the API key."""
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials",
        )

# 2. LOAD ARTIFACTS
MODEL_DIR = "../model"
DATA_DIR = "../data"

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_location_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.json")
DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, "zipcode_demographics.csv")

try:
    model = joblib.load(MODEL_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)
    with open(FEATURES_PATH, 'r') as f:
        model_features = json.load(f)
    zipcode_demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': int}).set_index('zipcode')
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load necessary artifacts. Error: {e}")


# 3. DEFINE INPUT DATA MODEL
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., example=3)
    bathrooms: float = Field(..., example=2.25)
    sqft_living: int = Field(..., example=2570)
    sqft_lot: int = Field(..., example=7242)
    floors: float = Field(..., example=2.0)
    waterfront: int = Field(..., example=0)
    view: int = Field(..., example=0)
    condition: int = Field(..., example=3)
    grade: int = Field(..., example=7)
    sqft_above: int = Field(..., example=2170)
    sqft_basement: int = Field(..., example=400)
    yr_built: int = Field(..., example=1951)
    yr_renovated: int = Field(..., example=1991)
    zipcode: int = Field(..., example=98125)
    lat: float = Field(..., example=47.7210)
    long: float = Field(..., example=-122.319)
    sqft_living15: int = Field(..., example=1690)
    sqft_lot15: int = Field(..., example=7639)


def engineer_features(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Apply all feature engineering steps to the raw dataset."""
    df_eng = df.copy()
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%dT%H%M%S")
    df_eng['date'] = pd.to_datetime(formatted_time)
    df_eng['year_sold'] = df_eng['date'].dt.year
    df_eng['month_sold'] = df_eng['date'].dt.month
    df_eng['house_age'] = df_eng['year_sold'] - df_eng['yr_built']
    df_eng['was_renovated'] = (df_eng['yr_renovated'] > 0).astype(int)
    df_eng['has_basement'] = (df_eng['sqft_basement'] > 0).astype(int)
    df_eng['location_cluster'] = kmeans.predict(df[['lat', 'long']])
    cols_to_drop = ['date', 'yr_built', 'yr_renovated', 'lat', 'long']
    df_eng = df_eng.drop(columns=cols_to_drop)
    return df_eng


# 4. API ENDPOINTS
@app.get("/", summary="Health Check")
def read_root():
    """Root endpoint for health checking."""
    return {"message": "Welcome to the phData Real Estate Prediction API"}

@app.post("/predict", summary="Predict House Price")
def predict_price(
    features: HouseFeatures,
    api_key: str = Depends(get_api_key) # Authentication
):
    """
    Receives house features, joins demographic data, and returns a price prediction.
    Requires a valid API key in the 'X-API-Key' header.
    """
    input_df = pd.DataFrame([features.dict()])
    eng_df = engineer_features(input_df, kmeans_model)
    
    if features.zipcode not in zipcode_demographics.index:
        raise HTTPException(
            status_code=404, 
            detail=f"Demographic data not found for zipcode {features.zipcode}"
        )
        
    full_df = eng_df.merge(
        zipcode_demographics, 
        how="left", 
        left_on="zipcode", 
        right_index=True
    ).drop(columns=['zipcode'])

    try:
        prediction_features = full_df[model_features]
    except KeyError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"An expected feature is missing: {e}"
        )

    log_prediction = model.predict(prediction_features)
    actual_price = np.expm1(log_prediction[0])

    return {
        "predicted_price": round(float(actual_price), 2),
        "model_version": "1.0-baseline"
    }
