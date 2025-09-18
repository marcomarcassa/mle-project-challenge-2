import os
import pathlib
import pandas
import joblib
from sklearn.cluster import KMeans

# CONFIGS
INPUT_DIR = "data"
OUTPUT_DIR = "processed_data"
MODELS_DIR = "model"
SALES_PATH = os.path.join(INPUT_DIR, "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join(INPUT_DIR, "zipcode_demographics.csv")

PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_kc_house_data.csv")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_location_model.pkl")

SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode', 'date', 'yr_built',
    'yr_renovated', 'lat', 'long'
]

def engineer_features(df: pandas.DataFrame) -> pandas.DataFrame:
    """Apply all feature engineering steps to the raw dataset."""
    print("Starting feature engineering...")
    df_eng = df.copy()

    # Date Features
    df_eng['date'] = pandas.to_datetime(df_eng['date'])
    df_eng['year_sold'] = df_eng['date'].dt.year
    df_eng['month_sold'] = df_eng['date'].dt.month

    # Age & Renovation Features
    df_eng['house_age'] = df_eng['year_sold'] - df_eng['yr_built']
    df_eng['was_renovated'] = (df_eng['yr_renovated'] > 0).astype(int)

    # Basement Feature
    df_eng['has_basement'] = (df_eng['sqft_basement'] > 0).astype(int)

    # Location Clustering
    print("Fitting KMeans model for location clustering...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df_eng['location_cluster'] = kmeans.fit_predict(df[['lat', 'long']])
    
    # Save the fitted KMeans model for later use in the API
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    print(f"KMeans model saved to {KMEANS_MODEL_PATH}")

    # Cleanup
    cols_to_drop = ['date', 'yr_built', 'yr_renovated', 'lat', 'long', 'zipcode']
    df_eng = df_eng.drop(columns=cols_to_drop)
    
    print("Feature engineering complete.")
    return df_eng

def main():
    """
    Main ETL script to load, process, and save the feature-engineered data.
    """
    print("--- Starting Data Processing ETL ---")
    
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # EXTRACT & TRANSFORM
    print("Loading and merging raw data...")
    sales_data = pandas.read_csv(SALES_PATH,
                                usecols=SALES_COLUMN_SELECTION,
                                dtype={'zipcode': str})
    demographics = pandas.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
    
    merged_data = sales_data.merge(demographics, how="left", on="zipcode")

    # FEATURE ENGINEERING
    processed_data = engineer_features(merged_data)
    
    # LOAD
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Successfully saved processed data to {PROCESSED_DATA_PATH}")
    print("--- ETL Finished ---")


if __name__ == "__main__":
    main()
