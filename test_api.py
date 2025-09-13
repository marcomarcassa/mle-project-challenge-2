import requests
import pandas as pd
import os

# --- 1. CONFIGURATION ---
# The URL where your FastAPI application is running
API_URL = "http://127.0.0.1:8000/predict"
# Path to the data file with examples to test
DATA_PATH = os.path.join("data", "future_unseen_examples.csv")
# Number of examples to test from the file
N_EXAMPLES = 5

# --- 2. SCRIPT LOGIC ---
def run_test():
    """
    Reads data, sends it to the prediction API, and prints the results.
    """
    print("--- Starting API Test ---")

    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
        test_data = df.head(N_EXAMPLES)
        print(f"Loaded {len(test_data)} examples from '{DATA_PATH}'")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'. Make sure the path is correct.")
        return

    # --- Send Requests ---
    for index, row in test_data.iterrows():
        # Convert the row to a dictionary, which is a valid JSON format
        payload = row.to_dict()

        print("\n----------------------------------------")
        print(f"Sending data for house #{index+1}:")
        print(f"  Zipcode: {payload.get('zipcode')}, SqFt Living: {payload.get('sqft_living')}")

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # --- Display Results ---
            prediction = response.json()
            print(f"✅ SUCCESS: API responded with status {response.status_code}")
            print(f"   => Predicted Price: ${prediction.get('predicted_price'):,.2f}")

        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED: Could not connect to the API at {API_URL}.")
            print(f"   Error: {e}")
            print("   Please ensure the FastAPI server is running.")
            break # Stop the test if the server is down

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_test()


### How to Run It

# 1.  Make sure your FastAPI server is **still running** in one terminal (`python -m uvicorn main:app --reload` from the `api` directory).
# 2.  Open a **new, separate terminal**.
# 3.  Activate the same Conda environment: `conda activate housing`.
# 4.  Navigate to the **root directory** of your project (the one that contains the `api/` folder).
# 5.  Run the test script:
#     python test_api.py
    
