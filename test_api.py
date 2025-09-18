import requests
import pandas as pd
import os

# --- 1. CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
DATA_PATH = os.path.join("data", "future_unseen_examples.csv")
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
        headers = {
            'X-API-Key': 'default_secret_key',
            'Content-Type': 'application/json'
        }
        print("\n----------------------------------------")
        print(f"Sending data for house #{index+1}:")
        print(f"  Zipcode: {payload.get('zipcode')}, SqFt Living: {payload.get('sqft_living')}")

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status() 

            # --- Display Results ---
            prediction = response.json()
            print(f"✅ SUCCESS: API responded with status {response.status_code}")
            print(f"   => Predicted Price: ${prediction.get('predicted_price'):,.2f}")

        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED: Could not connect to the API at {API_URL}.")
            print(f"   Error: {e}")
            print("   Please ensure the FastAPI server is running.")
            print(response.text if 'response' in locals() else "No response received.")
            break 

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_test()
