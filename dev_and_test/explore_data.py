import pandas as pd
from ydata_profiling import ProfileReport
import os

# --- Configuration ---
DATA_DIR = "../data"
INPUT_DATA_PATH = os.path.join(DATA_DIR, "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, "zipcode_demographics.csv")
OUTPUT_REPORT_PATH = "data_profiling_report.html"

# --- Main Script ---
def generate_report():
    """
    Loads, merges, and profiles the housing data, then saves an HTML report.
    """
    print("Loading data...")
    # Load the primary and demographic datasets
    df_houses = pd.read_csv(INPUT_DATA_PATH)
    df_demographics = pd.read_csv(DEMOGRAPHICS_PATH)

    # Merge them to get a complete picture
    df_full = pd.merge(df_houses, df_demographics, on='zipcode', how='left')
    print("Data loaded and merged successfully.")

    print(f"Generating profiling report for {len(df_full.columns)} columns...")
    # Generate the report
    profile = ProfileReport(
        df_full,
        title="Seattle Housing Data Profile",
        explorative=True # Enables more in-depth analysis
    )

    print(f"Saving report to '{OUTPUT_REPORT_PATH}'...")
    # Save the report to an interactive HTML file
    profile.to_file(OUTPUT_REPORT_PATH)
    print("Report saved successfully. Open the HTML file in your browser.")

if __name__ == "__main__":
    generate_report()
