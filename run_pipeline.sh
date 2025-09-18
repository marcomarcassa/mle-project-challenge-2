#!/bin/bash

# 1: Process the data 
echo "--- Running Data Processing ETL ---"
python process_data.py

# Check if the first script succeeded
if [ $? -ne 0 ]; then
  echo "Data processing failed. Aborting pipeline."
  exit 1
fi

# 2: Run Model Benchmarks
echo "--- Running LightGBM Benchmark ---"
python create_model_v2.py --model lgbm

echo "--- Running XGBoost Benchmark ---"
python create_model_v2.py --model xgb

echo "--- Running KNN Benchmark ---"
python create_model_v2.py --model knn

echo "--- Pipeline Finished ---"
