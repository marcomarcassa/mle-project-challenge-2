import json
import os
import pathlib
import argparse
import pickle  # Added for saving the model
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas
import mlflow
import lightgbm as lgb
import xgboost as xgb
from sklearn import model_selection, pipeline, preprocessing, neighbors
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ydata_profiling import ProfileReport

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# --- Suppress Warnings ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# --- CONFIGURATION ---
OUTPUT_DIR = "model"
RAW_DATA_PATH = "data/kc_house_data.csv"
PROCESSED_DATA_PATH = "processed_data/processed_kc_house_data.csv"
MODEL_PKL_DIR = "model" 


def load_raw_data(path: str) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the raw data and split into features and target."""
    df = pandas.read_csv(path)
    y = df.pop('price')
    x = df
    return x, y

def load_processed_data(path: str) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the processed data and split into features and target."""
    df = pandas.read_csv(path)
    y = df.pop('price')
    x = df
    return x, y


def get_model(model_name: str) -> Tuple[pipeline.Pipeline, Dict[str, Any]]:
    """
    Return a scikit-learn model pipeline and its hyperparameter grid.
    All models are wrapped in a pipeline for consistency.
    """
    if model_name == "knn":
        model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            neighbors.KNeighborsRegressor()
        )
        param_grid = {'kneighborsregressor__n_neighbors': [5, 10, 15]}

    elif model_name == "lgbm":
        model = pipeline.make_pipeline(
            lgb.LGBMRegressor(random_state=42, verbose=-1)
        )
        param_grid = {
            'lgbmregressor__n_estimators': [100, 200, 300],
            'lgbmregressor__learning_rate': [0.05, 0.1],
            'lgbmregressor__num_leaves': [20, 31, 40]
        }

    elif model_name == "xgb":
        model = pipeline.make_pipeline(
            xgb.XGBRegressor(random_state=42)
        )
        param_grid = {
            'xgbregressor__n_estimators': [100, 200, 300],
            'xgbregressor__learning_rate': [0.05, 0.1],
            'xgbregressor__max_depth': [3, 5, 7]
        }

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return model, param_grid

def generate_and_log_data_profile(
    x_train: pandas.DataFrame, y_train: pandas.Series, data_state: str
):
    """Generates processed data profile of the training set"""
    print("Generating processed data profile...")
    
    train_df = pandas.concat([x_train, y_train], axis=1)
    
    profile = ProfileReport(
        train_df,
        title=f"{data_state} Training Data Profile",
        minimal=True  # faster report generation
    )
    
    profile_path = pathlib.Path(OUTPUT_DIR) / f"{data_state}_training_data_profile.html"
    profile.to_file(profile_path)
    
    mlflow.log_artifact(str(profile_path), "data_profile")
    print("Data profile logged as MLflow artifact.")


def main(model_name: str):
    """Load processed data, train a specified model, and log with MLflow."""
    
    x_raw, y_raw = load_raw_data(RAW_DATA_PATH)
    x_raw_train, x_raw_test, y_raw_train, y_raw_test = model_selection.train_test_split(x_raw, y_raw, random_state=42)

    x, y = load_processed_data(PROCESSED_DATA_PATH)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=42)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # --- Create directory for .pkl files ---
    model_pkl_dir = pathlib.Path(OUTPUT_DIR)
    model_pkl_dir.mkdir(exist_ok=True)
    
    mlflow.set_experiment("Seattle House Price Prediction")

    with mlflow.start_run(run_name=f"{model_name}_run"):
        # --- Log the raw data profile ---
        generate_and_log_data_profile(x_raw_train, y_raw_train, data_state="Raw")
        # --- Log the processed data profile ---
        generate_and_log_data_profile(x_train, y_train, data_state="Processed")
        y_train_log = np.log1p(y_train)
        
        model_pipeline, param_grid = get_model(model_name)
        
        grid_search = model_selection.GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            n_jobs=-1
        ).fit(x_train, y_train_log)

        best_model = grid_search.best_estimator_
        
        # --- Save the best model to a .pkl file ---
        model_path = model_pkl_dir / f"model_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Model saved to: {model_path}")
        # ---------------------------------------------

        y_pred_log_test = best_model.predict(x_test)
        y_pred_test = np.expm1(y_pred_log_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)

        print(f"--- {model_name.upper()} Test Set Metrics ---")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R^2:  {r2:.4f}")
        print(f"MAE:  ${mae:,.2f}")

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=x_train.head()
        )

        features_path = output_dir / "model_features.json"
        with open(features_path, "w") as f:
            json.dump(list(x_train.columns), f)
        mlflow.log_artifact(str(features_path), "features")
        print("Run logged successfully to MLflow.")

        test_data = x_test.copy()
        test_data['actual_price'] = y_test
        test_data['predicted_price'] = y_pred_test
        
        test_data_path = pathlib.Path("model/test_data.csv")
        test_data.to_csv(test_data_path, index=False)
        print(f"Test data saved to {test_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["knn", "lgbm", "xgb"],
        help="The model to train and evaluate."
    )
    args = parser.parse_args()
    main(model_name=args.model)