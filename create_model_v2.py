import json
import os
import pathlib
import argparse
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas
import mlflow
import lightgbm as lgb
import xgboost as xgb
from sklearn import model_selection, pipeline, preprocessing, neighbors
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Suppress Warnings ---
# This addresses a warning about conflicting OpenMP libraries in the environment.
# It's a common issue in Conda environments, especially on Windows.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# --- CONFIGURATION ---
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode', 'date', 'yr_built',
    'yr_renovated', 'lat', 'long'
]
OUTPUT_DIR = "model_artifacts"


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics."""
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def engineer_features(
    x_train: pandas.DataFrame, x_test: pandas.DataFrame
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Apply feature engineering steps to the training and test data."""
    x_train_eng = x_train.copy()
    x_test_eng = x_test.copy()

    for df in [x_train_eng, x_test_eng]:
        df['date'] = pandas.to_datetime(df['date'])
        df['year_sold'] = df['date'].dt.year
        df['month_sold'] = df['date'].dt.month
        df['house_age'] = df['year_sold'] - df['yr_built']
        df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
        df['has_basement'] = (df['sqft_basement'] > 0).astype(int)

    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    kmeans.fit(x_train[['lat', 'long']])
    x_train_eng['location_cluster'] = kmeans.predict(x_train[['lat', 'long']])
    x_test_eng['location_cluster'] = kmeans.predict(x_test[['lat', 'long']])

    cols_to_drop = ['date', 'yr_built', 'yr_renovated', 'lat', 'long']
    x_train_eng = x_train_eng.drop(columns=cols_to_drop)
    x_test_eng = x_test_eng.drop(columns=cols_to_drop)

    return x_train_eng, x_test_eng


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
        # For pipelines, param names are prefixed with the step name (lowercase class name)
        param_grid = {'kneighborsregressor__n_neighbors': [5, 10, 15]}

    elif model_name == "lgbm":
        model = pipeline.make_pipeline(
            lgb.LGBMRegressor(random_state=42, verbose=-1)
        )
        # The step name is 'lgbmregressor'
        param_grid = {
            'lgbmregressor__n_estimators': [100, 200, 300],
            'lgbmregressor__learning_rate': [0.03, 0.05, 0.1],
            'lgbmregressor__num_leaves': [20, 31, 40]
        }

    elif model_name == "xgb":
        model = pipeline.make_pipeline(
            xgb.XGBRegressor(random_state=42)
        )
        # The step name is 'xgbregressor'
        param_grid = {
            'xgbregressor__n_estimators': [100, 200, 300],
            'xgbregressor__learning_rate': [0.03, 0.05, 0.1],
            'xgbregressor__max_depth': [3, 5, 7]
        }

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return model, param_grid


def main(model_name: str):
    """Load data, engineer features, train a specified model, and log with MLflow."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=42)

    y_train_log = np.log1p(y_train)
    x_train_eng, x_test_eng = engineer_features(x_train, x_test)

    mlflow.set_experiment("Seattle House Price Prediction")

    with mlflow.start_run(run_name=f"{model_name}_run"):  # Give runs a descriptive name
        model_pipeline, param_grid = get_model(model_name)

        grid_search = model_selection.GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            n_jobs=-1
        ).fit(x_train_eng, y_train_log)

        best_model = grid_search.best_estimator_

        y_pred_log_test = best_model.predict(x_test_eng)
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
            input_example=x_train_eng.head()
        )

        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        features_path = output_dir / "model_features.json"
        with open(features_path, "w") as f:
            json.dump(list(x_train_eng.columns), f)
        mlflow.log_artifact(str(features_path), "features")
        print("Run logged successfully to MLflow.")


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


"""

### How to Run More Robust Benchmarks

Now you have a powerful, repeatable system.

**Step 1: Install New Libraries**
Run this in your `housing` environment's terminal:
```sh
pip install lightgbm xgboost
```

**Step 2: Run an Experiment for Each Model**
Execute these commands one after the other in your terminal. Each one will create a new, separate run in your MLflow experiment.

```sh
# Run the LightGBM model
python main.py --model lgbm

# Run the XGBoost model
python main.py --model xgb

# Rerun the KNN baseline for comparison
python main.py --model knn

"""