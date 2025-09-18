import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import shap
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(page_title="Model Interpretation", page_icon="🧠", layout="wide")

# --- Helper function to display SHAP plots ---
def st_shap(plot, height=None):
    """
    Renders a SHAP plot object in Streamlit using the components.html function.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# --- Page Title and Introduction ---
st.title("🧠 Inside the Model's Mind")
st.markdown("""
This page offers a look under the hood of our price prediction model.
By understanding which features are most important and how they influence individual predictions, we can build trust in the model's outputs.
""")

st.sidebar.header("📁 Model Selection")
model_dir = "model/"

# Check if the model directory exists
if not os.path.isdir(model_dir):
    st.error(f"The directory '{model_dir}' was not found. Please create it and add your model files.")
    st.stop()

# Find all .pkl files in the directory
try:
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model") and f.endswith(".pkl") and not f.endswith("knn.pkl")]
    if not model_files:
        st.sidebar.error(f"No '.pkl' model files found in the '{model_dir}' directory.")
        st.stop()

    # Create a dropdown for the user to select a model
    selected_model_file = st.sidebar.selectbox("Choose a model file to analyze:", model_files)
    selected_model_path = os.path.join(model_dir, selected_model_file)
    st.sidebar.info(f"**Selected Model:** `{selected_model_file}`")

except Exception as e:
    st.error(f"An error occurred while scanning for model files: {e}")
    st.stop()

@st.cache_data
def load_artifacts_and_predict(model_path):
    """
    Loads a pre-trained model, feature names, and test data.
    It then generates predictions on the test data.
    """
    # Load the user-selected pre-trained model and the feature list
    model = joblib.load(model_path)
    with open("model/model_features.json", 'r') as f:
        feature_names = json.load(f)

    # Load the test data
    test_df = pd.read_csv("model/test_data.csv")

    # Separate features (X_test) and the actual target variable (y_test)
    X_test = test_df[feature_names]
    y_test_actual = test_df['actual_price']

    # Generate predictions (the model outputs log-transformed prices)
    y_pred_log = model.predict(X_test)
    y_pred_actual = np.exp(y_pred_log)

    return model, X_test, y_test_actual, y_pred_actual, feature_names


# --- Load Data and Artifacts based on selection ---
with st.spinner(f"Loading `{selected_model_file}` and data... (This runs once per model!)"):
    try:
        model, X_test, y_test, y_pred, feature_names = load_artifacts_and_predict(selected_model_path)
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure 'model/model_features.json', and 'model/test_data.csv' exist.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model or processing data: {e}")
        st.stop()


# --- Extract the core model from the pipeline for SHAP and Feature Importance ---
model_for_explainer = model.steps[-1][1] if isinstance(model, Pipeline) else model


# --- Visualization Tabs ---
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Actual vs. Predicted", "Prediction Breakdown (SHAP)"])

# --- Tab 1: Feature Importance ---
with tab1:
    st.header("Which Features Matter Most?")
    st.markdown("This chart shows the features that have the biggest impact on the model's predictions, ranked by importance.")

    try:
        if isinstance(model_for_explainer, lgb.LGBMRegressor):
            importance_values = model_for_explainer.booster_.feature_importance()
        elif isinstance(model_for_explainer, xgb.XGBRegressor):
            importance_values = model_for_explainer.feature_importances_
        else:
            st.warning(f"Model type `{type(model_for_explainer).__name__}` may not be supported for automated feature importance. Attempting to use a generic attribute.")
            if hasattr(model_for_explainer, 'feature_importances_'):
                importance_values = model_for_explainer.feature_importances_
            else:
                st.error(f"Could not determine feature importance for model type {type(model_for_explainer).__name__}.")
                importance_values = np.zeros(len(feature_names)) 

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance for: {selected_model_file}',
            labels={'Importance': 'Importance Score', 'Feature': 'House Feature'},
            height=800
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate feature importance plot. Error: {e}")


# --- Tab 2: Actual vs. Predicted ---
with tab2:
    st.header("How Accurate Are the Predictions?")
    st.markdown("This scatter plot compares the model's predicted prices against the actual sale prices for the test dataset.")

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE as % of Avg Price", f"{(mae / y_test.mean() * 100):.2f}%")
    col3.metric("RMSE as % of Avg Price", f"{(rmse / y_test.mean() * 100):.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        marker=dict(color='rgba(66, 134, 244, 0.6)', size=5),
        name='Predictions'
    ))
    fig.add_shape(type="line",
        x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", width=2, dash="dash")
    )
    fig.update_layout(
        title='Actual vs. Predicted Prices',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        showlegend=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Prediction Breakdown (SHAP) ---
with tab3:
    st.header("Why Did the Model Predict a Certain Price?")
    st.markdown("""
    This section uses SHAP (SHapley Additive exPlanations) to break down a single prediction.
    - The **base value** is the average prediction across all houses.
    - **Red bars** represent features that pushed the prediction **higher**.
    - **Blue bars** represent features that pushed the prediction **lower**.
    The final prediction is the sum of the base value and the impact of all features.
    """)

    with st.spinner("Calculating SHAP values... (This can take a moment)"):
        try:
            explainer = shap.TreeExplainer(model_for_explainer)
            shap_values = explainer.shap_values(X_test)

        except Exception as e:
            st.error(f"Could not compute SHAP values for this model. It might not be a tree-based model. Error: {e}")
            st.stop()

    selected_index = st.slider("Select a house from the test set to explain:", 0, len(X_test)-1, 10)

    st.subheader(f"Explanation for House #{selected_index}")
    st.write(f"**Actual Price:** `${y_test.iloc[selected_index]:,.0f}`")
    st.write(f"**Predicted Price:** `${y_pred[selected_index]:,.0f}`")

    st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index,:], X_test.iloc[selected_index,:]), 400)

    st.subheader("Raw Feature Values for Selected House")
    st.dataframe(X_test.iloc[[selected_index]])
