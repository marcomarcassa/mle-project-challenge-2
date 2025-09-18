import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans

st.set_page_config(page_title="Interactive Price Predictor", page_icon="🔮", layout="wide")

# --- Page Title and Introduction ---
st.title("🔮 Interactive Price Predictor")
st.markdown("""
This is the centerpiece of our app! Use the sidebar on the left to input the features of a house. 
The model will instantly predict its estimated market value. This is a great way to perform "what-if" analysis.
""")

# --- Model Training and Data Preparation (Cached) ---
@st.cache_resource
def get_model_and_data_info():
    """
    Loads data, engineers features, trains a model, and returns the model
    along with information about the data (for input widgets).
    """
    # Load data
    df = pd.read_csv("data/kc_house_data.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Feature Engineering (consistent with page 3)
    df['year_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    df['house_age'] = df['year_sold'] - df['yr_built']
    df['was_renovated'] = (df['yr_renovated'] != 0).astype(int)
    df.loc[df['was_renovated'] == 1, 'house_age'] = df['year_sold'] - df['yr_renovated']
    df['price_log'] = np.log(df['price'])

    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['lat', 'long']])

    # Define features and target
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
        'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'lat', 'long',
        'house_age', 'was_renovated', 'location_cluster'
    ]
    target = 'price_log'

    X = df[features]
    y_log = df[target]

    # Train Model
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X, y_log)

    # Get data ranges for input widgets
    data_info = {
        # --- Values from the first image ---
        'bedrooms': (int(df['bedrooms'].min()), int(df['bedrooms'].max()), 4),
        'bathrooms': (float(df['bathrooms'].min()), float(df['bathrooms'].max()), 2.0),
        'sqft_living': (int(df['sqft_living'].min()), int(df['sqft_living'].max()), 1910),
        'grade': sorted(df['grade'].unique()), # Default: 8
        'house_age': (int(df['house_age'].min()), int(df['house_age'].max()), 30), # Default updated
        'location_cluster': sorted(df['location_cluster'].unique()), # Default: 0
        'was_renovated': [0, 1], # Default: Yes (1)
        'sqft_lot': (int(df['sqft_lot'].min()), int(df['sqft_lot'].max()), 7618), # Default updated
        'floors': (float(df['floors'].min()), float(df['floors'].max()), 2.0), # Default updated
        'waterfront': [0, 1], # Default: Yes (1)
        'view': sorted(df['view'].unique()), # Default: 0
        'condition': sorted(df['condition'].unique()), # Default: 3
        'sqft_above': (int(df['sqft_above'].min()), int(df['sqft_above'].max()), 1560), # Default updated
        'sqft_basement': (int(df['sqft_basement'].min()), int(df['sqft_basement'].max()), 0), # Default updated
        'lat': (float(df['lat'].min()), float(df['lat'].max()), 47.5601), # Default updated
        'long': (float(df['long'].min()), float(df['long'].max()), -122.2139), # Default updated
    }

    return model, features, data_info

# --- Load Model and Data Info ---
try:
    model, features, data_info = get_model_and_data_info()
except FileNotFoundError:
    st.error("Error: Make sure 'kc_house_data.csv' is in a 'data' subfolder at the project root.")
    st.stop()


# --- Sidebar for User Inputs ---
st.sidebar.header("Input House Features")

def get_user_inputs():
    """Create widgets in the sidebar and return user inputs as a dictionary."""
    inputs = {}
    inputs['bedrooms'] = st.sidebar.slider("Bedrooms", *data_info['bedrooms'])
    inputs['bathrooms'] = st.sidebar.slider("Bathrooms", *data_info['bathrooms'], step=0.25)
    inputs['sqft_living'] = st.sidebar.slider("Square Feet (Living)", *data_info['sqft_living'])
    inputs['grade'] = st.sidebar.select_slider("Grade (Quality)", options=data_info['grade'], value=8)
    inputs['house_age'] = st.sidebar.slider("House Age (Years)", *data_info['house_age'])
    inputs['location_cluster'] = st.sidebar.selectbox("Location Cluster", options=data_info['location_cluster'])
    inputs['was_renovated'] = st.sidebar.radio("Was Renovated?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    
    with st.sidebar.expander("More Features"):
        inputs['sqft_lot'] = st.number_input("Square Feet (Lot)", *data_info['sqft_lot'])
        inputs['floors'] = st.slider("Floors", *data_info['floors'], step=0.5)
        inputs['waterfront'] = st.radio("Is Waterfront?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        inputs['view'] = st.selectbox("View Quality", options=data_info['view'])
        inputs['condition'] = st.select_slider("Condition", options=data_info['condition'], value=3)
        inputs['sqft_above'] = st.number_input("Square Feet (Above Ground)", *data_info['sqft_above'])
        inputs['sqft_basement'] = st.number_input("Square Feet (Basement)", *data_info['sqft_basement'])
        inputs['lat'] = st.number_input("Latitude", *data_info['lat'], format="%.4f")
        inputs['long'] = st.number_input("Longitude", *data_info['long'], format="%.4f")

    return inputs

user_inputs = get_user_inputs()

# --- Prediction and Display ---
input_df = pd.DataFrame([user_inputs])
input_df = input_df[features]

# Get prediction
prediction_log = model.predict(input_df)
prediction = np.exp(prediction_log[0])

# Display the prediction in a prominent way
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h3 style='text-align: center;'>Predicted House Price</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #28a745;'>${prediction:,.0f}</h1>", unsafe_allow_html=True)

# --- Display User Inputs for Context ---
st.markdown("---")
st.subheader("Inputs Used for This Prediction")
st.dataframe(input_df)
