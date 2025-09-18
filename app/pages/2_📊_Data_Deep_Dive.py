import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Data Deep Dive", page_icon="📊", layout="wide")

st.title("📊 The Data Deep Dive")
st.markdown("""
Welcome to the exploratory data analysis section. Here, we'll dissect the dataset to uncover trends, relationships, and distributions. 
These insights are fundamental to our feature engineering and modeling strategy.
""")

# --- Data Loading and Caching ---
@st.cache_data
def load_and_process_data():
    """Loads data and creates necessary features for plotting."""
    try:
        df = pd.read_csv("data/kc_house_data.csv")
    except FileNotFoundError:
        st.error("Error: Make sure 'kc_house_data.csv' is in a 'data' subfolder at the project root.")
        return None

    df['date'] = pd.to_datetime(df['date'])
    df['year_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # Create location clusters for consistent analysis across pages
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['lat', 'long']])
    
    return df

df = load_and_process_data()

if df is not None:
    # --- Section 1: Price Distribution ---
    st.header("1. How are house prices distributed?")
    price_view = st.radio(
        "Select Price View",
        ('Normal Price', 'Log-Transformed Price'),
        horizontal=True,
        label_visibility="collapsed"
    )

    if price_view == 'Normal Price':
        fig = px.histogram(df, x='price', nbins=100, title="Distribution of House Prices",
                           labels={'price': 'Sale Price ($)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("The distribution is heavily right-skewed, with a long tail of high-value properties. This is why we use a log transformation for model training.")
    else:
        df['price_log'] = np.log(df['price'])
        fig = px.histogram(df, x='price_log', nbins=100, title="Distribution of Log-Transformed House Prices",
                           labels={'price_log': 'Log of Sale Price'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("After a log transformation, the price distribution is much closer to a normal (bell-shaped) curve, which is ideal for many machine learning models.")

    st.markdown("---")

    # --- Section 2: Key Feature Relationships ---
    st.header("2. How do key features relate to price?")
    
    tab1, tab2, tab3 = st.tabs(["Price vs. Square Feet", "Price vs. Rooms", "Price vs. Location"])

    with tab1:
        st.subheader("Scatter Plot: Price vs. Square Feet")
        fig = px.scatter(df, x='sqft_living', y='price', 
                         title='Price vs. Living Area',
                         labels={'sqft_living': 'Living Area (Square Feet)', 'price': 'Sale Price ($)'},
                         opacity=0.5, trendline='ols', trendline_color_override='red')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("There is a strong, clear positive correlation: as the living area increases, the price tends to increase as well.")

    with tab2:
        st.subheader("Box Plots: Price by Number of Rooms")
        room_choice = st.selectbox("Choose a feature:", ('bedrooms', 'bathrooms'))
        fig = px.box(df, x=room_choice, y='price',
                     title=f'Price Distribution by Number of {room_choice.capitalize()}',
                     labels={room_choice: f'Number of {room_choice.capitalize()}', 'price': 'Sale Price ($)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Box plots show the median price and spread for each category. We can see that the median price generally increases with more bedrooms or bathrooms.")

    with tab3:
        st.subheader("Bar Chart: Average Price by Location Cluster")
        avg_price_by_cluster = df.groupby('location_cluster')['price'].mean().sort_values().reset_index()
        fig = px.bar(avg_price_by_cluster, x='location_cluster', y='price',
                     title='Average Price by Geographic Cluster',
                     labels={'location_cluster': 'Location Cluster ID', 'price': 'Average Sale Price ($)'})
        fig.update_layout(xaxis_type='category')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("This demonstrates the significant impact of location. Some geographic clusters (which we created using K-Means) have a much higher average sale price than others.")

    st.markdown("---")

    # --- Section 3: Time Travel ---
    st.header("3. How have prices changed over time?")
    st.markdown("This line chart shows the median house price for each month in the dataset, highlighting market trends and seasonality.")
    
    median_price_over_time = df.groupby('year_month')['price'].median().reset_index()
    
    fig = px.line(median_price_over_time, x='year_month', y='price',
                  title='Median House Price Over Time',
                  labels={'year_month': 'Month', 'price': 'Median Sale Price ($)'},
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)
