import streamlit as st

st.set_page_config(
    page_title="Seattle House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Seattle House Price Prediction")

st.subheader("Project Pages")

col1, col2 = st.columns(2)

with col1:

    st.page_link("pages/0_🚀_Model_training_hub.py", label="**0 Model Training Hub**", icon="🚀")
    st.write("The project's control center. Trigger new model training runs, view detailed data quality profiles, and access the MLflow tracking server to manage experiments.")

    st.page_link("pages/2_📊_Data_Deep_Dive.py", label="**2. Data Deep Dive**", icon="📊")
    st.write("Explore the relationships between house features and price.")

    st.page_link("pages/4_🔮_Price_Predictor.py", label="**4. Interactive Price Predictor**", icon="🔮")
    st.write("Get a live price estimate for a house with custom features.")



with col2:
    st.page_link("pages/1_🌍_Geographic_Story.py", label="**1. Geographic Story**", icon="🌍")
    st.write("Visualize house prices, model-driven location clusters, and demographic data on an interactive map of Seattle.")

    st.page_link("pages/3_🧠_Model's_Mind.py", label="**3. Inside the Model's Mind**", icon="🧠")
    st.write("See which features drive predictions and how the model performs.")
    