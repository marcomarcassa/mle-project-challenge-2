import streamlit as st
import pandas as pd
import pydeck as pdk
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Geographic Story", page_icon="🌍", layout="wide")

st.title("🌍 The Geographic Story")
st.markdown("Explore the Seattle housing market through interactive maps. See how prices, model-driven clusters, and demographics vary across the city.")

# --- Data Loading and Caching ---
@st.cache_data
def load_and_prepare_data(sales_path, demographics_path):
    """Loads, merges, and prepares data for visualization."""
    sales_df = pd.read_csv(sales_path, dtype={'zipcode': str})
    demographics_df = pd.read_csv(demographics_path, dtype={'zipcode': str})

    df = pd.merge(sales_df, demographics_df, on='zipcode', how='left')

    df.rename(columns={
        'medn_hshld_incm_amt': 'median_household_income',
        'ppltn_qty': 'population_density'
    }, inplace=True)

    df = df.dropna(subset=['lat', 'long', 'price', 'median_household_income', 'population_density'])

    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['lat', 'long']])

    # Normalize demographic data for better color mapping
    scaler = MinMaxScaler()
    df[['median_household_income_norm', 'population_density_norm']] = scaler.fit_transform(
        df[['median_household_income', 'population_density']]
    )

    return df

try:
    data = load_and_prepare_data("data/kc_house_data.csv", "data/zipcode_demographics.csv")
except FileNotFoundError:
    st.error("Error: Make sure 'kc_house_data.csv' and 'zipcode_demographics.csv' are in a 'data' subfolder at the project root.")
    st.stop()
except KeyError:
    st.error("A critical column was not found after renaming. Please check the schema in the app against your CSV files.")
    st.stop()


# --- Map Visualizations in Tabs ---
#tab1, tab2, tab3 = st.tabs(["Price Heatmap", "Location Cluster Explorer", "Demographics Overlay"])
tab1, tab2= st.tabs(["Price Heatmap", "Location Cluster Explorer"])

# --- Tab 1: Price Heatmap ---
with tab1:
    st.header("Price Heatmap")
    st.markdown("This map shows the concentration of house prices across Seattle. Warmer colors indicate higher prices.")

    min_price, max_price = st.slider(
        "Select Price Range ($)",
        int(data['price'].min()),
        int(data['price'].max()),
        (int(data['price'].min()), int(data['price'].quantile(0.75)))
    )
    filtered_data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_data,
        get_position=["long", "lat"],
        opacity=0.8,
        get_weight="price",
        threshold=0.05,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=47.6062, longitude=-122.3321, zoom=9, pitch=50),
        layers=[heatmap_layer],
        tooltip={"text": "Price: ${price}\nBedrooms: {bedrooms}"}
    ))

# --- Tab 2: Location Cluster Explorer ---
with tab2:
    st.header("Location Cluster Explorer")
    st.markdown("This map visualizes the 10 distinct geographic clusters identified by a K-Means algorithm. This is a key feature used by our prediction model.")

    cluster_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
        [255, 0, 255], [128, 0, 128], [255, 165, 0], [0, 128, 0], [75, 0, 130]
    ]
    data['color'] = data['location_cluster'].apply(lambda x: cluster_colors[x % len(cluster_colors)])

    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["long", "lat"],
        get_fill_color="color",
        get_radius=100,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=47.6062, longitude=-122.3321, zoom=9, pitch=50),
        layers=[cluster_layer],
        tooltip={"text": "Price: ${price}\nCluster ID: {location_cluster}"}
    ))


## --- Tab 3: Demographics Overlay ---
#with tab3:
#    st.header("Demographics Overlay")
#    st.markdown("Visualize demographic data aggregated by area. The height and color of the columns represent the intensity of the selected metric.")
#
#    demographic_choice = st.radio(
#        "Choose a demographic to display:",
#        ('Median Household Income', 'Population Density'),
#        horizontal=True
#    )
#
#    column_map = {
#        'Median Household Income': 'median_household_income',
#        'Population Density': 'population_density'
#    }
#    norm_column_map = {
#        'Median Household Income': 'median_household_income_norm',
#        'Population Density': 'population_density_norm'
#    }
#    selected_column = column_map[demographic_choice]
#    selected_norm_column = norm_column_map[demographic_choice]
#
#
#    hexagon_layer = pdk.Layer(
#        "HexagonLayer",
#        data=data,
#        get_position=["long", "lat"],
#        get_elevation=selected_column,
#        get_fill_color=f"[255 * (1 - {selected_norm_column}), 255 * {selected_norm_column}, 50, 140]",
#        elevation_scale=100,
#        extruded=True,
#        radius=200,
#        pickable=True,
#    )
#    st.pydeck_chart(pdk.Deck(
#        map_style=None,
#        initial_view_state=pdk.ViewState(latitude=47.6062, longitude=-122.3321, zoom=9, pitch=50),
#        layers=[hexagon_layer],
#        tooltip={"html": f"<b>{demographic_choice}:</b> {{{selected_column}}}"}
#    ))
#
#