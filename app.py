import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from scripts.preprocessing import load_data, preprocess_data
from scripts.kmeans_analysis import apply_kmeans, evaluate_kmeans
from scripts.apriori_analysis import apply_apriori, get_strong_rules
import streamlit.components.v1 as components

# Title of the app
st.title("Accident Data Analysis with QGIS Map Integration")

# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Upload Data", "QGIS Map"])

if page == "Upload Data":
    st.write("## Upload Accident Data CSV")

    # CSV File Upload
    uploaded_file = st.file_uploader("Upload Accident Data CSV", type="csv")
    if uploaded_file:
        # Load and preprocess the data
        df = load_data(uploaded_file)
        df = preprocess_data(df)

        # Display dataset
        st.write("### Uploaded Data", df.head())

        # K-Means Clustering
        st.write("## K-Means Clustering")
        n_clusters = st.slider("Select number of clusters", 2, 10, 5)
        df, kmeans_model = apply_kmeans(df, n_clusters=n_clusters)
        st.write("Clustered Data", df.head())

        # Display K-Means Evaluation
        kmeans_eval = evaluate_kmeans(kmeans_model, df)
        st.write(f"Inertia: {kmeans_eval['inertia']}")

        # Apriori Algorithm
        st.write("## Apriori Algorithm")
        apriori_rules = apply_apriori(df)
        strong_rules = get_strong_rules(apriori_rules)
        st.write("Strong Association Rules", strong_rules)

elif page == "QGIS Map":
    st.write("## QGIS Map Integration")
    
    # Sample QGIS Map visualization using Folium (assuming data is pre-exported to GeoJSON)
    # You can replace the GeoJSON path with your own exported map data or WMS link
    st.write("Below is the embedded QGIS Map:")
    components.iframe("http://localhost:8501//workspaces/Accident-Map-Prototype/qgis/index.html", width=1000, height=600)
