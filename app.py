# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from scripts.preprocessing import load_data, preprocess_data
from scripts.kmeans_analysis import apply_kmeans, evaluate_kmeans
from scripts.apriori_analysis import apply_apriori, get_strong_rules

# Title of the app
st.title("Accident Data Analysis with K-Means and Apriori")

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

    # Map Visualization with Folium
    st.write("## Map Visualization")
    accident_map = folium.Map(location=[13.35, 123.5], zoom_start=12)
    
    # Plot clusters on the map
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=['red', 'blue', 'green', 'orange', 'purple'][row['cluster']],
            fill=True
        ).add_to(accident_map)
    
    folium_static(accident_map)
