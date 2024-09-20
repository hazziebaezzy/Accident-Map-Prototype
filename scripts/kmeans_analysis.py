# scripts/kmeans_analysis.py
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def apply_kmeans(df, n_clusters=5):
    """Apply K-Means to the accident dataset."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df.select_dtypes(include=[np.number]))
    return df, kmeans

def evaluate_kmeans(kmeans, df):
    """Evaluate K-means performance with inertia and silhouette score."""
    inertia = kmeans.inertia_
    return {'inertia': inertia}
