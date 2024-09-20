# scripts/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file):
    """Load accident dataset from CSV file."""
    df = pd.read_csv(file)
    return df

def preprocess_data(df):
    """Preprocess accident dataset: handle missing values and normalize."""
    # Fill missing values
    df.fillna(df.median(), inplace=True)
    
    # Normalize numerical columns
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df
