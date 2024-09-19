import streamlit as st
import pandas as pd

st.title("My Streamlit Data App")

# Upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded
if uploaded_file:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write("Dataframe:")
    st.dataframe(df)
    
    # Show descriptive statistics
    st.write("Summary Statistics:")
    st.write(df.describe())
