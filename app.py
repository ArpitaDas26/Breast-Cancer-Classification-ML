import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title("Breast Cancer Classification App")

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Upload CSV
uploaded_file = st.file_uploader("Upload test_data.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    predictions = model.predict(X)
    
    df["Predicted"] = predictions
    
    st.write("Prediction Results:")
    st.write(df.head())