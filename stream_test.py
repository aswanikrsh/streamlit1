import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Encode categorical columns function
def encode_data(df):
    encoder = LabelEncoder()
    df[['Gender', 'Class', 'Seat_Type']] = df[['Gender', 'Class', 'Seat_Type']].apply(encoder.fit_transform)
    return df

# Predict survival function
def predict_survival(df):
    predictions = model.predict(df[['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']])
    return ['Survived' if p == 1 else 'Did not survive' for p in predictions]

# Streamlit UI
st.title('Passenger Survival Prediction')

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    # Clean and encode data
    df.columns = df.columns.str.strip()
    df = encode_data(df)
    
    # Predict and show results
    df['Prediction'] = predict_survival(df)
    st.write(df[['Name', 'Survival_Status', 'Prediction']])


