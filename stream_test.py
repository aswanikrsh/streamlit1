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
    try:
        # Only use relevant columns for prediction
        predictions = model.predict(df[['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']])
        return ['Survived' if p == 1 else 'Did not survive' for p in predictions]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return []

# Streamlit UI
st.title('Passenger Survival Prediction')

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Ensure columns are clean (strip spaces)
        df.columns = df.columns.str.strip()

        # Check if required columns are present
        required_columns = ['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']
        if not all(col in df.columns for col in required_columns):
            st.error(f"The uploaded file is missing one or more required columns: {', '.join(required_columns)}")
        else:
            # Handle missing values by filling with default values (can be adjusted)
            df.fillna({'Age': 0, 'Fare_Paid': 0, 'Gender': 'Female', 'Class': 'Third', 'Seat_Type': 'Aisle'}, inplace=True)

            # Encode categorical columns
            df = encode_data(df)

            # Predict and show results (exclude Name and Passenger_ID for prediction)
            df['Prediction'] = predict_survival(df)
            st.write(df[['Name', 'Survival_Status', 'Prediction']])  # Keep Name column for display, not for prediction

    except Exception as e:
        st.error(f"Error processing the file: {e}")



