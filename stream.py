import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('Passenger Survival Prediction')

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Ensure required columns
    required_cols = ['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']
    if not all(col in data.columns for col in required_cols):
        st.error("CSV file must contain these columns: Age, Gender, Class, Seat_Type, Fare_Paid")
    else:
        # Initialize LabelEncoder
        encoder = LabelEncoder()
        
        # Encode categorical columns
        data['Gender'] = encoder.fit_transform(data['Gender'])
        data['Class'] = encoder.fit_transform(data['Class'])
        data['Seat_Type'] = encoder.fit_transform(data['Seat_Type'])
        
        # Select required features
        data = data[required_cols]

        # Make predictions
        prediction = model.predict(data)
        
        # Display the predictions
        st.write("Prediction (0 = Not Survived, 1 = Survived):", prediction)



