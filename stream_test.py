import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title('Passenger Survival Prediction')

# File uploader to upload a CSV file
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# If a file is uploaded
if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the uploaded data
    st.write("Uploaded Data", data.head())
    
    # Ensure required columns exist
    required_columns = ['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']
    
    if all(col in data.columns for col in required_columns):
        # Load LabelEncoder used during model training
        encoder = LabelEncoder()
        
        # Encode categorical columns
        data['Gender'] = encoder.fit_transform(data['Gender'])
        data['Class'] = encoder.fit_transform(data['Class'])
        data['Seat_Type'] = encoder.fit_transform(data['Seat_Type'])
        
        # Prepare the feature data for prediction
        X = data[['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']]
        
        # Make predictions
        prediction = model.predict(X)
        
        # Display predictions (0 = Not Survived, 1 = Survived)
        st.write("Predictions (0 = Not Survived, 1 = Survived):", prediction)
    else:
        st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
