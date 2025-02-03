import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess the dataset
df = pd.read_csv('passenger_survival_dataset.csv')

# Create the LabelEncoder object
encoder = LabelEncoder()

# Encoding the 'Gender', 'Class', and 'Seat_Type' columns
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Class'] = encoder.fit_transform(df['Class'])
df['Seat_Type'] = encoder.fit_transform(df['Seat_Type'])

# Define the features (X) and the target (y)
X = df[['Age', 'Gender', 'Class', 'Seat_Type', 'Fare_Paid']]
y = df['Survival_Status']

# Function to make predictions
def predict_survival(age, gender, passenger_class, seat_type, fare_paid):
    # Encode the categorical variables manually (same encoding as in training)
    gender_encoded = 1 if gender.lower() == 'male' else 0
    class_encoded = {'First': 0, 'Second': 1, 'Third': 2}.get(passenger_class, 2)  # Default to Third
    seat_encoded = {'Window': 0, 'Middle': 1, 'Aisle': 2}.get(seat_type, 2)  # Default to Aisle

    # Prepare the input data for prediction (it needs to be a 2D array)
    input_data = np.array([[age, gender_encoded, class_encoded, seat_encoded, fare_paid]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    # Return the result
    return 'Survived' if prediction == 1 else 'Did not survive'

# Streamlit UI
st.title('Passenger Survival Prediction')
st.write("Enter the details below to predict whether the passenger survived.")

# Input fields for user to enter passenger details
age = st.number_input('Age', min_value=0, max_value=100, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
passenger_class = st.selectbox('Class', ['First', 'Second', 'Third'])
seat_type = st.selectbox('Seat Type', ['Window', 'Middle', 'Aisle'])
fare_paid = st.number_input('Fare Paid', min_value=0.0, value=10.0)

# Predict and show result when the user clicks the 'Predict' button
if st.button('Predict'):
    result = predict_survival(age, gender, passenger_class, seat_type, fare_paid)
    st.write(f"The passenger is predicted to have: {result}")

