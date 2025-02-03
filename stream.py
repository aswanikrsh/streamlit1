import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Passenger Survival Prediction')

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Make predictions
    prediction = model.predict(data)
    
    # Display the prediction
    st.write("Prediction:", prediction)

