import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import gdown

# Google Drive URLs
model_url = 'https://drive.google.com/uc?id=1u1bJnAZzxWBHl5537tcTa9-5qqSP-na8'
scaler_url = 'https://drive.google.com/uc?id=1XNJRbbyqv9tvhwf04GCvn-fFTGzQsMWY'
encoder_url = 'https://drive.google.com/uc?id=1k6X05we1Q2KvTQKo4qZJISn1vi_WE9G4'

# Download files from Google Drive
def download_file_from_google_drive(url, output):
    gdown.download(url, output, quiet=False)

download_file_from_google_drive(model_url, 'xgb_model.pkl')
download_file_from_google_drive(scaler_url, 'scaler.pkl')
download_file_from_google_drive(encoder_url, 'encoder.pkl')

# Load pre-trained preprocessing objects
def load_preprocessing_objects(scaler_path, encoder_path):
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        with open(encoder_path, 'rb') as file:
            encoder_info = pickle.load(file)
        return scaler, encoder_info
    except Exception as e:
        st.error(f"Error loading preprocessing objects: {e}")
        return None, None

# Load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Path to the model and preprocessing objects
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'
encoder_path = 'encoder.pkl'

# Load the model and preprocessing objects
model = load_model(model_path)
scaler, encoder_info = load_preprocessing_objects(scaler_path, encoder_path)

# Title of the app
st.title("Insurance Data Prediction")

# Check if the model and preprocessing objects were successfully loaded
if model and scaler and encoder_info:
    # Input fields for the features
    gender = st.selectbox("Gender", ["female", "male"])
    age = st.slider("Age", 18, 100, 30)
    driving_license = st.selectbox("Driving License", ["No", "Yes"])
    region_code = st.slider("Region Code", 0, 50, 1)
    previously_insured = st.selectbox("Previously Insured", ["No", "Yes"])
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 year", "1-2 years", "> 2 years"])
    vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"])
    annual_premium = st.number_input("Annual Premium", min_value=0, max_value=100000, value=10000)
    policy_sales_channel = st.slider("Sales Channel", 0, 200, 1)
    vintage = st.slider("Vintage", 0, 300, 1)

    # Prepare the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Driving_License': [1 if driving_license == "Yes" else 0],
        'Region_code': [region_code],
        'Previously_insured': [1 if previously_insured == "Yes" else 0],
        'Annual_Premium': [annual_premium],
        'Policy_Sales_Channel': [policy_sales_channel],
        'Vintage': [vintage],
        'Gender_Male': [1 if gender == "male" else 0],
        'Vehicle_Age_< 1 Year': [1 if vehicle_age == "< 1 year" else 0],
        'Vehicle_Age_> 2 Year': [1 if vehicle_age == "> 2 years" else 0],
        'Vehicle_Damage_Yes': [1 if vehicle_damage == "Yes" else 0]
    })

    # Ensure the columns match those used during training
    input_data_encoded = input_data.reindex(columns=encoder_info['one_hot_columns'], fill_value=0)

    # Apply scaling
    try:
        input_data_scaled = scaler.transform(input_data_encoded)
        
        # Perform prediction
        prediction = model.predict_proba(input_data_scaled)[:, 1]
        st.write(f"The predicted probability of the positive class is: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.error("The model or preprocessing objects could not be loaded. Please check the paths.")
