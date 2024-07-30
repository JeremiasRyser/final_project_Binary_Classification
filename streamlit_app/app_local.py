import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

# Cache the loading of preprocessing objects
@st.cache_data
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

# Cache the loading of the model
@st.cache_data
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to create a template CSV
def create_template_csv():
    template_data = {
        'Gender': ['female', 'male'],
        'Age': [25, 30],
        'Driving_License': [1, 0],
        'Region_code': [10, 20],
        'Previously_Insured': [1, 0],
        'Vehicle_Age': ['< 1 year', '> 2 years'],
        'Vehicle_Damage': [1, 0],
        'Annual_Premium': [10000, 15000],
        'Policy_Sales_Channel': [10, 20],
        'Vintage': [50, 100]
    }
    df_template = pd.DataFrame(template_data)
    return df_template

# Path to the model and preprocessing objects
model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'
encoder_path = 'encoder.pkl'

# Load the model and preprocessing objects
model = load_model(model_path)
scaler, encoder_info = load_preprocessing_objects(scaler_path, encoder_path)

# Title of the app
st.title("Insurance Data Prediction")

# Create two columns
col1, col2 = st.columns([2, 1])

# CSV file upload button
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# CSV template download button
with col2:
    st.write("")
    st.write("")
    st.write("")  # Add some spacing
    df_template = create_template_csv()
    csv_template = df_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name='template.csv',
        mime='text/csv'
    )

def generate_random_values():
    gender = random.choice(["female", "male"])
    age = random.randint(18, 100)
    driving_license = random.choice(["No", "Yes"])
    region_code = random.randint(0, 50)
    previously_insured = random.choice(["No", "Yes"])
    vehicle_age = random.choice(["< 1 year", "1-2 years", "> 2 years"])
    vehicle_damage = random.choice(["No", "Yes"])
    annual_premium = random.randint(0, 100000)
    policy_sales_channel = random.randint(0, 200)
    vintage = random.randint(0, 300)
    
    return gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage

# Add button to generate random values
if st.button("Generate Random Values"):
    gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage = generate_random_values()
else:
    gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage = "female", 30, "No", 1, "No", "< 1 year", "No", 10000, 1, 1

# Check if the model and preprocessing objects were successfully loaded
if model and scaler and encoder_info:
    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)

        # Perform preprocessing on the uploaded CSV data
        input_df['Gender_Male'] = input_df['Gender'].apply(lambda x: 1 if x == 'male' else 0)
        input_df['Vehicle_Age_< 1 Year'] = input_df['Vehicle_Age'].apply(lambda x: 1 if x == '< 1 year' else 0)
        input_df['Vehicle_Age_> 2 Year'] = input_df['Vehicle_Age'].apply(lambda x: 1 if x == '> 2 years' else 0)
        input_df['Vehicle_Damage_Yes'] = input_df['Vehicle_Damage'].apply(lambda x: 1 if x == 1 else 0)
        input_df['Driving_License'] = input_df['Driving_License'].apply(lambda x: 1 if x == 1 else 0)
        input_df['Previously_Insured'] = input_df['Previously_Insured'].apply(lambda x: 1 if x == 1 else 0)

        # Ensure the columns match those used during training
        input_data_encoded = input_df.reindex(columns=encoder_info['one_hot_columns'], fill_value=0)

        # Apply scaling
        try:
            input_data_scaled = scaler.transform(input_data_encoded)
            
            # Perform prediction
            predictions = model.predict_proba(input_data_scaled)[:, 1]
            input_df['Prediction'] = predictions * 100
            input_df['Prediction'] = input_df['Prediction'].apply(lambda x: f"{x:.2f}%")
            st.write(input_df)
            
            # Feature Importance
            importances = model.feature_importances_
            feature_names = encoder_info['one_hot_columns']

            # Sort features by importance
            sorted_idx = np.argsort(importances)
            sorted_features = np.array(feature_names)[sorted_idx]
            sorted_importances = importances[sorted_idx]

            # Plotting
            rcParams['font.family'] = 'Source Sans Pro'
            rcParams['font.size'] = 14
            plt.style.use('dark_background')
            fig, ax = plt.subplots()
            ax.barh(sorted_features, sorted_importances, color='#FF4B4B')
            ax.set_title("Feature Importance", color='white')
            ax.set_xlabel("Importance", color='white')
            ax.set_ylabel("Features", color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        # Input fields for the features
        gender = st.selectbox("Gender", ["female", "male"], index=["female", "male"].index(gender))
        age = st.slider("Age", 18, 100, age)
        driving_license = st.selectbox("Driving License", ["No", "Yes"], index=["No", "Yes"].index(driving_license))
        region_code = st.slider("Region Code", 0, 50, region_code)
        previously_insured = st.selectbox("Previously Insured", ["No", "Yes"], index=["No", "Yes"].index(previously_insured))
        vehicle_age = st.selectbox("Vehicle Age", ["< 1 year", "1-2 years", "> 2 years"], index=["< 1 year", "1-2 years", "> 2 years"].index(vehicle_age))
        vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"], index=["No", "Yes"].index(vehicle_damage))
        annual_premium = st.number_input("Annual Premium", min_value=0, max_value=100000, value=annual_premium)
        policy_sales_channel = st.slider("Sales Channel", 0, 200, policy_sales_channel)
        vintage = st.slider("Vintage", 0, 300, vintage)

        # Button to trigger prediction
        if st.button("Predict"):
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
                prediction = model.predict_proba(input_data_scaled)[:, 1][0] * 100
                st.write(f"The predicted probability of the positive response is: {prediction:.2f}%")
                
                # Feature Importance
                importances = model.feature_importances_
                feature_names = encoder_info['one_hot_columns']

                # Sort features by importance
                sorted_idx = np.argsort(importances)
                sorted_features = np.array(feature_names)[sorted_idx]
                sorted_importances = importances[sorted_idx]

                # Plotting
                rcParams['font.family'] = 'Source Sans Pro'
                rcParams['font.size'] = 14
                plt.style.use('dark_background')
                fig, ax = plt.subplots()
                ax.barh(sorted_features, sorted_importances, color='#FF4B4B')
                ax.set_title("Feature Importance", color='white')
                ax.set_xlabel("Importance", color='white')
                ax.set_ylabel("Features", color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    st.error("The model or preprocessing objects could not be loaded. Please check the paths.")
