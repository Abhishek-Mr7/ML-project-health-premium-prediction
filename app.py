import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder  # Make sure to include any dependencies if necessary

# Title of the app
st.title("Health Insurance Cost Prediction App")

# File uploader for model file (User uploads .pkl file)
uploaded_model = st.file_uploader("C:\\Users\\Abhishek MR\\OneDrive\\Desktop\\Machine learning\\health_premium_prediction (2).pkl"
, type=["pkl"])

# Check if a file is uploaded
if uploaded_model is not None:
    try:
        # Load the uploaded model
        model = pickle.load(uploaded_model)
        st.success("Model loaded successfully!")

        # Function to preprocess the input data
        def preprocess_input(input_dict):
            # Ensure your input matches the format required by the model.
            # For example, encoding categorical variables if needed, and normalizing/standardizing numeric inputs

            # Convert input_dict to a pandas DataFrame (modify the columns to fit your model's expected input)
            df = pd.DataFrame(input_dict, index=[0])  # A single-row dataframe with input values
            return df

        # Define the input fields (you can adjust them as per your model's input requirements)
        age = st.number_input('Age', min_value=18, max_value=100, step=1)
        number_of_dependants = st.number_input('Number of Dependants', min_value=0, max_value=20, step=1)
        income_lakhs = st.number_input('Income in Lakhs', min_value=0, max_value=200, step=1)
        genetical_risk = st.number_input('Genetical Risk', min_value=0, max_value=5, step=1)
        
        # Example: Add other categorical inputs like Gender, Region, etc.
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital Status', ['Unmarried', 'Married'])
        bmi_category = st.selectbox('BMI Category', ['Normal', 'Obesity', 'Overweight', 'Underweight'])
        
        # Collect all input data into a dictionary
        input_dict = {
            'Age': age,
            'Number of Dependants': number_of_dependants,
            'Income in Lakhs': income_lakhs,
            'Genetical Risk': genetical_risk,
            'Gender': gender,
            'Marital Status': marital_status,
            'BMI Category': bmi_category
        }

        # Prediction on button click
        if st.button("Predict"):
            # Preprocess the input data
            input_df = preprocess_input(input_dict)

            # Make prediction using the loaded model
            prediction = model.predict(input_df)

            # Show the predicted value
            st.success(f"Predicted Health Insurance Cost: ₹{prediction[0]}")

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

else:
    st.info("Please upload your trained model file to proceed.")
