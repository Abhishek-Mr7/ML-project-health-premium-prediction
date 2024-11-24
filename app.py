import streamlit as st
import pandas as pd
import pickle

# Load the model
with open("health_premium_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Function to calculate normalized risk
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14  # Maximum combined risk score
    min_score = 0   # Minimum risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

# Function to preprocess input data
def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Populate DataFrame with user input
    df['age'] = input_dict['Age']
    df['number_of_dependants'] = input_dict['Number of Dependants']
    df['income_lakhs'] = input_dict['Income in Lakhs']
    df['genetical_risk'] = input_dict['Genetical Risk']
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict['Insurance Plan'], 1)
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Encode categorical variables
    if input_dict['Gender'] == 'Male':
        df['gender_Male'] = 1
    if input_dict['Region'] == 'Northwest':
        df['region_Northwest'] = 1
    elif input_dict['Region'] == 'Southeast':
        df['region_Southeast'] = 1
    elif input_dict['Region'] == 'Southwest':
        df['region_Southwest'] = 1
    if input_dict['Marital Status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    if input_dict['BMI Category'] == 'Obesity':
        df['bmi_category_Obesity'] = 1
    elif input_dict['BMI Category'] == 'Overweight':
        df['bmi_category_Overweight'] = 1
    elif input_dict['BMI Category'] == 'Underweight':
        df['bmi_category_Underweight'] = 1
    if input_dict['Smoking Status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif input_dict['Smoking Status'] == 'Regular':
        df['smoking_status_Regular'] = 1
    if input_dict['Employment Status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif input_dict['Employment Status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    return df

# Function to predict health insurance cost
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])

# Streamlit app layout
st.title("Health Insurance Cost Predictor")

# Define input options
categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer', ''],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# Input fields
age = st.number_input('Age', min_value=18, max_value=100, step=1)
number_of_dependants = st.number_input('Number of Dependants', min_value=0, max_value=20, step=1)
income_lakhs = st.number_input('Income in Lakhs', min_value=0, max_value=200, step=1)
genetical_risk = st.number_input('Genetical Risk', min_value=0, max_value=5, step=1)
insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])
gender = st.selectbox('Gender', categorical_options['Gender'])
marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])
smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])
region = st.selectbox('Region', categorical_options['Region'])
medical_history = st.selectbox('Medical History', categorical_options['Medical History'])

# Prepare input dictionary
input_dict = {
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Genetical Risk': genetical_risk,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# Prediction button
if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f'Predicted Health Insurance Cost: ₹{prediction}')
