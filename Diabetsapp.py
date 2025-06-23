import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set up the page
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load the model safely
@st.cache_resource  # Ensures model loads only once and stays in memory
def load_model():
    try:
        model = joblib.load("Diabets.pkl")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model1 = load_model()

# UI for input
st.title("Diabetes Predictor")
st.markdown("Enter the health details below:")

with st.form("prediction_form"):
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input("Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
    Age = st.number_input("Age", min_value=1, max_value=120, value=33)

    submitted = st.form_submit_button("Predict")

if submitted and model1:
    input_df = pd.DataFrame([{
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }])

    st.subheader("Entered Data")
    st.dataframe(input_df)

    try:
        prediction = model1.predict(input_df)
        outcome = "Yes" if prediction[0] == 1 else "No"
        st.success(f"Diabetes Prediction: **{outcome}**")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
