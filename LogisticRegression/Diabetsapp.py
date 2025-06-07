import streamlit as st
import numpy as np
import pandas as pd
import joblib

#set up the configuration
st.set_page_config(page_title="Diabets Predicition",layout='centered')

#Load the model
model1 = joblib.load("Diabets.pkl")

# Streamlit UI to take inputs
with st.form("tip_form"):
    Pregnancies = st.number_input("Pregnancies:", min_value=0, max_value=6,value=2,step=1)
    Glucose = st.number_input("Glucose",min_value=40,max_value=300,value=40)
    BloodPressure = st.number_input("BloodPressure",min_value=40,max_value=200,value=50)
    SkinThickness = st.number_input("SkinThickness",min_value=0,max_value=200,value=10)
    Insulin = st.number_input("Insulin",min_value=0,max_value=200,value=30)
    BMI = st.number_input("BMI", min_value=1.0,max_value=200.0, value=2.0)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0,max_value=100.0, value=1.0)
    Age = st.number_input("Age", min_value=5,max_value=200, value=10)

    # Submit button
    submitted = st.form_submit_button("Predict Tip")


# Prediction on form submission
if submitted:
    input_df = pd.DataFrame([{
        'Pregnancies':Pregnancies,
        'Glucose':Glucose,
        "BloodPressure":BloodPressure,
        'SkinThickness':SkinThickness,
        "Insulin":Insulin,
        "BMI":BMI,
        "DiabetesPedigreeFunction":DiabetesPedigreeFunction,
        "Age":Age
    }])

    # Print input data
    st.write("Input Data:")
    st.dataframe(input_df)

    # Check the model type again just before prediction
    st.write(f"Model type before prediction: {type(model1)}")  # Should show <class 'sklearn.pipeline.Pipeline'>
   
    try:
        # Predict the tip
        prediction = model1.predict(input_df)

        # Ensure the output is a scalar value
        predicted_tip = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        # Display the predicted tip
        if predicted_tip == 0:
            #st.success(f"Outcomes: *{predicted_tip:}*")
            st.success("Diabets: Yes")
        else:
            st.success("Diabets: No")
            
    except Exception as e:
        st.write('i am here in exception')
        st.error(f" Error: {str(e)}")