import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict if a patient is likely to have diabetes.")

# Input fields based on the feature names in your model
st.sidebar.header("Patient Data")

def user_input_features():
    # These match the feature_names_in_ found in your .pkl file
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Result")
    if prediction[0] == 1:
        st.error("The model predicts the patient is likely to have Diabetes.")
    else:
        st.success("The model predicts the patient is unlikely to have Diabetes.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Diabetes: {prediction_proba[0][1]:.2%}")
    st.write(f"Probability of No Diabetes: {prediction_proba[0][0]:.2%}")
        st.write(f"**Confidence Level:** {np.max(prediction_proba[0]):.2%}")
