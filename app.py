import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Custom CSS for a cleaner, centered look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    .reportview-container .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Centering content using columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("🩺 Diabetes Health Predictor")
    st.write("Please provide the clinical details below to analyze the risk.")
    st.divider()

    # Input Section
    with st.container():
        # Using feature names from model[cite: 1]
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 110)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 79)
        bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.1, 32.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
        age = st.slider("Age (Years)", 21, 81, 33)

    # Prepare data for prediction[cite: 1]
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    st.divider()

    # Prediction Button
    if st.button("Run Diagnostic Analysis"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Categorical Mapping
        # 0 = Tested Negative, 1 = Tested Positive
        status = "Tested Positive" if prediction[0] == 1 else "Tested Negative"
        
        st.subheader("Analysis Result")
        
        if prediction[0] == 1:
            st.error(f"**Diagnostic Result:** {status}")
            st.write(f"The model has a **{prediction_proba[0][1]:.2%}** confidence in this result.")
        else:
            st.success(f"**Diagnostic Result:** {status}")
            st.write(f"The model has a **{prediction_proba[0][0]:.2%}** confidence in this result.")
            st.balloons()

    st.divider()
    st.caption("Disclaimer: This tool is for informational purposes only and does not replace professional medical advice.")
