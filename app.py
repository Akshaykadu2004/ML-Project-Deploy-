import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="Diabetes Health Predictor", layout="wide")

# Load the model [source: 1]
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_value=True)

# Centering content using columns
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    st.title("🩺 Diabetes Health Predictor")
    st.markdown("---")
    st.write("Please adjust the sliders below to match the patient's clinical data.")

    # Input Section
    with st.container():
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 110)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 80)
        bmi = st.slider("BMI (Body Mass Index)", 0.0, 67.1, 26.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age (Years)", 21, 100, 33)

    # Creating the feature dataframe [source: 1]
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    st.markdown("---")
    
    if st.button("Analyze Health Data"):
        # Model Inference
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        confidence = np.max(prediction_proba[0])

        # Categorical Mapping: Converting 0/1 to Text
        result_text = "Positive" if prediction[0] == 1 else "Negative"
        
        # Display Result
        st.subheader("Prediction Result")
        
        if result_text == "Positive":
            st.error(f"**Status:** {result_text}")
            st.warning(f"Confidence Level: {confidence:.2%}")
            st.info("Recommendation: Consult with a healthcare professional for further diagnostic testing.")
        else:
            st.success(f"**Status:** {result_text}")
            st.write(f"Confidence Level: {confidence:.2%}")
            st.balloons()

    st.markdown("---")
    st.caption("Note: This tool is for educational purposes and uses a K-Nearest Neighbors model for estimations.")
