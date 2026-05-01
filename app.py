import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Health Insights Dashboard", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Custom Styling for a modern "Card" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stButton > button:first-child {
        background-color: #00b4d8;
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-size: 1.2em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #0077b6;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Centering the entire interface
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    st.title("🩺 Patient Health Predictor")
    st.write("Complete the clinical profile below to evaluate diabetes risk.")
    st.markdown("---")

    # --- CATEGORICAL INPUTS ---
    # Pregnancies converted to categorical ranges
    preg_options = {
        "0 (None)": 0,
        "1-2 (Low)": 1,
        "3-5 (Moderate)": 4,
        "6+ (High)": 8
    }
    selected_preg = st.selectbox("Pregnancy History", options=list(preg_options.keys()))
    pregnancies = preg_options[selected_preg]

    # Glucose converted to categorical health status
    glucose_options = {
        "Normal (Less than 100 mg/dL)": 90,
        "Prediabetic (100-125 mg/dL)": 115,
        "High (126+ mg/dL)": 150
    }
    selected_glucose = st.selectbox("Glucose Level Status", options=list(glucose_options.keys()))
    glucose = glucose_options[selected_glucose]

    # --- REMAINING INPUTS (Attractive Sliders) ---
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 130, 72)
    
    col_left, col_right = st.columns(2)
    with col_left:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        bmi = st.number_input("BMI", 10.0, 70.0, 31.0)
    with col_right:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 850, 80)
        age = st.number_input("Age (Years)", 21, 100, 30)

    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    # Prepare Data
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    st.markdown("---")

    if st.button("Generate Health Report"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Categorical Result Mapping
        result = "Positive" if prediction[0] == 1 else "Negative"
        confidence = np.max(prediction_proba[0])

        if result == "Positive":
            st.error(f"### Result: {result}")
            st.write(f"The analysis indicates a **{confidence:.2%}** probability of diabetic indicators.")
        else:
            st.success(f"### Result: {result}")
            st.write(f"The analysis indicates a **{confidence:.2%}** probability of a healthy profile.")
            st.balloons()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Educational Tool: Based on a KNeighborsClassifier trained on the Pima Indians dataset[cite: 1].")
