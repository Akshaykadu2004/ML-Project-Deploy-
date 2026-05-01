import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration with an attractive title
st.set_page_config(
    page_title="AuraHealth | Diabetes Risk Intelligence", 
    page_icon="✨", 
    layout="wide"
)

# 2. Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. Enhanced Custom Styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to bottom, #ffffff, #f0f4f8);
    }
    
    /* Centered card-like container */
    [data-testid="stVerticalBlock"] > div:has(div.stButton) {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #007cf0, #00dfd8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,124,240,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 4. Centered Layout Construction
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    st.header("✨ AuraHealth Intelligence")
    st.write("Complete the patient profile to generate a predictive risk report.")
    st.markdown("---")

    # --- CATEGORICAL INPUTS (1st and 2nd Position) ---
    # 1st: Pregnancy History (Categorical)
    preg_map = {
        "None (0)": 0,
        "Low (1-2)": 1,
        "Moderate (3-5)": 4,
        "High (6+)": 8
    }
    selected_preg = st.selectbox("1. Pregnancy History", options=list(preg_map.keys()))
    pregnancies = preg_map[selected_preg]

    # 2nd: Glucose Status (Categorical)
    glucose_map = {
        "Normal (< 100 mg/dL)": 95,
        "Prediabetic (100-125 mg/dL)": 110,
        "Diabetic Level (126+ mg/dL)": 160
    }
    selected_glucose = st.selectbox("2. Glucose Status", options=list(glucose_map.keys()))
    glucose = glucose_map[selected_glucose]

    # --- REMAINING CLINICAL INPUTS ---
    st.write("#### Clinical Measurements")
    
    # Using columns for a compact, attractive look
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 21, 100, 30)
        bmi = st.number_input("BMI Index", 10.0, 70.0, 28.5)
        blood_pressure = st.number_input("Blood Pressure", 40, 140, 70)
        
    with c2:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.number_input("Insulin Level", 0, 850, 80)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

    # 5. Data Processing
    input_features = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    st.markdown("---")

    # 6. Result Generation
    if st.button("Generate Predictive Report"):
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features)
        
        # Categorical Result Mapping[cite: 1]
        status = "Tested Positive" if prediction[0] == 1 else "Tested Negative"
        confidence = np.max(prediction_proba[0])

        if prediction[0] == 1:
            st.error(f"### Assessment: {status}")
            st.write(f"The analysis indicates a **{confidence:.2%}** probability of diabetic health indicators.")
        else:
            st.success(f"### Assessment: {status}")
            st.write(f"The analysis indicates a **{confidence:.2%}** probability of a healthy clinical profile.")
            st.balloons()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by Scikit-Learn KNeighborsClassifier | Data: Pima Indians[cite: 1]")
