import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

# Set page to wide mode to help with centering
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Centering CSS
st.markdown("""
    <style>
    .main {
        display: flex;
        justify-content: center;
    }
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    h1, h3, p {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Diabetes Prediction App")
st.write("Enter the patient details below to determine the likelihood of diabetes.")

# Create three columns; we will put the input fields in the middle one to "center" it
left_co, cent_co, last_co = st.columns([1, 2, 1])

with cent_co:
    st.subheader("Patient Clinical Data")
    
    # Numerical Inputs
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, value=20)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, format="%.1f", value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.5)
    age = st.number_input("Age (Years)", min_value=1, step=1, value=30)

    # Centered Predict Button
    if st.button("Run Prediction Analysis", use_container_width=True):
        # Prepare the features for the model[cite: 1]
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        st.markdown("---")
        
        # Displaying Categorical Result instead of 0 or 1[cite: 1]
        if prediction[0] == 1:
            st.error("### Result: Diabetes Positive")
        else:
            st.success("### Result: Diabetes Negative")

        # Probability Visualization
        st.write(f"**Confidence Level:** {np.max(prediction_proba[0]):.2%}")
