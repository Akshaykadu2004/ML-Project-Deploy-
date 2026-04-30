import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

st.title("Diabetes Prediction App")
st.write("""
This app predicts if a patient has diabetes based on clinical data.
""")

st.sidebar.header("Patient Data Input")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    return features

input_df = user_input_features()

st.subheader("Patient Summary")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    status = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"The model predicts the patient is **{status}** for diabetes.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of being Positive: {prediction_proba[0][1]:.2%}")
    st.write(f"Probability of being Negative: {prediction_proba[0][0]:.2%}")
