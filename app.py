import streamlit as st
import pickle
import pandas as pd

# Load the model safely
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# Sidebar Input
st.sidebar.header("Patient Data")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.number_input("Glucose", 0, 300, 100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.number_input("Age", 1, 120, 30)

    # IMPORTANT: Column order must match model training
    data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    return data

input_df = user_input_features()

# Show inputs
st.subheader("📋 Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("🔍 Result")

        if prediction[0] == 1:
            st.error("⚠️ Patient is likely to have Diabetes")
        else:
            st.success("✅ Patient is unlikely to have Diabetes")

        st.subheader("📊 Prediction Probability")
        st.write(f"Diabetes: {prediction_proba[0][1]:.2%}")
        st.write(f"No Diabetes: {prediction_proba[0][0]:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
