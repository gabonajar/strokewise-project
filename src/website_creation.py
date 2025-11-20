import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root
DATA_PATH = "/Users/gabrielnajarvelasquez/developments/Stroke Prediction Project/dat/healthcare-dataset-stroke-data-cleaned.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# === Load dataset for reference ===
df = pd.read_csv(DATA_PATH)

# === Load the latest model ===
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
if not model_files:
    st.error("No model files found in models directory!")
    st.stop()

# Sort alphabetically, pick the last one (latest)
latest_model_file = sorted(model_files)[-1]
model_path = os.path.join(MODELS_DIR, latest_model_file)
model = joblib.load(model_path)

# === Streamlit UI ===
st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("🧠 Stroke Prediction Web App")

st.markdown("""
This web app predicts the risk of stroke based on patient health information.
""")

# --- User inputs ---
st.sidebar.header("Enter Patient Information")

def user_input_features():
    age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), 30)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", float(df.avg_glucose_level.min()), float(df.avg_glucose_level.max()), 100.0)
    bmi = st.sidebar.slider("BMI", float(df.bmi.min()), float(df.bmi.max()), 25.0)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly_smoked", "never_smoked", "smokes", "Unknown"])

    data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Other": 1 if gender == "Other" else 0,
        "ever_married_Yes": 1 if ever_married == "Yes" else 0,
        "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_children": 1 if work_type == "children" else 0,
        "Residence_type_Urban": 1 if residence_type == "Urban" else 0,
        "smoking_status_formerly_smoked": 1 if smoking_status == "formerly_smoked" else 0,
        "smoking_status_never_smoked": 1 if smoking_status == "never_smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status == "smokes" else 0
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction ---
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.write("✅ Stroke Risk:" if prediction == 1 else "❌ No Stroke Risk")
st.write(f"Probability of Stroke: {prediction_proba:.2f}")

# --- Optional: Show input features ---
st.subheader("Patient Information")
st.dataframe(input_df)
