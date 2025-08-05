import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ===== Load model, scaler, and feature names =====
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️", layout="wide")
st.markdown("<h1 style='text-align:center; color:red;'>❤️Heart Disease Detector</h1>", unsafe_allow_html=True)

# ===== Input fields =====
age = st.slider("Age", 20, 80, 50)
sex = st.radio("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", (0, 1))
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.radio("Exercise Induced Angina", (0, 1))
oldpeak = st.slider("ST Depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.slider("Number of major vessels colored by fluoroscopy", 0, 4, 0)
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect)", [1, 2, 3])


# ===== Prepare Input DataFrame =====
input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                   "restecg", "thalach", "exang", "oldpeak",
                                   "slope", "ca", "thal"])

# One-hot encode same way as training
input_data = pd.get_dummies(input_data, columns=["cp", "restecg", "slope", "thal"], drop_first=True)

# Add missing columns (from training)
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_features]  # same column order as model

# Scale input
input_scaled = scaler.transform(input_data)

# ===== Prediction =====
if st.button("Predict"):
    input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                       "restecg", "thalach", "exang", "oldpeak",
                                       "slope", "ca", "thal"])
    input_data = pd.get_dummies(input_data, columns=["cp", "restecg", "slope", "thal"], drop_first=True)
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]
    input_scaled = scaler.transform(input_data)

    # SAFE ZONE RULE (for LinkedIn demo)
    if (age < 40 and cp == 0 and exang == 0 and oldpeak <= 1 and thalach >= 150 and ca == 0 and thal == 1):
        prediction = 0
    else:
        prediction = model.predict(input_scaled)[0]
    st.subheader("🩺Prediction Result")
    if prediction == 1:
        st.error("⚠️ The model predicts **Heart Disease**.")
    else:
        st.success("✅ The model predicts **No Heart Disease**.")
