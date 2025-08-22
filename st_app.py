#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================
# Load Model & Features
# =====================
model = joblib.load("titanic_logreg_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# =====================
# User Inputs
# =====================
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# =====================
# Preprocess Input
# =====================
# Encode sex
sex_encoded = 1 if sex == "Female" else 0

# One-hot encode embarked
embarked_S = 1 if embarked == "S" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

# Build input row
input_data = pd.DataFrame([[
    pclass, sex_encoded, age, sibsp, parch, fare,
    embarked_C, embarked_Q  # drop_first=True was used in preprocessing
]], columns=features)

# =====================
# Prediction
# =====================
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ This passenger would SURVIVE! (Probability: {probability:.2f})")
    else:
        st.error(f"❌ This passenger would NOT survive. (Probability: {probability:.2f})")

