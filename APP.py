import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the fetal state model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = [
    "Cut Sign (CT)",
    "Reverse Halo Sign (RHS)",
    "Surgical History (SH)",
    "Tumor Indicator (TI)",
    "Diabetes Mellitus (DM)",
    "Halo Sign (HS)",
    "High Blood Pressure (HBP)",
    "Multiple Lung Comorbidity (MLC)",
    "Smoking Status",
    "Airspace Involvement (AI)",
    "Lesion Location (LL)",
    "Sex (Male=1)",
    "Calcification (CA)",
    "Air Bronchogram (AB)",
    "Satellite Lesion (SL)",
    "Number of Satellite Lesions (NSL)",
    "Distance Between Satellite Lesions and the Main Stem Lesion (DBS)",
    "Long Axis of Satellite Lesions",
    "Rimmed Sign"
]

# Streamlit user interface
st.title("Fetal State Predictor")

# Input features
ct = st.selectbox(feature_names[0], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
rhs = st.selectbox(feature_names[1], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
sh = st.selectbox(feature_names[2], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ti = st.selectbox(feature_names[3], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
dm = st.selectbox(feature_names[4], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
hs = st.selectbox(feature_names[5], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
hbp = st.selectbox(feature_names[6], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
mlc = st.selectbox(feature_names[7], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
smoking_status = st.selectbox(feature_names[8], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ai = st.selectbox(feature_names[9], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ll = st.selectbox(feature_names[10], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
sex = st.selectbox(feature_names[11], options=[0, 1], index=0, help="0=Female, 1=Male")
ca = st.selectbox(feature_names[12], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ab = st.selectbox(feature_names[13], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
sl = st.selectbox(feature_names[14], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
nsl = st.selectbox(feature_names[15], options=[0, 1, 2], index=0, help="0=Normal, 1=Suspicious, 2=Have")
dbs = st.number_input(feature_names[16], min_value=0, max_value=5, value=0)
long_axis = st.number_input(feature_names[17], min_value=0, max_value=3, value=0)
rimmed_sign = st.selectbox(feature_names[18], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")

# Collect input values into a list
feature_values = [ct, rhs, sh, ti, dm, hs, hbp, mlc, smoking_status, ai, ll, sex, ca, ab, sl, nsl, dbs, long_axis, rimmed_sign]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Create DataFrame from user input
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    # Predict class and probabilities using DataFrame
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"According to our model, the fetus is in an anomalous state. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of being anomalous. "
            "It is strongly advised to seek immediate medical attention for further evaluation."
        )
    elif predicted_class == 1:
        advice = (
            f"According to our model, the fetus is in a normal state. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of being normal. "
            "It is recommended to continue monitoring the fetus's health regularly."
        )
    else:
        advice = (
            f"According to our model, the fetus is in a suspicious state. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of being suspicious. "
            "Further evaluation and close monitoring are recommended."
        )

    st.write(advice)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer(features_df)

    # Display SHAP waterfall plot only for the predicted class
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[:,:,predicted_class][0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
