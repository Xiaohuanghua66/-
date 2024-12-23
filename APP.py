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
    "LB", "AC", "FM", "UC", "DP", "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Mode", "Mean", "Median", "Variance", "Tendency"
]

# Streamlit user interface
st.title("Fetal State Predictor")

# Input features
lb = st.number_input("Fetal baseline heart rate (LB):", min_value=50, max_value=200, value=120)
ac = st.number_input("Accelerations (AC):", min_value=0, max_value=10, value=0)
fm = st.number_input("Fetal movements (FM):", min_value=0, max_value=20, value=0)
uc = st.number_input("Uterine contractions (UC):", min_value=0, max_value=20, value=0)
dp = st.number_input("Light decelerations (DP):", min_value=0, max_value=10, value=0)
astv = st.number_input("Percentage of time with abnormal short-term variability (ASTV):", min_value=0, max_value=100, value=73)
mstv = st.number_input("Mean value of short-term variability (MSTV):", min_value=0.0, max_value=10.0, value=0.5)
altv = st.number_input("Percentage of time with abnormal long-term variability (ALTV):", min_value=0, max_value=100, value=43)
mltv = st.number_input("Mean value of long-term variability (MLTV):", min_value=0.0, max_value=20.0, value=2.4)
width = st.number_input("Width of FHR histogram (Width):", min_value=0, max_value=200, value=64)
min_val = st.number_input("Minimum FHR value (Min):", min_value=0, max_value=200, value=62)
max_val = st.number_input("Maximum FHR value (Max):", min_value=0, max_value=200, value=126)
nmax = st.number_input("Number of histogram peaks (Nmax):", min_value=0, max_value=10, value=2)
mode = st.number_input("Histogram mode (Mode):", min_value=0, max_value=200, value=120)
mean = st.number_input("Histogram mean (Mean):", min_value=0, max_value=200, value=137)
median = st.number_input("Histogram median (Median):", min_value=0, max_value=200, value=121)
variance = st.number_input("Histogram variance (Variance):", min_value=0, max_value=200, value=73)
tendency = st.selectbox("Tendency (0=Normal, 1=Suspicious):", options=[0, 1])

# Collect input values into a list
feature_values = [lb, ac, fm, uc, dp, astv, mstv, altv, mltv, width, min_val, max_val, nmax, mode, mean, median, variance, tendency]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
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