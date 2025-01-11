import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the fetal state model
model = joblib.load('XGBoost.pkl')

# Define feature names
import streamlit as st

feature_names = [
"Sex",  # Sex (Male=1)
"Smoke",  # Smoking Status
"Surgical history",  # Surgical History (SH)
"HBP",  # High Blood Pressure (HBP)
"DM",  # Diabetes Mellitus (DM)
"MLC",  # Multiple Lung Comorbidity (MLC)
"Tumour indicator",  # Tumor Indicator (TI)
"Location",  # Lesion Location (LL)
"Airspace",  # Airspace Involvement (AI)
"Air bronchogram",  # Air Bronchogram (AB)
"Calcification",  # Calcification (CA)
"Rimmed sign",  # Rimmed Sign
"Satellite lesion",  # Satellite Lesion (SL)
"Long diameter of satellite lesion ",  # Long Axis of Satellite Lesions
"Distance between satellite lesion and main stem lesion",  # Distance Between Satellite Lesions and the Main Stem Lesion (DBS)
"Number of satellite lesion",  # Number of Satellite Lesions (NSL)
"Halo sign",  # Halo Sign (HS)
"Cut sign",  # Cut Sign (CT)
"Reverse halo sign"  # Reverse Halo Sign (RHS)
]

# Streamlit user interface
st.title("Pulmonary lesions Predictor")

# Input features
sex = st.selectbox(feature_names[17], options=[0, 1], index=0, help="0=Female, 1=Male")
smoke = st.selectbox(feature_names[18], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
sh = st.selectbox(feature_names[2], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
hbp = st.selectbox(feature_names[6], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
dm = st.selectbox(feature_names[4], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
mlc = st.selectbox(feature_names[16], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ti = st.selectbox(feature_names[3], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ll = st.selectbox(feature_names[5], options=[1, 2, 3, 4, 5], index=0, help="1=Right Upper Lung, 2=Right Middle Lung, 3=Right Lower Lung, 4=Left Upper Lung, 5=Left Lower Lung")
ai = st.selectbox(feature_names[1], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ab = st.selectbox(feature_names[8], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ca = st.selectbox(feature_names[7], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
rimmed_sign = st.selectbox(feature_names[13], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
sl = st.selectbox(feature_names[10], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
long_axis = st.number_input(feature_names[9], min_value=0, max_value=3, value=0)
dbs = st.number_input(feature_names[12], min_value=0, max_value=5, value=0)
nsl = st.selectbox(feature_names[15], options=[0, 1, 2], index=0, help="0=Normal, 1=Suspicious, 2=Have")
hs = st.selectbox(feature_names[14], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
ct = st.selectbox(feature_names[0], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")
rhs = st.selectbox(feature_names[11], options=[0, 1], index=0, help="0=Normal, 1=Suspicious")

# Collect input values into a list
feature_values = [ct, rhs, sh, ti, dm, hs, hbp, mlc, smoke, ai, ll, sex, ca, ab, sl, nsl, dbs, long_axis, rimmed_sign]

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

    # 将 predicted_class 映射为新的类别：0 -> 1, 1 -> 2, 2 -> 3
    mapped_class = predicted_class + 1

    # 根据映射后的类别给出相应的建议
    if mapped_class == 1:
        advice = (
            f"According to our model, the fetus is diagnosed with granulomatous inflammation. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of having granulomatous inflammation. "
            "It is strongly advised to initiate anti-inflammatory treatment."
        )
    elif mapped_class == 2:
        advice = (
            f"According to our model, the fetus is diagnosed with benign tumors. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of having benign tumors. "
            "It is recommended to continue regular follow-up and monitoring."
        )
    else:  # mapped_class == 3
        advice = (
            f"According to our model, the fetus is diagnosed with non-small cell lung cancer. "
            f"The model predicts that the fetus has a {probability:.1f}% probability of having non-small cell lung cancer. "
            "Further treatment and evaluation are recommended."
        )

    # 显示建议
    st.write(advice)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer(features_df)

    # Display SHAP waterfall plot only for the predicted class
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[:,:,predicted_class][0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
