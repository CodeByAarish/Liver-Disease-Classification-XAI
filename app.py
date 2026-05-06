import streamlit as st
import pandas as pd
import pickle
import numpy as np


def cap_outliers(df):
    return df

# Page Configuration
st.set_page_config(
    page_title="HepaScan | Liver Clinical Decision Support",
    page_icon="⚕️",
    layout="wide"
)

# --- PROFESSIONAL MEDICAL THEME (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004a99;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #003366;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #004a99;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .medical-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. Model Loading with Error Handling
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('liver_model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Diagnostic Model File ('liver_model.pkl') not found. Please contact administrator.")
        return None

model = load_model()

# Header Section
st.title("⚕️ HepaScan: Advanced Liver Diagnostic System")
st.markdown("---")
st.info("Clinical Decision Support System (CDSS) for Healthcare Professionals. Please enter validated lab results below.")

# 2. Structured Input Form
with st.container():
    st.subheader("📋 Patient Clinical Markers")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=110, value=30, help="Patient's biological age.")
        bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0, step=0.1, format="%.1f")
        
    with col2:
        sgot = st.number_input("SGOT / AST (U/L)", value=40.0, step=1.0)
        alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0, step=1.0)

    with col3:
        albumin = st.number_input("Albumin (g/dL)", value=3.5, step=0.1, format="%.1f")
        prothrombin = st.number_input("Prothrombin Time (Seconds)", value=12.0, step=0.1)

st.markdown("---")

# 3. Prediction & Clinical Interpretation
if st.button("GENERATE DIAGNOSTIC REPORT"):
    if model:
        # Data preparation
        input_data = pd.DataFrame([[age, bilirubin, albumin, sgot, alk_phos, prothrombin]],
                                columns=['Age', 'Bilirubin', 'Albumin', 'SGOT', 'Alk_Phosphate', 'Prothrombin'])
        
        # Inference
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("📑 Diagnostic Summary")
        
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction[0] == 1:
                st.error("### Result: High Probability of Hepatic Pathology")
                st.write("**Interpretation:** The clinical markers provided align with profiles commonly associated with chronic liver conditions. Immediate clinical correlation and specialist consultation are advised.")
            else:
                st.success("### Result: Low Risk / Healthy Profile")
                st.write("**Interpretation:** The input markers fall within statistically normal ranges. No immediate hepatic abnormalities detected by the system.")

        with res_col2:
            if probability is not None:
                st.metric(label="Risk Probability", value=f"{round(probability * 100, 2)}%")
            
        # Medical Disclaimer
        st.warning("**Disclaimer:** This tool is for informational purposes only and does not replace professional medical judgment. All automated findings must be verified by a licensed medical practitioner.")
    
# Footer
st.markdown("<br><hr><center><small>HepaScan CDSS v1.0 | Developed for Clinical Research</small></center>", unsafe_allow_html=True)