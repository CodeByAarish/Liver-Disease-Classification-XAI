import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- CUSTOM FUNCTIONS FOR MODEL LOADING ---
def cap_outliers(df):
    return df

# Page Configuration
st.set_page_config(
    page_title="HepaScan | Liver Diagnostic System",
    page_icon="⚕️",
    layout="wide"
)

# --- RED, BLACK & BLUE MEDICAL THEME (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 {
        color: #ff4b4b; 
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        text-align: center;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    h3, .stSubheader { color: #1f77b4 !important; font-weight: bold; }
    label p {
        color: #ff4b4b !important; 
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background-color: #1a1c23;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
        border: 1px solid #333;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background: linear-gradient(135deg, #1f77b4 0%, #ff4b4b 100%);
        color: white;
        font-weight: bold;
        border: none;
        font-size: 1.2rem;
    }
    .footer-credits {
        position: fixed;
        right: 20px;
        bottom: 20px;
        text-align: right;
        font-family: 'Courier New', monospace;
        background-color: rgba(0, 0, 0, 0.9);
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ff4b4b;
        font-size: 13px;
        color: #ffffff;
        z-index: 100;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. Model Loading
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('liver_model.pkl', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.title("⚕️ HepaScan: Liver Diagnostic Framework")
st.info("💡 **Clinical Note:** This system uses Advanced Machine Learning for hepatic analysis. Labels explain clinical significance.")

# 2. Input Form
with st.container():
    st.subheader("📋 Diagnostic Variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=110, value=30)
        st.caption("Patient's biological age.")
        
        gender = st.selectbox("Gender", options=["Male", "Female"])
        st.caption("Biological sex.")
        
        bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0, step=0.1)
        st.caption("Measures jaundice or bile duct blockage.")
        
    with col2:
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3, step=0.1)
        st.caption("Portion of bilirubin processed by liver.")
        
        alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0, step=1.0)
        st.caption("High levels indicate bile duct issues.")
        
        sgpt = st.number_input("SGPT / ALT (U/L)", value=35.0, step=1.0)
        st.caption("Primary indicator of liver cell damage.")

    with col3:
        sgot = st.number_input("SGOT / AST (U/L)", value=40.0, step=1.0)
        st.caption("Enzyme indicating liver damage.")
        
        total_proteins = st.number_input("Total Proteins (g/dL)", value=6.8, step=0.1)
        st.caption("Sum of all proteins in blood.")
        
        albumin = st.number_input("Albumin (g/dL)", value=3.5, step=0.1)
        st.caption("Main protein made by liver.")

# Secondary Row
c1, c2, c3 = st.columns(3)
with c1:
    globulin = st.number_input("Globulin (g/dL)", value=3.0, step=0.1)
    st.caption("Proteins that help fight infection.")
with c2:
    ag_ratio = st.number_input("A/G Ratio", value=1.1, step=0.1)
    st.caption("Ratio of Albumin to Globulin.")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Prediction Logic
if st.button("RUN DIAGNOSTIC ANALYSIS"):
    if model:
        try:
            # FIX: Gender encoded ko hata kar direct string bhej rahe hain 
            # kyunki aapka pipeline 'Male'/'Female' expect kar raha hai.
            data_dict = {
                'Age': [age], 
                'Gender': [gender], # 'Male' or 'Female' string
                'Total_Bilirubin': [bilirubin],
                'Direct_Bilirubin': [direct_bilirubin], 
                'Alkaline_Phosphotase': [alk_phos],
                'SGPT_ALT': [sgpt], 
                'SGOT_AST': [sgot], 
                'Total_Protiens': [total_proteins],
                'Albumin': [albumin], 
                'Globulin': [globulin], 
                'A/G_Ratio': [ag_ratio]
            }
            
            input_data = pd.DataFrame(data_dict)
            
            # Apply outlier capping if it was part of training
            input_data = cap_outliers(input_data)

            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.subheader("📑 Final Assessment Report")
            
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if prediction[0] == 1:
                    st.error("### ⚠️ Result: Pathological Risk Detected")
                    st.write("**Assessment:** Clinical profile indicates high risk. Consultation at J.N. Medical College (A.M.U) is advised.")
                else:
                    st.success("### ✅ Result: Normal Physiological Profile")
                    st.write("**Assessment:** Parameters are within standard limits.")

            with res_col2:
                if probability is not None:
                    st.metric(label="Risk Percentage", value=f"{round(probability * 100, 2)}%")
            
        except Exception as e:
            st.error(f"Computation Error: {e}")

# Footer
st.markdown(f"""
    <div class="footer-credits">
        <span style="color:#ff4b4b;"><strong>Name:</strong></span> Aarish Ali (GQ2864)<br>
        <span style="color:#1f77b4;"><strong>Project:</strong></span> HepaScan: Liver Disease XAI<br>
        <span style="color:#ff4b4b;"><strong>Dept:</strong></span> Dept. of Statistics & Operations Research, A.M.U
    </div>
    """, unsafe_allow_html=True)