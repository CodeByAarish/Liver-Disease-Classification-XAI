import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. CORE FUNCTIONAL DEFINITIONS (CRITICAL FOR PICKLE) ---
def cap_outliers(df):
    """
    Mandatory placeholder for the model's preprocessing pipeline.
    Must be defined before model loading to prevent AttributeError.
    """
    return df

@st.cache_resource
def load_diagnostic_model():
    try:
        with open('liver_final_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Critical System Failure: Unable to initialize diagnostic core. Error: {e}")
        return None

@st.cache_resource
def get_shap_explainer(_model):
    """Initializes the SHAP TreeExplainer for high-speed inference."""
    return shap.TreeExplainer(_model)

# Initialize the model
model = load_diagnostic_model()

# --- 2. GLOBAL UI CONFIGURATION ---
st.set_page_config(
    page_title="HepaScan | Advanced Hepatic Diagnostic Suite",
    page_icon="⚕️",
    layout="wide"
)

# Professional Clinical Theme (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .main { background-color: #0b0e14; font-family: 'Inter', sans-serif; }
    
    .main-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0px;
        padding-top: 20px;
    }
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Professional Input Cards */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre;
        background-color: #161b22;
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #1e40af; color: white; }

    /* Buttons & Interactions */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
    }

    /* Sidebar/Footer Credits */
    .footer-credits {
        position: fixed;
        right: 20px;
        bottom: 20px;
        text-align: right;
        background: rgba(22, 27, 34, 0.9);
        padding: 15px 25px;
        border-radius: 12px;
        border: 1px solid #30363d;
        font-size: 12px;
        color: #8b949e;
        backdrop-filter: blur(8px);
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ HepaScan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Hepatic Diagnostic Framework & Predictive Analytics</p>', unsafe_allow_html=True)

# --- 3. CLINICAL DATA ENTRY ---
with st.container():
    st.info("🔬 **Clinical Protocol:** Please input validated laboratory results to proceed with the AI-driven hepatic assessment.")
    
    t1, t2 = st.tabs(["📊 Patient Biometrics", "🧪 Laboratory Biomarkers"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Chronological Age (Years)", min_value=1, max_value=110, value=30)
            gender = st.selectbox("Biological Sex", options=["Male", "Female"])
        with c2:
            bilirubin = st.number_input("Total Serum Bilirubin (mg/dL)", value=1.0, step=0.1)
            direct_bilirubin = st.number_input("Direct (Conjugated) Bilirubin (mg/dL)", value=0.3, step=0.1)
        with c3:
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0, step=1.0)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", value=1.1, step=0.1)

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            sgpt = st.number_input("SGPT / Alanine Aminotransferase (ALT) (U/L)", value=35.0, step=1.0)
            sgot = st.number_input("SGOT / Aspartate Aminotransferase (AST) (U/L)", value=40.0, step=1.0)
        with c2:
            total_proteins = st.number_input("Total Serum Proteins (g/dL)", value=6.8, step=0.1)
            albumin = st.number_input("Serum Albumin (g/dL)", value=3.5, step=0.1)
        with c3:
            globulin = st.number_input("Serum Globulin (g/dL)", value=3.3, step=0.1)
            st.warning("Manual verification of results is recommended before diagnostic execution.")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. PREDICTION & XAI SUITE ---
if st.button("Execute Diagnostic Analysis"):
    if model:
        try:
            # Map input to DataFrame (preserving training column sequence)
            data_dict = {
                'Age': [age], 'Gender': [gender], 'Total_Bilirubin': [bilirubin],
                'Direct_Bilirubin': [direct_bilirubin], 'Alkaline_Phosphotase': [alk_phos],
                'SGPT_ALT': [sgpt], 'SGOT_AST': [sgot], 'Total_Protiens': [total_proteins],
                'Albumin': [albumin], 'A/G_Ratio': [ag_ratio], 'Globulin': [globulin]
            }
            input_df = pd.DataFrame(data_dict)
            input_df = cap_outliers(input_df)

            # Execution
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown("### 📑 Clinical Assessment Report")
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if prediction[0] == 1:
                    st.error("#### 🚩 Status: Pathological Risk Identified")
                    st.markdown("""
                    **Diagnostic Inference:** The clinical profile correlates with known hepatic pathology signatures. 
                    Clinical correlation and further serological titration are strictly advised.
                    """)
                else:
                    st.success("#### ✅ Status: Normal Physiological Profile")
                    st.markdown("""
                    **Diagnostic Inference:** The analyzed biomarkers fall within established homeostatic ranges. 
                    No immediate clinical risk detected.
                    """)

            with res_col2:
                if probability is not None:
                    st.metric(label="Systemic Risk Index", value=f"{round(probability * 100, 2)}%")
            
            # --- SHAP EXPLAINABLE AI ---
            st.markdown("---")
            st.subheader("🔬 Biometric Feature Contribution Analysis (XAI)")
            st.write("Delineating the impact of each clinical biomarker on the final diagnostic prognosis.")

            # Generate SHAP Analysis
            explainer = get_shap_explainer(model)
            shap_values = explainer(input_df)
            
            # Matplotlib Plot Configuration
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            
            # Apply "Dashing" Dark Mode Styling to the plot
            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            st.pyplot(fig)
            
            with st.expander("🔬 Interpretation of XAI Metrics"):
                st.write("""
                * **Red Bars (Right):** Parameters that specifically increased the probability of a pathological result.
                * **Blue Bars (Left):** Parameters that served as protective factors, decreasing the risk index.
                * **f(x):** The localized prediction probability for this specific patient.
                * **E[f(x)]:** The global baseline average across the reference clinical dataset.
                """)

        except Exception as e:
            st.error(f"Computation Error: {e}")
    else:
        st.error("System Core Offline: Please ensure 'liver_final_model.pkl' is correctly placed in the root directory.")

# --- 5. FOOTER & CREDENTIALS ---
st.markdown(f"""
    <div class="footer-credits">
        <strong style="color:white;">Aarish Ali</strong> (GQ2864)<br>
        Lead Researcher | HepaScan Initiative<br>
        <span style="font-size:10px;">Dept. of Statistics & Operations Research, A.M.U</span>
    </div>
    """, unsafe_allow_html=True)