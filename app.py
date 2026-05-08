import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. MODEL LOADING & OPTIMIZATION ---
@st.cache_resource
def load_model():
    try:
        with open('liver_final_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Critical System Failure: Unable to initialize diagnostic core. {e}")
        return None

@st.cache_resource
def get_shap_explainer(_model):
    # SHAP explainer is cached to ensure rapid inference
    return shap.TreeExplainer(_model)

model = load_model()

# Page Configuration
st.set_page_config(
    page_title="HepaScan | Advanced Hepatic Diagnostic Suite",
    page_icon="⚕️",
    layout="wide"
)

# --- 2. PREMIUM CLINICAL UI (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .main { background-color: #0b0e14; font-family: 'Inter', sans-serif; }
    
    /* Header Styling */
    .main-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Section Cards */
    .input-card {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }

    /* Metric Badges */
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #38bdf8 !important; }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.8em;
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
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }

    /* Footer */
    .footer-credits {
        position: fixed;
        right: 20px;
        bottom: 20px;
        text-align: right;
        background: rgba(22, 27, 34, 0.95);
        padding: 15px 25px;
        border-radius: 12px;
        border: 1px solid #30363d;
        font-size: 12px;
        color: #8b949e;
        backdrop-filter: blur(10px);
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ HepaScan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Hepatic Diagnostic System & Predictive Analytics</p>', unsafe_allow_html=True)

# --- 3. CLINICAL INPUT INTERFACE ---
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Category 1: Demographics & Primary Markers
    t1, t2 = st.tabs(["📊 Patient Biometrics", "🧪 Laboratory Biomarkers"])
    
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Chronological Age", min_value=1, max_value=110, value=30)
            gender = st.selectbox("Biological Sex", options=["Male", "Female"])
        with c2:
            bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0, step=0.1)
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3, step=0.1)
        with c3:
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0, step=1.0)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", value=1.1, step=0.1)

    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            sgpt = st.number_input("SGPT / Alanine Aminotransferase (U/L)", value=35.0, step=1.0)
            sgot = st.number_input("SGOT / Aspartate Aminotransferase (U/L)", value=40.0, step=1.0)
        with c2:
            total_proteins = st.number_input("Total Serum Proteins (g/dL)", value=6.8, step=0.1)
            albumin = st.number_input("Serum Albumin (g/dL)", value=3.5, step=0.1)
        with c3:
            globulin = st.number_input("Serum Globulin (g/dL)", value=3.3, step=0.1)
            st.info("Ensure all values are calibrated to standard laboratory reference ranges.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. DIAGNOSTIC INFERENCE & XAI ---
if st.button("Execute Diagnostic Analysis"):
    if model:
        try:
            # Data Mapping (Preserving your exact column names)
            data_dict = {
                'Age': [age], 'Gender': [gender], 'Total_Bilirubin': [bilirubin],
                'Direct_Bilirubin': [direct_bilirubin], 'Alkaline_Phosphotase': [alk_phos],
                'SGPT_ALT': [sgpt], 'SGOT_AST': [sgot], 'Total_Protiens': [total_proteins],
                'Albumin': [albumin], 'A/G_Ratio': [ag_ratio], 'Globulin': [globulin]
            }
            
            input_df = pd.DataFrame(data_dict)
            
            # Predict
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown("### 📑 Clinical Assessment Report")
            
            # Result Visualization
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if prediction[0] == 1:
                    st.error("#### 🚩 Status: Pathological Profile Identified")
                    st.markdown("""
                    **Clinical Conclusion:** The systemic analysis indicates a high probability of hepatic dysfunction. 
                    Further serological investigation and imaging (Ultrasound/FibroScan) are recommended.
                    """)
                else:
                    st.success("#### ✅ Status: Normal Physiological Profile")
                    st.markdown("""
                    **Clinical Conclusion:** Current biomarkers are consistent with normal hepatic function. 
                    Standard periodic monitoring is suggested.
                    """)

            with res_col2:
                if probability is not None:
                    st.metric(label="Systemic Risk Index", value=f"{round(probability * 100, 2)}%")
            
            # --- SHAP XAI SECTION ---
            st.markdown("---")
            st.subheader("🔬 Biometric Feature Contribution Analysis (XAI)")
            st.write("The following visualization delineates how individual clinical parameters influenced the diagnostic output.")

            explainer = get_shap_explainer(model)
            # Pre-processing Gender for SHAP if model requires numeric (check your training)
            # If your model needs numeric Gender, add mapping here.
            
            shap_values = explainer(input_df)
            
            # Rendering SHAP Waterfall Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # If binary classification, we use index [0] for the specific prediction
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            
            # Professional plot styling
            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            st.pyplot(fig)
            
            with st.expander("Interpret XAI Data"):
                st.write("""
                * **Red Bars:** Biomarkers that increased the risk of a 'Pathological' diagnosis.
                * **Blue Bars:** Biomarkers that decreased the risk, pushing the result toward 'Normal'.
                * **Base Value:** The average prediction probability across the entire research dataset.
                """)

        except Exception as e:
            st.error(f"Computation Error: {e}")
    else:
        st.warning("Model core not detected. Please verify 'liver_final_model.pkl' presence.")

# Footer Credits
st.markdown(f"""
    <div class="footer-credits">
        <strong>Aarish Ali</strong> (GQ2864)<br>
        Lead Researcher | HepaScan Initiative<br>
        <span style="font-size:10px;">Dept. of Statistics & Operations Research, A.M.U</span>
    </div>
    """, unsafe_allow_html=True)