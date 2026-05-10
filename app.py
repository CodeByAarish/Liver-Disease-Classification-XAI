import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 1. CORE FUNCTIONAL DEFINITIONS ---
def cap_outliers(df):
    return df

@st.cache_resource
def load_diagnostic_model():
    try:
        with open('liver_final_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Critical System Failure: {e}")
        return None

# --- 2. GLOBAL UI CONFIGURATION ---
st.set_page_config(page_title="HepaScan | Diagnostic Suite", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .main { background-color: #0b0e14; font-family: 'Inter', sans-serif; }
    .main-title { color: #ffffff; font-size: 3rem; font-weight: 800; text-align: center; text-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    .subtitle { color: #94a3b8; text-align: center; font-size: 1.1rem; margin-bottom: 30px; }
    
    /* Red Glow Shadow Button */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 4.5em;
        background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);
        color: white; font-weight: 800; border: none;
        transition: all 0.4s ease; text-transform: uppercase;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.9);
    }

    .range-box { color: #10b981; font-size: 0.85rem; font-weight: 600; margin-top: -15px; margin-bottom: 10px;}
    .desc-box { color: #94a3b8; font-size: 0.8rem; font-style: italic; margin-bottom: 5px;}
    .med-card { background: #161b22; padding: 20px; border-radius: 15px; border-left: 5px solid #ef4444; margin-bottom: 20px; }
    
    .footer-credits {
        position: fixed; right: 20px; bottom: 20px; text-align: right;
        background: rgba(22, 27, 34, 0.9); padding: 15px; border-radius: 12px;
        border: 1px solid #ef4444; font-size: 12px; color: #8b949e; z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ HepaScan</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Hepatic Diagnostic Framework & Predictive Analytics</p>', unsafe_allow_html=True)

model = load_diagnostic_model()

# --- 3. CLINICAL DATA ENTRY ---
with st.container():
    t1, t2 = st.tabs(["📊 Patient Biometrics", "🧪 Laboratory Biomarkers"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (Years)", 1, 110, 30)
            st.markdown('<p class="desc-box">Biological aging affects liver recovery speed.</p>', unsafe_allow_html=True)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c2:
            bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0)
            st.markdown('<p class="desc-box">Measures bile pigment.</p><p class="range-box">Range: 0.1–1.2</p>', unsafe_allow_html=True)
            dbilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3)
            st.markdown('<p class="desc-box">Processed bile pigment.</p><p class="range-box">Range: 0.0–0.3</p>', unsafe_allow_html=True)

    with t2:
        col1, col2, col3 = st.columns(3)
        with col1:
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0)
            st.markdown('<p class="range-box">Range: 44–147</p>', unsafe_allow_html=True)
            ag_ratio = st.number_input("A/G Ratio", value=1.1)
            st.markdown('<p class="range-box">Range: 1.1–2.5</p>', unsafe_allow_html=True)
        with col2:
            sgpt = st.number_input("SGPT / ALT (U/L)", value=35.0)
            st.markdown('<p class="range-box">Range: 7–56</p>', unsafe_allow_html=True)
            sgot = st.number_input("SGOT / AST (U/L)", value=40.0)
            st.markdown('<p class="range-box">Range: 10–40</p>', unsafe_allow_html=True)
        with col3:
            total_proteins = st.number_input("Total Proteins (g/dL)", value=6.8)
            st.markdown('<p class="range-box">Range: 6.0–8.3</p>', unsafe_allow_html=True)
            albumin = st.number_input("Albumin (g/dL)", value=3.5)
            st.markdown('<p class="range-box">Range: 3.4–5.4</p>', unsafe_allow_html=True)

# Normal values for charts
normal_vals = {'Bilirubin': 0.6, 'DBilirubin': 0.15, 'ALP': 95, 'ALT': 31, 'AST': 25, 'Protein': 7.1, 'Albumin': 4.4}
user_vals = {'Bilirubin': bilirubin, 'DBilirubin': dbilirubin, 'ALP': alk_phos, 'ALT': sgpt, 'AST': sgot, 'Protein': total_proteins, 'Albumin': albumin}

# --- 4. PREDICTION & ANALYTICS ---
if st.button("Execute Diagnostic Analysis"):
    if model:
        try:
            data_dict = {'Age': [age], 'Gender': [gender], 'Total_Bilirubin': [bilirubin], 'Direct_Bilirubin': [dbilirubin], 'Alkaline_Phosphotase': [alk_phos], 'SGPT_ALT': [sgpt], 'SGOT_AST': [sgot], 'Total_Protiens': [total_proteins], 'Albumin': [albumin], 'A/G_Ratio': [ag_ratio], 'Globulin': [total_proteins-albumin]}
            input_df = pd.DataFrame(data_dict)
            
            prob = model.predict_proba(input_df)[0][1]
            st.markdown(f"### Assessment Report: {'🚩 Pathological' if prob > 0.5 else '✅ Normal'}")
            st.metric("Systemic Risk Index", f"{round(prob*100, 2)}%")

            # --- VISUALIZATIONS ---
            c1, c2 = st.columns(2)
            
            with c1:
                # Stacked Bar Chart
                fig_bar = go.Figure(data=[
                    go.Bar(name='Normal Baseline', x=list(normal_vals.keys()), y=list(normal_vals.values()), marker_color='#10b981'),
                    go.Bar(name='Patient Input', x=list(user_vals.keys()), y=list(user_vals.values()), marker_color='#ef4444')
                ])
                fig_bar.update_layout(title="Normal vs Patient Comparison", barmode='group', template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_dict=True)

            with c2:
                # Pie Chart
                fig_pie = go.Figure(data=[go.Pie(labels=list(user_vals.keys()), values=list(user_vals.values()), hole=.4)])
                fig_pie.update_layout(title="Biomarker Proportion", template="plotly_dark")
                st.plotly_chart(fig_pie)

            # Line Graph
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=list(user_vals.keys()), y=list(normal_vals.values()), mode='lines+markers', name='Normal Line', line=dict(color='#10b981', dash='dash')))
            fig_line.add_trace(go.Scatter(x=list(user_vals.keys()), y=list(user_vals.values()), mode='lines+markers', name='Patient Line', line=dict(color='#ef4444', width=3)))
            fig_line.update_layout(title="Variance Analysis Line Graph", template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)

            # --- STRONG SHAP XAI ---
            st.markdown("---")
            actual_model = model.steps[-1][1]
            preprocessor = model[:-1]
            transformed_x = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer(transformed_x)
            
            st.subheader("🔬 AI Logic Attribution")
            fig_shap, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white')
            st.pyplot(fig_shap)

            # --- SMART ADVISOR ---
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            top_idx = np.argmax(np.abs(shap_values.values[0]))
            top_feature = input_df.columns[top_idx]
            
            st.write(f"### 👨‍⚕️ Advisor: Focused on **{top_feature}**")
            if prob > 0.5:
                st.write(f"**Symptoms:** High levels of {top_feature} may cause fatigue, yellowing of skin/eyes, or abdominal swelling.")
                st.write("**Diet:** Strictly avoid alcohol, fried spicy foods, and excess salt. Eat boiled vegetables and drink warm water.")
                st.write("**Specialist:** Consult a **Hepatologist** at JNMC/AMU immediately for a FibroScan.")
            else:
                st.write("Results are stable. Maintain a high-fiber diet and regular exercise.")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e: st.error(f"Error: {e}")

st.markdown(f"""
    <style>
    .footer-credits {{
        position: fixed;
        right: 20px;
        bottom: 20px;
        text-align: right;
        background: rgba(22, 27, 34, 0.85); /* Transparent Dark */
        padding: 15px 25px;
        border-radius: 15px;
        border: 1px solid #ef4444; /* Red border to match your theme */
        font-size: 13px;
        color: #8b949e;
        backdrop-filter: blur(10px); /* Glassmorphism effect */
        z-index: 1000;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    </style>
    <div class="footer-credits">
        <strong style="color:white;">Aarish Ali</strong> (GQ2864)<br>
        Lead Researcher | <b>HepaScan AI</b><br>
        <span style="font-size:11px;">Dept. of Statistics & Operations Research<br>Aligarh Muslim University (A.M.U)</span>
    </div>
    """, unsafe_allow_html=True)