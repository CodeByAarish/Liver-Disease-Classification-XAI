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

# --- 2. GLOBAL UI CONFIGURATION & ANIMATED CSS ---
st.set_page_config(page_title="HepaScan | Diagnostic Suite", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .main { background-color: #0b0e14; font-family: 'Inter', sans-serif; color: white; }
    
    /* Title Animation */
    .main-title { 
        color: #ffffff; font-size: 3.5rem; font-weight: 800; text-align: center; 
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        animation: pulse 3s infinite;
    }
    @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }

    /* Dynamic Moving Cards for Variables */
    div[data-testid="stVerticalBlock"] > div:has(div.variable-card) {
        transition: transform 0.3s ease;
    }
    .variable-card {
        background: #161b22; padding: 20px; border-radius: 15px;
        border: 1px solid #30363d; margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .variable-card:hover {
        transform: translateY(-5px);
        border-color: #ef4444;
        box-shadow: 0 10px 20px rgba(239, 68, 68, 0.2);
    }

    /* Red Glow Moving Button */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 5em;
        background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);
        color: white; font-weight: 800; border: none;
        transition: all 0.4s ease; text-transform: uppercase;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
        position: relative; overflow: hidden;
    }
    .stButton>button:hover {
        box-shadow: 0 0 40px rgba(239, 68, 68, 1);
        transform: scale(1.01);
    }

    .range-box { color: #10b981; font-size: 0.85rem; font-weight: 700; }
    .desc-box { color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 8px; }
    .med-card { background: #1c2128; padding: 25px; border-radius: 20px; border-left: 8px solid #ef4444; margin-top: 20px; }
    
    .footer-credits {
        position: fixed; right: 20px; bottom: 20px; text-align: right;
        background: rgba(22, 27, 34, 0.9); padding: 15px 25px; border-radius: 15px;
        border: 1px solid #ef4444; font-size: 13px; color: #8b949e; z-index: 1000;
        backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ HepaScan AI</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8;">Neural Diagnostic Framework for Hepatic Pathology</p>', unsafe_allow_html=True)

model = load_diagnostic_model()

# --- 3. DYNAMIC DATA ENTRY ---
with st.container():
    t1, t2 = st.tabs(["📊 Patient Core", "🧪 Biochemical Panel"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            age = st.number_input("Patient Age", 1, 110, 30)
            st.markdown('<p class="desc-box">Controls metabolic baseline. Higher age may indicate slower toxin clearance.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c2:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            bilirubin = st.number_input("Total Bilirubin", value=1.0)
            st.markdown('<p class="desc-box">Indicator of RBC breakdown. Excess causes Jaundice.</p><p class="range-box">Normal: 0.1–1.2 mg/dL</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            dbilirubin = st.number_input("Direct Bilirubin", value=0.3)
            st.markdown('<p class="desc-box">Processed bile. High levels suggest bile duct blockage.</p><p class="range-box">Normal: 0.0–0.3 mg/dL</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            alk_phos = st.number_input("Alkaline Phosphatase (ALP)", value=100.0)
            st.markdown('<p class="desc-box">Enzyme found in bile ducts. Elevation suggests cholestasis.</p><p class="range-box">Normal: 44–147 U/L</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", value=1.1)
            st.markdown('<p class="desc-box">A vital check of protein balance.</p><p class="range-box">Normal: 1.1–2.5</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            sgpt = st.number_input("SGPT (ALT)", value=35.0)
            st.markdown('<p class="desc-box">Primary marker of liver cell death/inflammation.</p><p class="range-box">Normal: 7–56 U/L</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            sgot = st.number_input("SGOT (AST)", value=40.0)
            st.markdown('<p class="desc-box">Enzyme released during heart or liver tissue damage.</p><p class="range-box">Normal: 10–40 U/L</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            total_proteins = st.number_input("Total Serum Proteins", value=6.8)
            st.markdown('<p class="desc-box">Total protein mass in blood.</p><p class="range-box">Normal: 6.0–8.3 g/dL</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            albumin = st.number_input("Serum Albumin", value=3.5)
            st.markdown('<p class="desc-box">Liver-made protein. Essential for fluid balance.</p><p class="range-box">Normal: 3.4–5.4 g/dL</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Comparison Data
features = ['Bilirubin', 'DBilirubin', 'ALP', 'ALT', 'AST', 'Protein', 'Albumin']
norms = [0.6, 0.15, 95, 31, 25, 7.1, 4.4]
users = [bilirubin, dbilirubin, alk_phos, sgpt, sgot, total_proteins, albumin]

# --- 4. ENGINE & ADVANCED ANALYTICS ---
if st.button("🚀 EXECUTE NEURAL DIAGNOSIS"):
    if model:
        try:
            data_dict = {'Age': [age], 'Gender': [gender], 'Total_Bilirubin': [bilirubin], 'Direct_Bilirubin': [dbilirubin], 'Alkaline_Phosphotase': [alk_phos], 'SGPT_ALT': [sgpt], 'SGOT_AST': [sgot], 'Total_Protiens': [total_proteins], 'Albumin': [albumin], 'A/G_Ratio': [ag_ratio], 'Globulin': [total_proteins-albumin]}
            input_df = pd.DataFrame(data_dict)
            
            prob = model.predict_proba(input_df)[0][1]
            
            # --- RESULTS UI ---
            st.markdown("---")
            c_res1, c_res2 = st.columns([2, 1])
            with c_res1:
                if prob > 0.5:
                    st.error(f"### 🚩 PATHOLOGICAL RISK DETECTED\n**Index:** {round(prob*100, 2)}%\nConfidence levels indicate high correlation with hepatic disease patterns.")
                else:
                    st.success(f"### ✅ PHYSIOLOGICAL STABILITY DETECTED\n**Index:** {round(prob*100, 2)}%\nBiometric markers align with homeostatic standards.")

            with c_res2:
                # Radar Chart for Balance
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=norms, theta=features, fill='toself', name='Normal Baseline', line_color='#10b981'))
                fig_radar.add_trace(go.Scatterpolar(r=users, theta=features, fill='toself', name='Patient Profile', line_color='#ef4444'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(users)+10])), template="plotly_dark", title="Clinical Variance Radar")
                st.plotly_chart(fig_radar, use_container_width=True)

            # --- DETAILED INSIGHT CHARTS ---
            st.markdown("### 📈 Biomarker Variance Analytics")
            v1, v2 = st.columns(2)
            with v1:
                # Grouped Bar
                fig_bar = go.Figure(data=[
                    go.Bar(name='Standard', x=features, y=norms, marker_color='#10b981'),
                    go.Bar(name='Patient', x=features, y=users, marker_color='#ef4444')
                ])
                fig_bar.update_layout(barmode='group', template="plotly_dark", title="Standard vs Patient Comparison")
                st.plotly_chart(fig_bar, use_container_width=True)
            with v2:
                # Line Variance
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=features, y=norms, mode='lines+markers', name='Normal Threshold', line=dict(color='#10b981', dash='dash')))
                fig_line.add_trace(go.Scatter(x=features, y=users, mode='lines+markers', name='Patient Actual', line=dict(color='#ef4444', width=4)))
                fig_line.update_layout(template="plotly_dark", title="Deviation Line Analysis")
                st.plotly_chart(fig_line, use_container_width=True)

            # --- DYNAMIC SHAP EXPLANATION ---
            st.markdown("---")
            st.subheader("🔬 AI Decision Core (XAI)")
            actual_model = model.steps[-1][1]
            preprocessor = model[:-1]
            transformed_x = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer(transformed_x)
            
            fig_shap, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white')
            st.pyplot(fig_shap)

            # --- DYNAMIC ADVISOR LOGIC ---
            top_shaps = np.argsort(shap_values.values[0])[-2:][::-1]
            f1, f2 = input_df.columns[top_shaps[0]], input_df.columns[top_shaps[1]]

            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.write(f"### 👨‍⚕️ Intelligent Clinical Interpretation")
            st.write(f"The AI's decision is primarily driven by **{f1}** and **{f2}**.")
            
            if prob > 0.5:
                st.markdown("#### ⚠️ Potential Clinical Progression")
                if "Bilirubin" in f1 or "Bilirubin" in f2:
                    st.write("- **Analysis:** High Bilirubin suggests hepatic processing failure. Symptoms include **Jaundice**, dark urine, and pale stools.")
                if "ALT" in f1 or "AST" in f2:
                    st.write("- **Analysis:** Elevated enzymes suggest active hepatocyte (liver cell) death. You may experience **Right-side abdominal pain** and chronic fatigue.")
                
                st.markdown("#### 🥗 Precision Nutrition & Lifestyle")
                st.write("- **Target:** Reduce liver workload. Zero alcohol, low sodium, and high antioxidant intake (Beetroot, Berries).")
                st.markdown("#### 🏥 Medical Action Plan")
                st.write("Consult a **Consultant Hepatologist** immediately. Recommend: Liver Ultrasound + Serum Viral Markers.")
            else:
                st.write("✅ **Maintenance:** Liver function appears optimal. Keep a high-protein, low-sugar diet to support cellular health.")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e: st.error(f"Computation Error: {e}")

# --- FOOTER ---
st.markdown(f"""
    <div class="footer-credits">
        <strong style="color:white;">Aarish Ali</strong> (GQ2864) <24DSMSA113><br>
        Lead AI Researcher | <b>HepaScan Framework</b><br>
        <span style="font-size:11px;">Dept. of Statistics & Operations Research<br>Aligarh Muslim University (A.M.U)</span>
    </div>
    """, unsafe_allow_html=True)
