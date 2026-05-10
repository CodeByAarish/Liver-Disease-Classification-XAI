import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
    .main { background-color: #0b0e14; font-family: 'Inter', sans-serif; color: white; }
    .main-title { color: #ffffff; font-size: 3.5rem; font-weight: 800; text-align: center; text-shadow: 0 0 20px rgba(239, 68, 68, 0.5); animation: pulse 3s infinite; }
    @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }

    /* Variable Card Styling - PURE CSS TRANSITIONS */
    .variable-card {
        background: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .variable-card:hover { transform: translateY(-8px); border-color: #ef4444; box-shadow: 0 10px 25px rgba(239, 68, 68, 0.25); }

    .stButton>button {
        width: 100%; border-radius: 12px; height: 5em; background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);
        color: white; font-weight: 800; border: none; transition: all 0.4s ease; box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
    }
    .stButton>button:hover { box-shadow: 0 0 40px rgba(239, 68, 68, 1); transform: scale(1.02); }

    .range-box { color: #10b981; font-size: 0.9rem; font-weight: 700; margin-top: 5px; display: block;}
    .desc-box { color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-top: 5px;}
    .med-card { background: #1c2128; padding: 30px; border-radius: 20px; border-left: 10px solid #ef4444; margin: 20px 0; }
    
    .footer-credits {
        position: fixed; right: 20px; bottom: 20px; text-align: right; background: rgba(22, 27, 34, 0.9);
        padding: 15px 25px; border-radius: 15px; border: 1px solid #ef4444; font-size: 13px; color: #8b949e;
        z-index: 1000; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ HepaScan AI</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8; font-size: 1.2rem;">Advanced Neural Diagnostic Framework for Hepatic Pathology</p>', unsafe_allow_html=True)

model = load_diagnostic_model()

# --- 3. DYNAMIC DATA ENTRY (Cards with descriptions) ---
with st.container():
    t1, t2 = st.tabs(["👤 Patient Demographics", "🧪 Biochemical Markers"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            age = st.number_input("Patient Age", 1, 110, 30)
            st.markdown('<span class="desc-box">Determines metabolic baseline. Older age may correlate with slower liver regeneration.</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c2:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0)
            st.markdown('<span class="desc-box">Waste product from RBC breakdown. High levels indicate Jaundice or Liver obstruction.</span><span class="range-box">Normal: 0.1–1.2</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            dbilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3)
            st.markdown('<span class="desc-box">Conjugated bile. High levels suggest bile duct blockage.</span><span class="range-box">Normal: 0.0–0.3</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0)
            st.markdown('<span class="desc-box">Bile duct enzyme. Elevated in cholestasis or bone growth.</span><span class="range-box">Normal: 44–147</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            sgpt = st.number_input("SGPT / ALT (U/L)", value=35.0)
            st.markdown('<span class="desc-box">Most specific liver enzyme. High levels mean liver cell damage.</span><span class="range-box">Normal: 7–56</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            sgot = st.number_input("SGOT / AST (U/L)", value=40.0)
            st.markdown('<span class="desc-box">Released during tissue damage. Used to detect inflammation.</span><span class="range-box">Normal: 10–40</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            total_proteins = st.number_input("Total Protein (g/dL)", value=6.8)
            st.markdown('<span class="desc-box">Measures Albumin and Globulin. High indicates infection.</span><span class="range-box">Normal: 6.0–8.3</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="variable-card">', unsafe_allow_html=True)
            albumin = st.number_input("Albumin (g/dL)", value=3.5)
            st.markdown('<span class="desc-box">Main protein produced by liver. Low indicates chronic disease.</span><span class="range-box">Normal: 3.4–5.4</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Calculated Variable
ag_ratio = albumin / ((total_proteins - albumin) if (total_proteins - albumin) != 0 else 1)
globulin = total_proteins - albumin

# --- 4. ENGINE & ADVANCED ANALYTICS ---
if st.button("🚀 EXECUTE NEURAL DIAGNOSIS"):
    if model:
        try:
            # Data Mapping
            data_dict = {'Age': [age], 'Gender': [gender], 'Total_Bilirubin': [bilirubin], 'Direct_Bilirubin': [dbilirubin], 
                         'Alkaline_Phosphotase': [alk_phos], 'SGPT_ALT': [sgpt], 'SGOT_AST': [sgot], 
                         'Total_Protiens': [total_proteins], 'Albumin': [albumin], 'A/G_Ratio': [ag_ratio], 'Globulin': [globulin]}
            input_df = pd.DataFrame(data_dict)
            
            # Diagnostic Execution
            prob = model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            # Result Metric Alignment
            m1, m2 = st.columns([1, 1])
            with m1:
                status = "🚩 PATHOLOGICAL" if prob > 0.5 else "✅ PHYSIOLOGICAL STABILITY"
                st.markdown(f"### Diagnostic Result: {status}")
            with m2:
                st.metric("Systemic Risk Index", f"{round(prob*100, 2)}%", delta="- Safe" if prob < 0.5 else "Risk Identified")

            # --- 5 GRAPHS SUITE ---
            st.markdown("### 📈 Automated Clinical Visualization")
            
            # Row 1: Comparison Graphs
            g1, g2 = st.columns(2)
            with g1:
                # 1. Radar Chart (Clinical Variance)
                features = ['Bilirubin', 'DBilirubin', 'ALP', 'ALT', 'AST', 'Protein', 'Albumin']
                norms = [0.6, 0.15, 95, 31, 25, 7.1, 4.4]
                users = [bilirubin, dbilirubin, alk_phos, sgpt, sgot, total_proteins, albumin]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=norms, theta=features, fill='toself', name='Healthy Standard', line_color='#10b981'))
                fig_radar.add_trace(go.Scatterpolar(r=users, theta=features, fill='toself', name='Patient Input', line_color='#ef4444'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", title="Clinical Variance Radar")
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with g2:
                # 2. Risk Contribution Pie (Value Proportion)
                fig_pie = px.pie(values=users, names=features, hole=0.4, title="Marker Concentration Breakdown", color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

            # Row 2: Analysis Graphs
            g3, g4 = st.columns(2)
            with g3:
                # 3. Stacked Grouped Comparison
                fig_bar = go.Figure(data=[
                    go.Bar(name='Normal Threshold', x=features, y=norms, marker_color='#10b981'),
                    go.Bar(name='Patient Input', x=features, y=users, marker_color='#ef4444')
                ])
                fig_bar.update_layout(barmode='group', template="plotly_dark", title="Standard vs Patient Bar Analysis")
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with g4:
                # 4. AST/ALT Ratio Trend (Specific Diagnostic Insight)
                ratio = sgot/sgpt if sgpt != 0 else 0
                fig_ratio = go.Figure(go.Indicator(mode = "gauge+number", value = ratio, title = {'text': "De Ritis Ratio (AST/ALT)"},
                    gauge = {'axis': {'range': [None, 3]}, 'bar': {'color': "#ef4444"}, 'steps': [{'range': [0, 1], 'color': "green"}, {'range': [1, 2], 'color': "orange"}]}))
                fig_ratio.update_layout(template="plotly_dark")
                st.plotly_chart(fig_ratio, use_container_width=True)

            # 5. Deviation Line Graph
            st.markdown("#### 5. Biomarker Deviation Profile")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=features, y=norms, mode='lines+markers', name='Healthy Range', line=dict(color='#10b981', dash='dash')))
            fig_line.add_trace(go.Scatter(x=features, y=users, mode='lines+markers', name='Patient Actual', line=dict(color='#ef4444', width=4)))
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)

            # --- SHAP Decision Core (FIXED FEATURE NAMES) ---
            st.markdown("---")
            st.subheader("🔬 AI Decision Core (Explainable AI)")
            
            # Extract names properly
            actual_model = model.steps[-1][1]
            preprocessor = model[:-1]
            transformed_x = preprocessor.transform(input_df)
            
            # Map column names back
            feature_names = input_df.columns
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer(transformed_x)
            
            fig_shap, ax = plt.subplots(figsize=(10, 4))
            # Critical: assign feature names to the SHAP explanation object
            shap_exp = shap.Explanation(values=shap_values.values[0], base_values=explainer.expected_value, 
                                        data=transformed_x[0], feature_names=feature_names)
            shap.plots.waterfall(shap_exp, show=False)
            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white')
            st.pyplot(fig_shap)

            # --- DYNAMIC INTELLIGENT PRESCRIPTION ---
            st.markdown('<div class="med-card">', unsafe_allow_html=True)
            st.write("### 👨‍⚕️ Intelligent Clinical Interpretation & Prescription")
            
            # Dynamic Logic per Patient
            high_markers = [features[i] for i, v in enumerate(users) if v > norms[i]*1.5]
            
            if prob > 0.5:
                st.write(f"**Primary Risk Drivers Identified:** {', '.join(high_markers) if high_markers else 'General systemic instability'}")
                st.markdown("#### 🚩 Condition Analysis")
                if "ALT" in high_markers or "AST" in high_markers:
                    st.write("- Critical hepatocyte (liver cell) inflammation detected. This signifies active liver injury.")
                if "Bilirubin" in high_markers:
                    st.write("- Impaired bile excretion or excessive RBC breakdown detected (Biliary Obstruction risk).")
                
                st.markdown("#### 💊 Actionable Prescription")
                st.write("- **Medical Intervention:** Immediate consultation with a **Hepatologist** at JNMC/AMU.")
                st.write("- **Diagnostic Recommendation:** Serum Viral Markers (HBsAg, HCV) and an Abdominal Ultrasound (USG).")
                st.write("- **Lifestyle Protocol:** Absolute cessation of alcohol. Limit sodium to <2000mg/day. Increase antioxidant intake.")
                st.write("- **Emergency Symptoms:** Seek help if you experience jaundice, confusion (encephalopathy), or dark-colored urine.")
            else:
                st.write("✅ **Stability Confirmed:** Your biomarkers are currently within safe physiological limits.")
                st.write("- **Recommendation:** Continue annual screening. Maintain a high-fiber, low-fructose diet to prevent Non-Alcoholic Fatty Liver (NAFLD).")
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
