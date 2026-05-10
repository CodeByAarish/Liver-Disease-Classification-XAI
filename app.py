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

/* MAIN */

.main {
    background-color: #0b0e14;
    font-family: 'Inter', sans-serif;
    color: white;
}

/* TITLE */

.main-title {
    color: #ffffff;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* VARIABLE CARD */

.variable-card {
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #30363d;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.variable-card:hover {
    transform: translateY(-8px);
    border-color: #ef4444;
    box-shadow: 0 10px 25px rgba(239, 68, 68, 0.25);
}

/* BUTTON */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 5em;
    background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);
    color: white;
    font-weight: 800;
    border: none;
    transition: all 0.4s ease;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.5);
}

.stButton > button:hover {
    box-shadow: 0 0 40px rgba(239, 68, 68, 1);
    transform: scale(1.02);
}

/* ========================= */
/* ALL INPUT BOXES STYLING */
/* ========================= */

.stNumberInput input,
.stTextInput input,
.stTextArea textarea,
.stDateInput input,
.stTimeInput input,
.stSelectbox div[data-baseweb="select"] > div {

    background-color: #161b22 !important;
    color: white !important;

    border: 2px solid #30363d !important;
    border-radius: 14px !important;

    transition: all 0.3s ease !important;

    box-shadow: 0 0 8px rgba(239, 68, 68, 0.15) !important;
}

/* HOVER EFFECT */

.stNumberInput input:hover,
.stTextInput input:hover,
.stTextArea textarea:hover,
.stDateInput input:hover,
.stTimeInput input:hover,
.stSelectbox div[data-baseweb="select"] > div:hover {

    border-color: #ef4444 !important;

    box-shadow:
        0 0 10px rgba(239, 68, 68, 0.4),
        0 0 20px rgba(239, 68, 68, 0.25) !important;

    transform: translateY(-2px);
}

/* FOCUS EFFECT */

.stNumberInput input:focus,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stDateInput input:focus,
.stTimeInput input:focus {

    border-color: #ef4444 !important;

    box-shadow:
        0 0 12px rgba(239, 68, 68, 0.6),
        0 0 30px rgba(239, 68, 68, 0.35) !important;

    transform: scale(1.01);
}

/* LABELS */

label {
    color: white !important;
    font-weight: 700 !important;
}

/* RANGE BOX */

.range-box {
    color: #10b981;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 5px;
    display: block;
}

/* DESCRIPTION */

.desc-box {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.4;
    margin-top: 5px;
}

/* MEDICAL CARD */

.med-card {
    background: #1c2128;
    padding: 30px;
    border-radius: 20px;
    border-left: 10px solid #ef4444;
    margin: 20px 0;
}

/* ========================= */
/* METRIC CARD STYLING */
/* ========================= */

[data-testid="stMetric"] {
    background-color: #000000 !important;
    border: 2px solid #ef4444 !important;
    padding: 20px !important;
    border-radius: 18px !important;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4) !important;
}

/* METRIC LABEL */

[data-testid="stMetricLabel"] {
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* METRIC VALUE */

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    text-shadow: 0 0 12px rgba(239,68,68,0.4);
}

/* RISK IDENTIFIED */

[data-testid="stMetricDelta"] {
    color: #ff0000 !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    text-shadow: 0 0 10px rgba(255,0,0,0.8);
}

/* FOOTER */

.footer-credits {
    position: fixed;
    right: 20px;
    bottom: 20px;
    text-align: right;
    background: rgba(22, 27, 34, 0.9);
    padding: 15px 25px;
    border-radius: 15px;
    border: 1px solid #ef4444;
    font-size: 13px;
    color: #8b949e;
    z-index: 1000;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
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
            
            age = st.number_input("Patient Age", 1, 110, 30)
            st.markdown('<span class="desc-box">Determines metabolic baseline. Older age may correlate with slower liver regeneration.</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c2:
            
            bilirubin = st.number_input("Total Bilirubin (mg/dL)", value=1.0)
            st.markdown('<span class="desc-box">Waste product from RBC breakdown. High levels indicate Jaundice or Liver obstruction.</span><span class="range-box">Normal: 0.1–1.2</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        col1, col2, col3 = st.columns(3)
        with col1:
           
            dbilirubin = st.number_input("Direct Bilirubin (mg/dL)", value=0.3)
            st.markdown('<span class="desc-box">Conjugated bile. High levels suggest bile duct blockage.</span><span class="range-box">Normal: 0.0–0.3 mg/dL</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            alk_phos = st.number_input("Alkaline Phosphatase (U/L)", value=100.0)
            st.markdown('<span class="desc-box">Bile duct enzyme. Elevated in cholestasis or bone growth.</span><span class="range-box">Normal: 44–147 U/L</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            
            sgpt = st.number_input("SGPT / ALT (U/L)", value=35.0)
            st.markdown('<span class="desc-box">Most specific liver enzyme. High levels mean liver cell damage.</span><span class="range-box">Normal: 7–56 U/L</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            sgot = st.number_input("SGOT / AST (U/L)", value=40.0)
            st.markdown('<span class="desc-box">Released during tissue damage. Used to detect inflammation.</span><span class="range-box">Normal: 10–40 U/L</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            
            total_proteins = st.number_input("Total Protein (g/dL)", value=6.8)
            st.markdown('<span class="desc-box">Measures Albumin and Globulin. High indicates infection.</span><span class="range-box">Normal: 6.0–8.3 g/dL</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            albumin = st.number_input("Albumin (g/dL)", value=3.5)
            st.markdown('<span class="desc-box">Main protein produced by liver. Low indicates chronic disease.</span><span class="range-box">Normal: 3.4–5.4 g/dL</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Calculated Variable
ag_ratio = albumin / ((total_proteins - albumin) if (total_proteins - albumin) != 0 else 1)
globulin = total_proteins - albumin

# --- 4. ENGINE & ADVANCED ANALYTICS ---
if st.button("🚀 EXECUTE NEURAL DIAGNOSIS"):
    if model:
        try:
            # Data Mapping
            data_dict = {
                'Age': [age],
                'Gender': [gender],
                'Total_Bilirubin': [bilirubin],
                'Direct_Bilirubin': [dbilirubin],
                'Alkaline_Phosphotase': [alk_phos],
                'SGPT_ALT': [sgpt],
                'SGOT_AST': [sgot],
                'Total_Protiens': [total_proteins],
                'Albumin': [albumin],
                'A/G_Ratio': [ag_ratio],
                'Globulin': [globulin]
            }

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
                st.metric(
                    "Systemic Risk Index",
                    f"{round(prob*100, 2)}%",
                    delta="- Safe" if prob < 0.5 else "Risk Identified"
                )

            # --- 5 GRAPHS SUITE ---
            st.markdown("### 📈 Automated Clinical Visualization")

            # Row 1
            g1, g2 = st.columns(2)

            with g1:
                # Radar Chart
                features = ['Bilirubin', 'DBilirubin', 'ALP', 'ALT', 'AST', 'Protein', 'Albumin']
                norms = [0.6, 0.15, 95, 31, 25, 7.1, 4.4]

                users = [
                    bilirubin,
                    dbilirubin,
                    alk_phos,
                    sgpt,
                    sgot,
                    total_proteins,
                    albumin
                ]

                fig_radar = go.Figure()

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=norms,
                        theta=features,
                        fill='toself',
                        name='Healthy Standard',
                        line_color='#10b981'
                    )
                )

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=users,
                        theta=features,
                        fill='toself',
                        name='Patient Input',
                        line_color='#ef4444'
                    )
                )

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    template="plotly_dark",
                    title="Clinical Variance Radar"
                )

                st.plotly_chart(fig_radar, use_container_width=True)

            with g2:
                # Pie Chart
                fig_pie = px.pie(
                    values=users,
                    names=features,
                    hole=0.4,
                    title="Marker Concentration Breakdown",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )

                fig_pie.update_layout(template="plotly_dark")

                st.plotly_chart(fig_pie, use_container_width=True)

            # Row 2
            g3, g4 = st.columns(2)

            with g3:
                # Bar Chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        name='Normal Threshold',
                        x=features,
                        y=norms,
                        marker_color='#10b981'
                    ),
                    go.Bar(
                        name='Patient Input',
                        x=features,
                        y=users,
                        marker_color='#ef4444'
                    )
                ])

                fig_bar.update_layout(
                    barmode='group',
                    template="plotly_dark",
                    title="Standard vs Patient Bar Analysis"
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            with g4:
                # AST ALT Gauge
                ratio = sgot / sgpt if sgpt != 0 else 0

                fig_ratio = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=ratio,
                        title={'text': "De Ritis Ratio (AST/ALT)"},
                        gauge={
                            'axis': {'range': [None, 3]},
                            'bar': {'color': "#ef4444"},
                            'steps': [
                                {'range': [0, 1], 'color': "green"},
                                {'range': [1, 2], 'color': "orange"}
                            ]
                        }
                    )
                )

                fig_ratio.update_layout(template="plotly_dark")

                st.plotly_chart(fig_ratio, use_container_width=True)

            # Deviation Line Graph
            st.markdown("#### 5. Biomarker Deviation Profile")

            fig_line = go.Figure()

            fig_line.add_trace(
                go.Scatter(
                    x=features,
                    y=norms,
                    mode='lines+markers',
                    name='Healthy Range',
                    line=dict(color='#10b981', dash='dash')
                )
            )

            fig_line.add_trace(
                go.Scatter(
                    x=features,
                    y=users,
                    mode='lines+markers',
                    name='Patient Actual',
                    line=dict(color='#ef4444', width=4)
                )
            )

            fig_line.update_layout(template="plotly_dark")

            st.plotly_chart(fig_line, use_container_width=True)

            # --- SHAP Decision Core ---
            st.markdown("---")
            st.subheader("🔬 AI Decision Core (Explainable AI)")

            actual_model = model.steps[-1][1]
            preprocessor = model[:-1]

            transformed_x = preprocessor.transform(input_df)

            feature_names = input_df.columns

            explainer = shap.TreeExplainer(actual_model)

            shap_values = explainer(transformed_x)

            fig_shap, ax = plt.subplots(figsize=(10, 4))

            shap_exp = shap.Explanation(
                values=shap_values.values[0],
                base_values=explainer.expected_value,
                data=transformed_x[0],
                feature_names=feature_names
            )

            shap.plots.waterfall(shap_exp, show=False)

            plt.gcf().set_facecolor('#0b0e14')
            ax.set_facecolor('#0b0e14')
            ax.tick_params(colors='white')

            st.pyplot(fig_shap)

            # ==========================================================
            # --- UPDATED DYNAMIC INTELLIGENT PRESCRIPTION ---
            # ==========================================================

            st.markdown('<div class="med-card">', unsafe_allow_html=True)

            st.write("### 👨‍⚕️ Intelligent Clinical Interpretation & Prescription")

            # Detect abnormal markers
            high_markers = [
                features[i]
                for i, v in enumerate(users)
                if v > norms[i] * 1.5
            ]

            if prob > 0.5:

                st.error(f"🚨 High Risk Liver Profile Detected ({round(prob*100,2)}%)")

                st.write("### 🔬 Primary Risk Drivers Identified:")

                st.write(
                    ", ".join(high_markers)
                    if high_markers
                    else "General systemic instability"
                )

                # ======================================================
                # CONDITION ANALYSIS
                # ======================================================

                st.markdown("## 🚩 Condition Analysis")

                if "ALT" in high_markers or "AST" in high_markers:

                    st.warning("""
                    Severe hepatocyte inflammation detected.

                    Possible Conditions:
                    - Viral Hepatitis
                    - Fatty Liver Disease
                    - Alcoholic Liver Injury
                    - Drug-induced Liver Toxicity
                    """)

                if "Bilirubin" in high_markers or "DBilirubin" in high_markers:

                    st.warning("""
                    Bilirubin elevation suggests impaired bile metabolism.

                    Possible Conditions:
                    - Jaundice
                    - Bile duct obstruction
                    - Gall bladder dysfunction
                    - Liver filtration impairment
                    """)

                if "ALP" in high_markers:

                    st.warning("""
                    Elevated Alkaline Phosphatase detected.

                    Possible Conditions:
                    - Cholestasis
                    - Gallstones
                    - Biliary obstruction
                    - Liver inflammation
                    """)

                if "Protein" in high_markers:

                    st.warning("""
                    Protein imbalance detected.

                    Possible Conditions:
                    - Chronic inflammation
                    - Liver synthesis dysfunction
                    - Immune activation
                    """)

                if "Albumin" in high_markers:

                    st.warning("""
                    Albumin abnormality detected.

                    Possible indications:
                    - Chronic liver disease
                    - Nutritional imbalance
                    - Reduced liver protein synthesis
                    """)

                # ======================================================
                # EXPECTED SYMPTOMS
                # ======================================================

                st.markdown("## 🤒 Expected Symptoms")

                symptoms = []

                if "Bilirubin" in high_markers:
                    symptoms.extend([
                        "Yellowing of eyes and skin (Jaundice)",
                        "Dark-colored urine",
                        "Pale stools"
                    ])

                if "ALT" in high_markers or "AST" in high_markers:
                    symptoms.extend([
                        "Fatigue and weakness",
                        "Pain in upper right abdomen",
                        "Nausea or vomiting",
                        "Muscle weakness"
                    ])

                if "ALP" in high_markers:
                    symptoms.extend([
                        "Digestive discomfort",
                        "Itchy skin",
                        "Loss of appetite"
                    ])

                if "Protein" in high_markers:
                    symptoms.extend([
                        "Body weakness",
                        "Inflammation",
                        "Fluid imbalance"
                    ])

                symptoms = list(set(symptoms))

                for s in symptoms:
                    st.write(f"• {s}")

                # ======================================================
                # FOOD RECOMMENDATIONS
                # ======================================================

                st.markdown("## 🥗 Recommended Liver Recovery Diet")

                foods = []

                if "ALT" in high_markers or "AST" in high_markers:
                    foods.extend([
                        "Green leafy vegetables",
                        "Turmeric milk",
                        "Beetroot juice",
                        "Walnuts",
                        "Olive oil"
                    ])

                if "Bilirubin" in high_markers:
                    foods.extend([
                        "Sugarcane juice",
                        "Coconut water",
                        "Papaya",
                        "Boiled vegetables",
                        "Hydration-rich fluids"
                    ])

                if "ALP" in high_markers:
                    foods.extend([
                        "Vitamin D rich foods",
                        "Egg whites",
                        "Fish",
                        "Low-fat yogurt"
                    ])

                if "Protein" in high_markers:
                    foods.extend([
                        "Lentils",
                        "Oats",
                        "Paneer",
                        "Protein-balanced meals"
                    ])

                foods = list(set(foods))

                for f in foods:
                    st.write(f"✅ {f}")

                # ======================================================
                # FOODS TO AVOID
                # ======================================================

                st.markdown("## ❌ Foods & Habits To Avoid")

                avoid = [
                    "Alcohol",
                    "Smoking",
                    "Processed foods",
                    "Deep fried food",
                    "Soft drinks",
                    "Excess sugar",
                    "High sodium intake",
                    "Fast food",
                    "Excess red meat"
                ]

                for a in avoid:
                    st.write(f"❌ {a}")

                # ======================================================
                # MEDICAL RECOMMENDATIONS
                # ======================================================

                st.markdown("## 💊 Recommended Medical Actions")

                st.write("""
                - Immediate consultation with a Hepatologist
                - Liver Function Test (LFT)
                - Serum Viral Marker Test (HBsAg, HCV)
                - Ultrasound Abdomen (USG)
                - Routine enzyme monitoring every 2-4 weeks
                - Maintain proper hydration and rest
                """)

                # ======================================================
                # RISK LEVEL
                # ======================================================

                st.markdown("## 📊 Clinical Risk Classification")

                if prob >= 0.90:
                    st.error("🔴 CRITICAL RISK LEVEL")

                elif prob >= 0.75:
                    st.warning("🟠 HIGH RISK LEVEL")

                elif prob >= 0.50:
                    st.warning("🟡 MODERATE RISK LEVEL")

                # ======================================================
                # EMERGENCY ALERTS
                # ======================================================

                st.markdown("## 🚑 Emergency Symptoms")

                st.error("""
                Seek immediate medical attention if patient experiences:
                - Severe jaundice
                - Confusion / encephalopathy
                - Blood vomiting
                - Severe abdominal swelling
                - Continuous vomiting
                - Loss of consciousness
                - Breathing difficulty
                """)

            else:

                st.success("✅ Liver biomarkers are within relatively stable physiological limits.")

                st.markdown("## 🌿 Preventive Recommendations")

                st.write("""
                - Maintain hydration
                - Exercise regularly
                - Avoid excessive oily food
                - Annual liver screening recommended
                - Maintain healthy body weight
                - Follow a balanced high-fiber diet
                """)

                st.markdown("## 🥗 Recommended Daily Foods")

                healthy_foods = [
                    "Fresh fruits",
                    "Green vegetables",
                    "Whole grains",
                    "Nuts",
                    "Curd",
                    "Lean protein",
                    "Hydration-rich drinks"
                ]

                for food in healthy_foods:
                    st.write(f"✅ {food}")

                st.markdown("## 😌 Healthy Symptoms")

                st.write("""
                - Stable energy levels
                - Normal digestion
                - Healthy appetite
                - Clear skin and eyes
                - Balanced metabolism
                """)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Computation Error: {e}")

# --- FOOTER ---
st.markdown(f"""
    <div class="footer-credits">
        <strong style="color:white;">Aarish Ali</strong> (GQ2864)<br>
        Lead AI Researcher | <b>HepaScan Framework</b><br>
        <span style="font-size:11px;">
        Dept. of Statistics & Operations Research<br>
        Aligarh Muslim University (A.M.U)
        </span>
    </div>
    """, unsafe_allow_html=True)
