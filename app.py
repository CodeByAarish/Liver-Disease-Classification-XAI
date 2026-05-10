# =========================================================
#                HEPASCAN AI - UPDATED UI
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# 1. MODEL LOADER
# =========================================================
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

# =========================================================
# 2. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HepaScan AI",
    page_icon="⚕️",
    layout="wide"
)

# =========================================================
# 3. CUSTOM CSS (UPDATED)
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN BG */
.stApp {
    background-color: #0b0e14;
    color: white;
}

/* TITLE */
.main-title {
    color: white;
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    margin-top: -20px;
    text-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
}

/* SUBTITLE */
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 35px;
}

/* CARD */
.variable-card {
    background: #161b22;
    padding: 15px;
    border-radius: 16px;
    border: 1px solid #30363d;
    margin-bottom: 12px;
    transition: all 0.3s ease-in-out;
}

/* REMOVE EMPTY SPACE */
.variable-card:hover {
    transform: translateY(-4px);
    border-color: #ef4444;
    box-shadow: 0 0 18px rgba(239, 68, 68, 0.35);
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 65px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #b91c1c 0%, #ef4444 100%);
    color: white;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
    box-shadow: 0 0 18px rgba(239, 68, 68, 0.4);
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.9);
}

/* DESCRIPTION */
.desc-box {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 5px;
}

/* RANGE */
.range-box {
    color: #10b981;
    font-size: 0.85rem;
    font-weight: 700;
}

/* RESULT CARD */
.med-card {
    background: #161b22;
    padding: 25px;
    border-radius: 18px;
    border-left: 8px solid #ef4444;
    margin-top: 20px;
}

/* FOOTER */
.footer-credits {
    position: fixed;
    right: 20px;
    bottom: 20px;
    text-align: right;
    background: rgba(22, 27, 34, 0.95);
    padding: 12px 20px;
    border-radius: 15px;
    border: 1px solid #ef4444;
    font-size: 12px;
    color: #8b949e;
    z-index: 9999;
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}

/* REMOVE STREAMLIT EXTRA SPACING */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. HEADER
# =========================================================

st.markdown(
    '<p class="main-title">⚕️ HepaScan AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Advanced Neural Diagnostic Framework for Hepatic Pathology</p>',
    unsafe_allow_html=True
)

# =========================================================
# 5. LOAD MODEL
# =========================================================

model = load_diagnostic_model()

# =========================================================
# 6. INPUT SECTION
# =========================================================

tab1, tab2 = st.tabs([
    "👤 Patient Demographics",
    "🧪 Biochemical Markers"
])

# ---------------- TAB 1 ---------------- #

with tab1:

    c1, c2 = st.columns(2)

    with c1:

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=110,
            value=30
        )

        st.markdown(
            '<div class="desc-box">Determines metabolic baseline and liver regeneration capacity.</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        gender = st.selectbox(
            "Biological Sex",
            ["Male", "Female"]
        )

    with c2:

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        bilirubin = st.number_input(
            "Total Bilirubin (mg/dL)",
            value=1.0
        )

        st.markdown(
            '<div class="desc-box">High bilirubin may indicate jaundice or liver obstruction.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 0.1 – 1.2</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2 ---------------- #

with tab2:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        dbilirubin = st.number_input(
            "Direct Bilirubin (mg/dL)",
            value=0.3
        )

        st.markdown(
            '<div class="desc-box">High values suggest bile duct blockage.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 0.0 – 0.3</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        alk_phos = st.number_input(
            "Alkaline Phosphatase (U/L)",
            value=100.0
        )

        st.markdown(
            '<div class="desc-box">Elevated during cholestasis or liver inflammation.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 44 – 147</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        sgpt = st.number_input(
            "SGPT / ALT (U/L)",
            value=35.0
        )

        st.markdown(
            '<div class="desc-box">Most specific liver damage marker.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 7 – 56</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        sgot = st.number_input(
            "SGOT / AST (U/L)",
            value=40.0
        )

        st.markdown(
            '<div class="desc-box">Elevated in tissue damage and inflammation.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 10 – 40</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        total_proteins = st.number_input(
            "Total Protein (g/dL)",
            value=6.8
        )

        st.markdown(
            '<div class="desc-box">Measures Albumin + Globulin balance.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 6.0 – 8.3</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-card">', unsafe_allow_html=True)

        albumin = st.number_input(
            "Albumin (g/dL)",
            value=3.5
        )

        st.markdown(
            '<div class="desc-box">Main protein synthesized by liver.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="range-box">Normal: 3.4 – 5.4</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 7. CALCULATED VARIABLES
# =========================================================

globulin = total_proteins - albumin

ag_ratio = albumin / (
    globulin if globulin != 0 else 1
)

# =========================================================
# 8. PREDICTION BUTTON
# =========================================================

if st.button("🚀 EXECUTE NEURAL DIAGNOSIS"):

    if model:

        try:

            data = {
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

            input_df = pd.DataFrame(data)

            prob = model.predict_proba(input_df)[0][1]

            st.markdown("---")

            c1, c2 = st.columns(2)

            with c1:

                if prob > 0.5:
                    st.error("🚩 PATHOLOGICAL CONDITION DETECTED")
                else:
                    st.success("✅ PHYSIOLOGICAL STABILITY")

            with c2:

                st.metric(
                    "Systemic Risk Index",
                    f"{round(prob*100,2)}%"
                )

            # =================================================
            # GRAPHS
            # =================================================

            st.markdown("## 📊 Clinical Visualization")

            features = [
                'Bilirubin',
                'DBilirubin',
                'ALP',
                'ALT',
                'AST',
                'Protein',
                'Albumin'
            ]

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

            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=norms,
                    theta=features,
                    fill='toself',
                    name='Healthy'
                )
            )

            fig.add_trace(
                go.Scatterpolar(
                    r=users,
                    theta=features,
                    fill='toself',
                    name='Patient'
                )
            )

            fig.update_layout(
                template='plotly_dark'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # INTERPRETATION
            # =================================================

            st.markdown(
                '<div class="med-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "## 👨‍⚕️ Intelligent Clinical Interpretation"
            )

            if prob > 0.5:

                st.warning(
                    "Elevated liver risk detected."
                )

                st.write(
                    "- Recommend hepatologist consultation."
                )

                st.write(
                    "- Suggested tests: HBsAg, HCV, Ultrasound."
                )

                st.write(
                    "- Avoid alcohol and processed foods."
                )

            else:

                st.success(
                    "Biomarkers appear within healthy limits."
                )

                st.write(
                    "- Continue annual screening."
                )

                st.write(
                    "- Maintain healthy nutrition."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Error: {e}")

# =========================================================
# 9. FOOTER
# =========================================================

st.markdown("""
<div class="footer-credits">
<strong style="color:white;">Aarish Ali</strong><br>
Lead AI Researcher | HepaScan Framework<br>
Dept. of Statistics & Operations Research<br>
Aligarh Muslim University
</div>
""", unsafe_allow_html=True)