# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # 1. Page Configuration
# st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

# # 2. Load the Exported Model
# @st.cache_resource
# def load_model():
#     # Make sure this filename matches exactly what you exported from your notebook
#     with open('liver_model.pkl', 'rb') as f:
#         return pickle.load(f)

# # Global variables to hold model data
# try:
#     model_data = load_model()
#     model = model_data['model']
#     features_list = model_data['features']
# except FileNotFoundError:
#     st.error("⚠️ liver_model.pkl not found! Please run the export cell in your notebook first.")
#     st.stop()
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # 3. UI Design
# st.title("🏥 Liver Patient Prediction System")
# st.markdown("Enter patient clinical markers below to analyze the probability of liver disease.")
# st.markdown("---")

# # 4. Input Layout using Columns
# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age", 1, 100, 30)
#     gender = st.selectbox("Gender", ("Male", "Female"))
#     total_bilirubin = st.number_input("Total Bilirubin", 0.0, 80.0, 1.0)
#     direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.5)
#     alkphos = st.number_input("Alkaline Phosphotase", 0, 3000, 200)

# with col2:
#     sgpt = st.number_input("SGPT (Alamine Aminotransferase)", 0, 2000, 50)
#     sgot = st.number_input("SGOT (Aspartate Aminotransferase)", 0, 5000, 50)
#     total_proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.0)
#     albumin = st.number_input("Albumin", 0.0, 6.0, 3.0)
#     ag_ratio = st.number_input("A/G Ratio", 0.0, 3.0, 1.0)

# # Preprocessing to match numeric training data logic
# gender_bin = 1 if gender == "Male" else 0

# # Create input dictionary with names EXACTLY as they appeared in your project
# input_dict = {
#     'Age': age,
#     'Gender': gender_bin,
#     'Total_Bilirubin': total_bilirubin,
#     'Direct_Bilirubin': direct_bilirubin,
#     'Alkaline_Phosphotase': alkphos,
#     'Alamine_Aminotransferase': sgpt,
#     'Aspartate_Aminotransferase': sgot,
#     'Total_Protiens': total_proteins,
#     'Albumin': albumin,
#     'Albumin_and_Globulin_Ratio': ag_ratio
# }

# input_df = pd.DataFrame([input_dict])

# # 5. Prediction Logic
# st.markdown("---")
# if st.button("Run Diagnostic Analysis", use_container_width=True):
#     try:
#         # Reorder columns to match the training set order automatically
#         final_input = input_df[features_list]
        
#         # Use your trained classifier (CatBoost, XGBoost, etc.) to predict
#         prediction = model.predict(final_input)
#         prediction_proba = model.predict_proba(final_input)
        
#         confidence = np.max(prediction_proba) * 100

#         if prediction[0] == 1:
#             st.error(f"### Result: High Risk of Liver Disease")
#             st.metric("Confidence Level", f"{confidence:.2f}%")
#         else:
#             st.success(f"### Result: Patient appears Healthy")
#             st.metric("Confidence Level", f"{confidence:.2f}%")
            
#     except Exception as e:
#         st.error(f"An error occurred during prediction: {e}")
#         st.info("Ensure the 'features_list' matches your model's expected input columns.")

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Model ko load kijiye
model = pickle.load(open('liver_model.pkl', 'rb'))

st.title("🏥 Liver Disease Prediction System")
st.write("Aarish, enter patient clinical markers below:")

# 2. User input fields (Based on your dataset features)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    bilirubin = st.number_input("Bilirubin", value=1.0)
    albumin = st.number_input("Albumin", value=3.0)

with col2:
    sgot = st.number_input("SGOT", value=40.0)
    alk_phos = st.number_input("Alk_Phosphate", value=100.0)
    prothrombin = st.number_input("Prothrombin Time", value=10.0)

# 3. Prediction Logic
if st.button("Predict Results"):
    # Input ko dataframe mein convert karein
    input_df = pd.DataFrame([[age, bilirubin, albumin, sgot, alk_phos, prothrombin]], 
                            columns=['Age', 'Bilirubin', 'Albumin', 'SGOT', 'Alk_Phosphate', 'Prothrombin'])
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("⚠️ Prediction: Patient likely has Liver Disease.")
    else:
        st.success("✅ Prediction: Patient is Healthy.")