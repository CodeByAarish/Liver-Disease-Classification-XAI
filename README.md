# 🏥 A SHAP Liver Disease Classification System and Risk Stratification
**A SHAP-Explainable Machine Learning and Deep Learning Hybrid Framework for Liver Disease Classification and Risk Stratification**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.8%25-orange.svg)

## 📌 Project Overview
This project is designed to assist medical professionals in identifying liver disease in patients based on various diagnostic features. Using a highly optimized **LightGBM** classifier, the system achieves near-perfect accuracy on validation datasets, making it a robust tool for clinical decision support.

## 🚀 Key Features
*   **High Performance:** Achieved an **F1-Score of 99.8%** using advanced Gradient Boosting.
*   **End-to-End Pipeline:** Includes automated data cleaning, handling missing values, and feature scaling using Scikit-Learn Pipelines.[cite: 3]
*   **Interactive UI:** A user-friendly Streamlit dashboard for real-time predictions.
*   **Production Ready:** Optimized and serialized model using `joblib` for instant inference.[cite: 3]

## 🛠️ Tech Stack
*   **Language:** Python
*   **ML Libraries:** Scikit-Learn, LightGBM, XGBoost, CatBoost
*   **Preprocessing:** RobustScaler, PowerTransformer, SimpleImputer
*   **Deployment:** Streamlit, Joblib

## 📂 Project Structure
```text
📂 Liver-Project/
├── 📄 app.py              # Streamlit Web Application
├── 📄 final_model.pkl     # Serialized LightGBM Pipeline
├── 📄 requirements.txt    # Project Dependencies
├── 📄 notebooks/          # Model Training & HPT Notebooks
└── 📄 data/               # Training & Testing Datasets
