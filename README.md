# AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning system that predicts customer churn, identifies high-risk accounts, and explains churn drivers to support data-driven retention strategies.

Built using **Scikit-Learn**, **SHAP**, and **FastAPI** on the IBM Telco Customer Churn dataset.

---

## 📌 Project Overview
Customer acquisition costs significantly exceed retention costs. This project implements a production-style machine learning pipeline that:
- Fixes data quality issues and applies leak-free preprocessing via a serialized `ColumnTransformer`.
- Benchmarks linear baselines against non-linear tree ensembles using Stratified K-Fold Cross-Validation.
- Explains individual and global churn drivers using **TreeSHAP**.
- Serves real-time inference via a low-latency **FastAPI** REST API with strict **Pydantic** schema validation.

---

## 🏗️ System Architecture

```text
[ IBM Telco Raw Data ]
          │
          ▼
[ Stratified Split & Cleaning ] (Zero Data Leakage)
          │
          ▼
[ ColumnTransformer Pipeline ] ──► StandardScaler + OneHotEncoder
          │
          ▼
[ Tuned Random Forest Classifier ] ──► (ROC-AUC: 0.8475, Recall: 80%)
          │
          ├──► [ TreeSHAP Interpretability ] ──► Risk Driver Attribution
          │
          ▼
[ FastAPI Inference API ] ──► Schema-validated REST endpoints (/predict)
