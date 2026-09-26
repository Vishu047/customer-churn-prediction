# AI-Powered Customer Churn Prediction & Decision Support System

An end-to-end Machine Learning system that predicts customer attrition, quantifies financial churn risk, and generates actionable, explainable retention strategies.

Built with **Python 3.12**, **Scikit-Learn**, **TreeSHAP**, and **FastAPI** using the IBM Telco Customer Churn dataset.

---

## 📌 Project Overview

Customer acquisition costs typically exceed retention costs by a factor of 5 to 7. This project implements a production-style machine learning pipeline that shifts churn management from reactive intervention to proactive risk mitigation.

### Key Highlights
- **Zero Data Leakage:** Preprocessing transformations fit strictly on training splits using Scikit-Learn `ColumnTransformer`.
- **Business-First Metric Optimization:** Prioritized **Recall (80%)** and **ROC-AUC (0.8475)** over raw accuracy on imbalanced data (73.5% Stayed vs. 26.5% Churned) to minimize costly False Negatives.
- **Model Explainability:** Implemented cooperative game-theory attribution via **TreeSHAP** to surface both global risk catalysts and individual customer drivers.
- **Production Serving:** Engineered a low-latency **FastAPI** REST microservice with strict **Pydantic** schema validation and automated **pytest** unit tests.

---

## 🏗️ System Architecture

```text
[ Raw Data: IBM Telco (7,043 rows) ]
                 │
                 ▼
[ Data Cleaning & Stratified Train/Test Split ] (80/20 Split, Zero Leakage)
                 │
                 ▼
[ ColumnTransformer Pipeline ] ──► StandardScaler (4 features) + OneHotEncoder (15 features)
                 │
                 ▼
[ Tuned Random Forest Classifier ] ──► 5-Fold CV ROC-AUC: 0.8475 | Test Recall: 80%
                 │
                 ├──► [ TreeSHAP Interpretability ] ──► Risk Driver Attribution & Beeswarm Analysis
                 │
                 ▼
[ FastAPI Inference Microservice ] ──► Strict Pydantic Schema Validation (Port 8000)
                 │
                 ▼
[ Client / Business Consumer ] ──► Real-Time Churn Probability, Risk Tier, & Recommendations
