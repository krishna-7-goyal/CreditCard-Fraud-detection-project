# 💳 Credit Card Fraud Detection System

A real-time, intelligent, and interpretable machine learning application to detect fraudulent credit card transactions. Built using Python, XGBoost, SHAP, and Streamlit.

![Fraud Detection Banner](https://img.shields.io/badge/FraudDetection-RealTime-blueviolet)  
>  99% precision, recall, and F1-score on both fraud and legitimate transactions.

---

## 📌 Project Overview

Credit card fraud costs the global economy billions annually. This project aims to provide a **real-time fraud detection system** using machine learning, capable of detecting suspicious transactions and explaining its decisions using SHAP.

Key highlights:

- Built with **XGBoost** for accuracy and speed
- Tackles extreme **class imbalance** with SMOTE and strategic undersampling
- Deployed using **Streamlit** for real-time predictions
- Powered by **SHAP** for transparent, interpretable AI


## 🚀 Features

- **Interactive Streamlit UI** for entering transaction details
- **Probability-based fraud classification**
- **SHAP visualizations** to explain model predictions
- **Time-based fraud analytics** (hour/day trends)
- Predicts based on:
  - Transaction Amount
  - Gender
  - City Population
  - Transaction Category
  - Latitude & Longitude
  - Time (Hour & Day)

---

## 🧠 Model Details

- **Algorithm:** XGBoost Classifier
- **Feature Engineering:**
  - One-hot encoding for categorical variables
  - Extracted temporal features (hour, day)
- **Balancing Strategy:**
  - Strategic undersampling (10x fraud ratio)
  - SMOTE for synthetic oversampling
- **Explainability:**
  - SHAP values aggregated per input feature group

### 📈 Performance Metrics

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Legitimate    | 0.99      | 0.99   | 0.99     |
| Fraudulent    | 0.99      | 0.99   | 0.99     |
| **Accuracy**  |           |        | **0.99** |

---

## 🧰 Tech Stack

| Category        | Tools Used                                      |
|----------------|--------------------------------------------------|
| Programming     | Python 3.x                                      |
| Modeling        | XGBoost, scikit-learn, imbalanced-learn (SMOTE) |
| Explainability  | SHAP                                            |
| Web App         | Streamlit                                       |
| Serialization   | Joblib                                          |
| Visualization   | Matplotlib, Seaborn                             |
| Environment     | Jupyter Notebook, VS Code                       |

---

## 📦 Project Structure
├── app.py # Streamlit deployment code
├── fraud_model.pkl # Trained XGBoost model
├── model_features.pkl # Feature columns used by the model
├── fraud_balanced.csv # Balanced training dataset
├── requirements.txt # List of Python dependencies
└── README.md # Project overview

---

