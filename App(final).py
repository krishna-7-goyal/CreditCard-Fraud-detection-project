import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Page config
st.set_page_config(page_title="💳 Fraud Detection App", layout="wide")
st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("Detect suspicious transactions with detailed insights and feature explanations.")

# Load model and feature names
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("model_features.pkl")

# Sidebar layout
st.sidebar.header("📥 Enter Transaction Details")
amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
gender = st.sidebar.selectbox("Gender", ['M', 'F'])
city_pop = st.sidebar.slider("City Population", 100, 1000000, 500000)
category = st.sidebar.selectbox("Transaction Category", [
    'shopping_net', 'shopping_pos', 'gas_transport', 'grocery_pos', 'grocery_net',
    'home', 'kids_pets', 'entertainment', 'travel', 'food_dining', 'health_fitness',
    'misc_net', 'misc_pos', 'personal_care'])
lat = st.sidebar.number_input("Latitude", value=40.730610)
long = st.sidebar.number_input("Longitude", value=-73.935242)
hour = st.sidebar.slider("Hour of Transaction", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)

# Prepare input
input_dict = {
    'amt': amt,
    'gender': gender,
    'city_pop': city_pop,
    'category': category,
    'lat': lat,
    'long': long,
    'hour': hour,
    'day': day
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# Predict
proba = model.predict_proba(input_encoded)[0][1]
label = "Fraudulent Transaction 🚨" if proba > 0.4 else "Legitimate Transaction ✅"

colA, colB = st.columns([2, 3])
with colA:
    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.metric("Fraud Probability", f"{proba:.2f}")

# SHAP explanation for the current input
explainer = shap.Explainer(model)
shap_values = explainer(input_encoded)
shap_dict = dict(zip(feature_names, shap_values.values[0]))

# Group SHAP values by original feature name
group_map = defaultdict(float)
for fname, val in shap_dict.items():
    if "gender_" in fname:
        group_map['gender'] += abs(val)
    elif "category_" in fname:
        group_map['category'] += abs(val)
    else:
        group_map[fname] += abs(val)

# Show top contributing features
sorted_feats = sorted(group_map.items(), key=lambda x: x[1], reverse=True)[:5]
names, scores = zip(*sorted_feats)

with colB:
    st.markdown("#### 🔍 Top Factors for Prediction")
    fig, ax = plt.subplots()
    sns.barplot(x=scores, y=names, palette='coolwarm', ax=ax)
    ax.set_xlabel("Impact Score (SHAP)")
    st.pyplot(fig)

# Fraud pattern insights
st.markdown("---")
st.subheader("📊 Fraud Trends & Insights")
df = pd.read_csv("fraud_balanced.csv")
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Fraud Rate by Hour**")
    hourly_fraud = df.groupby('hour')['is_fraud'].mean()
    fig1, ax1 = plt.subplots()
    sns.lineplot(x=hourly_fraud.index, y=hourly_fraud.values, marker='o', ax=ax1)
    ax1.set_ylabel("Fraud Rate")
    ax1.set_xlabel("Hour of Day")
    st.pyplot(fig1)

with col2:
    st.markdown("**Fraud Rate by Day of Month**")
    daily_fraud = df.groupby('day')['is_fraud'].mean()
    fig2, ax2 = plt.subplots()
    sns.lineplot(x=daily_fraud.index, y=daily_fraud.values, marker='o', color='green', ax=ax2)
    ax2.set_ylabel("Fraud Rate")
    ax2.set_xlabel("Day")
    st.pyplot(fig2)

st.markdown("---")