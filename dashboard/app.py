

# # dashboard/app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # =========================
# # Paths
# # =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "../data/processed")
# MODEL_DIR = os.path.join(BASE_DIR, "../model")

# CLEAN_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_clean.csv")
# FEATURES_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_features.csv")
# MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
# COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")

# # =========================
# # Load data & model
# # =========================
# df_clean = pd.read_csv(CLEAN_DATA_PATH)
# df_features = pd.read_csv(FEATURES_DATA_PATH)
# model = joblib.load(MODEL_PATH)
# columns_used = joblib.load(COLUMNS_PATH)

# st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
# st.title("🏦 Hybrid Fraud Detection System")

# # =========================
# # 1️⃣ Key Metrics
# # =========================
# st.subheader("📊 Key Metrics")

# total_txn = len(df_clean)
# fraud_txn = int(df_clean["Is_Fraud"].sum())
# fraud_rate = fraud_txn / total_txn * 100

# col1, col2, col3 = st.columns(3)
# col1.metric("Total Transactions", total_txn)
# col2.metric("Fraud Transactions", fraud_txn)
# col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# # =========================
# # 2️⃣ Fraud Analytics
# # =========================
# st.subheader("📈 Fraud Analysis")

# # Fraud by Hour
# st.markdown("**Fraud Rate by Transaction Hour**")
# fraud_by_hour = df_clean.groupby("Transaction_Hour")["Is_Fraud"].mean()
# st.bar_chart(fraud_by_hour)

# # Fraud by Amount Bucket
# df_clean["amount_bucket"] = pd.cut(
#     df_clean["Transaction_Amount"],
#     bins=[0, 500, 1000, 3000, 10000, np.inf],
#     labels=["0-500", "500-1000", "1000-3000", "3000-10000", "10000+"]
# )
# st.markdown("**Fraud Rate by Amount Bucket**")
# fraud_by_amt = df_clean.groupby("amount_bucket")["Is_Fraud"].mean()
# st.bar_chart(fraud_by_amt)

# # =========================
# # 3️⃣ Real-Time Hybrid Prediction
# # =========================
# st.subheader("🚨 Real-Time Transaction Risk Check")
# st.markdown("### Transaction Details")

# # --- Numeric Inputs ---
# txn_amount = st.number_input("Transaction Amount", 0.0, value=2500.0)
# txn_hour = st.slider("Transaction Hour", 0, 23, 12)
# txn_weekday = st.slider("Transaction Weekday (0=Mon)", 0, 6, 2)
# account_balance = st.number_input("Account Balance", 0.0, value=50000.0)
# avg_balance = st.number_input("Customer Avg Balance", 0.0, value=45000.0)
# balance_dev = account_balance - avg_balance

# # --- Categorical Inputs ---
# Transaction_Device = st.selectbox(
#     "Transaction Device",
#     ["ATM", "POS", "Mobile App", "USSD", "Voice Assistant"]
# )

# Device_Type = st.selectbox(
#     "Device Type",
#     ["Mobile", "Desktop", "Tablet"]
# )

# Account_Type = st.selectbox(
#     "Account Type",
#     df_clean["Account_Type"].unique()
# )

# # =========================
# # Build input vector
# # =========================
# input_dict = {
#     "Transaction_Amount": txn_amount,
#     "Transaction_Hour": txn_hour,
#     "Transaction_Weekday": txn_weekday,
#     "Account_Balance": account_balance,
#     "customer_avg_balance": avg_balance,
#     "balance_dev": balance_dev
# }

# X_new = pd.DataFrame([input_dict])

# # One-hot encode categorical columns safely
# for col in columns_used:
#     X_new[col] = 0

# # Map categorical selections to one-hot columns
# for col in columns_used:
#     if col.startswith("Transaction_Device_") and col.endswith(Transaction_Device):
#         X_new[col] = 1
#     if col.startswith("Device_Type_") and col.endswith(Device_Type):
#         X_new[col] = 1
#     if col.startswith("Account_Type_") and col.endswith(Account_Type):
#         X_new[col] = 1

# X_new = X_new[columns_used]

# # =========================
# # 4️⃣ ML Probability
# # =========================
# ml_prob = model.predict_proba(X_new)[0][1]

# # =========================
# # 5️⃣ Rule Engine
# # =========================
# rule_score = 0
# if txn_amount > 3000:
#     rule_score += 3
# elif txn_amount > 1500:
#     rule_score += 2

# if txn_hour in [0,1,2,3,4,5]:
#     rule_score += 2

# rule_score_norm = min(rule_score / 7, 1.0)

# # =========================
# # 6️⃣ Hybrid Score
# # =========================
# hybrid_score = 0.7 * ml_prob + 0.3 * rule_score_norm

# # =========================
# # 7️⃣ Show metrics and decision
# # =========================
# st.markdown("### 🔎 Risk Assessment")
# col1, col2, col3 = st.columns(3)
# col1.metric("ML Fraud Probability", f"{ml_prob:.2f}")
# col2.metric("Rule Score", rule_score)
# col3.metric("Hybrid Risk Score", f"{hybrid_score:.2f}")

# if rule_score >= 6 or hybrid_score >= 0.6:
#     st.error("🚨 FRAUD ALERT — Transaction BLOCKED")
# else:
#     st.success("✅ Transaction APPROVED")



import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "../model")

FEATURES_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")  # Updated to match your trained RF
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")

# =========================
# Load data & model
# =========================
df_features = pd.read_csv(FEATURES_DATA_PATH)
model = joblib.load(MODEL_PATH)
columns_used = joblib.load(COLUMNS_PATH)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🏦 Hybrid Fraud Detection System")

# =========================
# 1️⃣ Key Metrics
# =========================
st.subheader("📊 Key Metrics")
total_txn = len(df_features)
fraud_txn = int(df_features["Is_Fraud"].sum())
fraud_rate = fraud_txn / total_txn * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_txn)
col2.metric("Fraud Transactions", fraud_txn)
col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# =========================
# 2️⃣ Fraud Analytics
# =========================
st.subheader("📈 Fraud Analysis")

# Fraud by Hour
st.markdown("**Fraud Rate by Transaction Hour**")
fraud_by_hour = df_features.groupby("Transaction_Hour")["Is_Fraud"].mean()
st.bar_chart(fraud_by_hour)

# Fraud by Amount Bucket
df_features["amount_bucket"] = pd.cut(
    df_features["Transaction_Amount"],
    bins=[0, 500, 1000, 3000, 10000, np.inf],
    labels=["0-500", "500-1000", "1000-3000", "3000-10000", "10000+"]
)
st.markdown("**Fraud Rate by Amount Bucket**")
fraud_by_amt = df_features.groupby("amount_bucket")["Is_Fraud"].mean()
st.bar_chart(fraud_by_amt)

# =========================
# 3️⃣ Real-Time Hybrid Prediction
# =========================
st.subheader("🚨 Real-Time Transaction Risk Check")
st.markdown("### Transaction Details")

# --- Numeric Inputs ---
txn_amount = st.number_input("Transaction Amount", 0.0, value=2500.0)
txn_hour = st.slider("Transaction Hour", 0, 23, 12)
txn_weekday = st.slider("Transaction Weekday (0=Mon)", 0, 6, 2)
account_balance = st.number_input("Account Balance", 0.0, value=50000.0)
avg_balance = st.number_input("Customer Avg Balance", 0.0, value=45000.0)
balance_dev = account_balance - avg_balance

# --- Categorical Inputs ---
Transaction_Device = st.selectbox(
    "Transaction Device",
    ["ATM", "POS", "Mobile App", "USSD", "Voice Assistant"]
)
Device_Type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
Account_Type = st.selectbox("Account Type", df_features["Account_Type"].unique())

# =========================
# Build input vector safely
# =========================
input_dict = {
    "Transaction_Amount": txn_amount,
    "Transaction_Hour": txn_hour,
    "Transaction_Weekday": txn_weekday,
    "Account_Balance": account_balance,
    "customer_avg_balance": avg_balance,
    "balance_dev": balance_dev
}

X_new = pd.DataFrame([input_dict])

# Initialize all columns to 0
for col in columns_used:
    X_new[col] = 0

# Map categorical selections
for col in columns_used:
    if f"Transaction_Device_{Transaction_Device}" == col:
        X_new[col] = 1
    if f"Device_Type_{Device_Type}" == col:
        X_new[col] = 1
    if f"Account_Type_{Account_Type}" == col:
        X_new[col] = 1

X_new = X_new[columns_used]

# =========================
# ML Probability
# =========================
ml_prob = model.predict_proba(X_new)[0][1]

# =========================
# Rule Engine
# =========================
rule_score = 0
if txn_amount > 3000:
    rule_score += 3
elif txn_amount > 1500:
    rule_score += 2

if txn_hour in [0,1,2,3,4,5]:
    rule_score += 2

rule_score_norm = min(rule_score / 7, 1.0)

# =========================
# Hybrid Score
# =========================
hybrid_score = 0.7 * ml_prob + 0.3 * rule_score_norm

# =========================
# Show metrics & decision
# =========================
st.markdown("### 🔎 Risk Assessment")
col1, col2, col3 = st.columns(3)
col1.metric("ML Fraud Probability", f"{ml_prob:.2f}")
col2.metric("Rule Score", rule_score)
col3.metric("Hybrid Risk Score", f"{hybrid_score:.2f}")

if rule_score >= 6 or hybrid_score >= 0.6:
    st.error("🚨 FRAUD ALERT — Transaction BLOCKED")
else:
    st.success("✅ Transaction APPROVED")
