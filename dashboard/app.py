# import streamlit as st
# import pandas as pd
# import joblib

# # =========================
# # Load data and model
# # =========================
# df = pd.read_csv("../data/processed/bank_transactions_features.csv")
# rf_model = joblib.load("../model/rf_model.pkl")
# columns_used = joblib.load("../model/columns_used.pkl")  # columns used in training

# st.title("Bank Transactions Fraud Detection Dashboard")

# # =========================
# # 1. Key Metrics
# # =========================
# st.subheader("Key Metrics")
# total_txn = len(df)
# fraud_txn = df['Is_Fraud'].sum()
# fraud_rate = fraud_txn / total_txn * 100

# st.metric("Total Transactions", total_txn)
# st.metric("Fraud Transactions", fraud_txn)
# st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

# # =========================
# # 2. Fraud Analysis Charts
# # =========================
# st.subheader("Fraud Rate by Transaction Device")
# if any('Transaction_Device_' in c for c in df.columns):
#     fraud_by_device = df.groupby('Transaction_Device')['Is_Fraud'].mean()
#     st.bar_chart(fraud_by_device)
# else:
#     st.write("Transaction_Device info not available")

# st.subheader("Fraud Rate by Account Type")
# fraud_by_account = df.groupby('Account_Type')['Is_Fraud'].mean()
# st.bar_chart(fraud_by_account)

# st.subheader("Fraud Rate by Transaction Hour")
# fraud_by_hour = df.groupby('Transaction_Hour')['Is_Fraud'].mean()
# st.bar_chart(fraud_by_hour)

# # =========================
# # 3. Real-Time Fraud Alert
# # =========================
# st.subheader("Test Transaction Fraud Alert")

# # Numeric inputs
# Account_Balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
# txn_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
# txn_weekday = st.number_input("Transaction Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
# customer_avg_balance = st.number_input("Customer Average Balance", min_value=0.0, value=50000.0)
# balance_dev = Account_Balance - customer_avg_balance

# # Binary features
# new_device = st.selectbox("New Device?", [0,1])
# new_location = st.selectbox("New Location?", [0,1])
# high_risk_merchant = st.selectbox("High-Risk Merchant?", [0,1])

# # Categorical features
# Transaction_Device = st.selectbox("Transaction Device", df['Transaction_Device'].unique())
# Device_Type = st.selectbox("Device Type", df['Device_Type'].unique())
# Account_Type = st.selectbox("Account Type", df['Account_Type'].unique())

# # =========================
# # 4. Prepare input for model
# # =========================
# input_dict = {
#     'Account_Balance':[Account_Balance],
#     'txn_hour':[txn_hour],
#     'txn_weekday':[txn_weekday],
#     'customer_avg_balance':[customer_avg_balance],
#     'balance_dev':[balance_dev],
#     'new_device':[new_device],
#     'new_location':[new_location],
#     'high_risk_merchant':[high_risk_merchant]
# }

# # One-hot encode categorical features according to training columns
# for col in columns_used:
#     if 'Transaction_Device_' in col:
#         device_name = col.replace('Transaction_Device_', '')
#         input_dict[col] = [1 if device_name == Transaction_Device else 0]
#     if 'Device_Type_' in col:
#         dev_name = col.replace('Device_Type_', '')
#         input_dict[col] = [1 if dev_name == Device_Type else 0]
#     if 'Account_Type_' in col:
#         acc_name = col.replace('Account_Type_', '')
#         input_dict[col] = [1 if acc_name == Account_Type else 0]

# X_new = pd.DataFrame(input_dict)

# # Align columns exactly as in training
# X_new = X_new.reindex(columns=columns_used, fill_value=0)

# # =========================
# # 5. Make prediction
# # =========================
# prediction = rf_model.predict(X_new)[0]
# st.write("⚠️ FRAUD ALERT! Transaction is suspicious!" if prediction==1 else "✅ Transaction Approved")








import streamlit as st
import pandas as pd
import joblib

# =========================
# Load assets
# =========================
df = pd.read_csv("../data/processed/bank_transactions_features.csv")
model = joblib.load("../model/rf_model.pkl")
columns_used = joblib.load("../model/columns_used.pkl")

st.title("Bank Transactions Fraud Detection Dashboard")

# =========================
# Test Transaction
# =========================
st.subheader("Test Transaction Fraud Alert")

# --- Inputs ---
Account_Balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
Transaction_Amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
Transaction_Hour = st.slider("Transaction Hour", 0, 23, 12)
Transaction_Weekday = st.slider("Transaction Weekday (0=Mon)", 0, 6, 2)

Transaction_Device = st.selectbox(
    "Transaction Device",
    [c.replace("Transaction_Device_", "") for c in df.columns if c.startswith("Transaction_Device_")]
)

Device_Type = st.selectbox(
    "Device Type",
    [c.replace("Device_Type_", "") for c in df.columns if c.startswith("Device_Type_")]
)

Account_Type = st.selectbox(
    "Account Type",
    [c.replace("Account_Type_", "") for c in df.columns if c.startswith("Account_Type_")]
)

# =========================
# Build input row (SAFE)
# =========================
X_new = pd.DataFrame(0, index=[0], columns=columns_used)

# Numeric features
X_new["Account_Balance"] = Account_Balance
X_new["Transaction_Amount"] = Transaction_Amount
X_new["Transaction_Hour"] = Transaction_Hour
X_new["Transaction_Weekday"] = Transaction_Weekday

# One-hot selections
X_new[f"Transaction_Device_{Transaction_Device}"] = 1
X_new[f"Device_Type_{Device_Type}"] = 1
X_new[f"Account_Type_{Account_Type}"] = 1

# =========================
# Predict
# =========================
if st.button("Run Fraud Check"):
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0][1]

    if prediction == 1:
        st.error(f"⚠️ FRAUD ALERT! Risk Score: {probability:.2%}")
    else:
        st.success(f"✅ Transaction Approved. Risk Score: {probability:.2%}")
