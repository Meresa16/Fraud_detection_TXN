


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

# FEATURES_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_features.csv")
# MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl")       # your trained RF model
# COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl") # columns used in training

# # =========================
# # Load model and columns
# # =========================
# model = joblib.load(MODEL_PATH)
# columns_used = joblib.load(COLUMNS_PATH)

# st.set_page_config(page_title="Fraud Prediction", layout="centered")
# st.title("🏦 Bank Transaction Fraud Prediction")

# st.markdown("Enter transaction details below to get an instant fraud prediction:")

# # =========================
# # User Inputs
# # =========================
# txn_amount = st.number_input("Transaction Amount", 0.0, value=2500.0)
# txn_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
# txn_weekday = st.slider("Transaction Weekday (0=Mon)", 0, 6, 2)
# account_balance = st.number_input("Account Balance", 0.0, value=50000.0)
# avg_balance = st.number_input("Customer Avg Balance", 0.0, value=45000.0)
# balance_dev = account_balance - avg_balance

# Transaction_Device = st.selectbox(
#     "Transaction Device",
#     ["ATM", "POS", "Mobile App", "USSD", "Voice Assistant"]
# )

# Device_Type = st.selectbox(
#     "Device Type",
#     ["Mobile", "Desktop", "Tablet"]
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

# # Add missing columns (one-hot categorical)
# for col in columns_used:
#     X_new[col] = 0

# # Map categorical inputs to one-hot columns
# for col in columns_used:
#     if col.startswith("Transaction_Device_") and col.endswith(Transaction_Device):
#         X_new[col] = 1
#     if col.startswith("Device_Type_") and col.endswith(Device_Type):
#         X_new[col] = 1

# X_new = X_new[columns_used]

# # =========================
# # Make prediction
# # =========================
# fraud_prob = model.predict_proba(X_new)[0][1]

# st.markdown("### 🔎 Prediction Result")
# st.metric("Fraud Probability", f"{fraud_prob:.2%}")

# if fraud_prob >= 0.5:
#     st.error("🚨 Transaction is likely FRAUDULENT")
# else:
#     st.success("✅ Transaction appears SAFE")








import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# =========================
# 1. Configuration & Paths
# =========================
st.set_page_config(
    page_title="Fraud Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Adjust these path names to match exactly where your main.py saved them
# In the previous step, we saved to ../models/ and ../data/processed/
DATA_DIR = os.path.join(BASE_DIR, "../data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "../models") 

FEATURES_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_clean.csv") # Used for dropdown lists & freq maps
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl") # Or "rf_model_imbalanced.pkl"
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")

# =========================
# 2. Load Resources (Cached)
# =========================
@st.cache_resource
def load_model_and_metadata():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.error(f"Model files not found. Please run the training pipeline first.\nExpected: {MODEL_PATH}")
        return None, None
    
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLUMNS_PATH)
    return model, cols

@st.cache_data
def load_reference_data():
    """Load clean data to get unique values for dropdowns and compute frequency maps."""
    if not os.path.exists(FEATURES_DATA_PATH):
        st.error(f"Data file not found: {FEATURES_DATA_PATH}")
        return None, None, None
    
    df = pd.read_csv(FEATURES_DATA_PATH)
    
    # Generate Frequency Maps (Must match logic in preprocessing.py)
    freq_maps = {}
    for col in ["City", "Bank_Branch", "Transaction_Location"]:
        if col in df.columns: # Clean data might already have these dropped if fully processed
            # If the clean CSV has the original columns, we compute the map
            freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        elif f"{col}_Freq" in df.columns:
            # If clean CSV only has the _Freq column, we can't reconstruct the map easily 
            # without the original labels. 
            # Assuming 'bank_transactions_clean.csv' from step 3 (which dropped originals? 
            # No, preprocessing step 3 usually drops them. 
            # If they are dropped, we can't make a dropdown. 
            # *Fallback*: We assume the CSV loaded here still has categorical columns 
            # OR we rely on a separate mapping file.
            pass

    return df, freq_maps

model, columns_used = load_model_and_metadata()
df_clean, freq_maps = load_reference_data()

# =========================
# 3. Sidebar Inputs
# =========================
st.sidebar.header("📝 Transaction Details")

# -- Numeric Inputs --
txn_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=250.0, step=10.0)
account_balance = st.sidebar.number_input("Current Account Balance ($)", min_value=0.0, value=5000.0, step=100.0)

# -- Time Inputs --
txn_hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)
txn_weekday = st.sidebar.selectbox("Day of Week", 
    options=[0,1,2,3,4,5,6], 
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
    index=2
)

# -- Categorical Inputs (Dynamic Dropdowns) --
# We try to get unique lists from df_clean, otherwise fall back to static lists
def get_unique(col_name, default_list):
    if df_clean is not None and col_name in df_clean.columns:
        return df_clean[col_name].unique().tolist()
    return default_list

cities = get_unique("City", ["New York", "London", "Paris", "Tokyo"])
devices = get_unique("Transaction_Device", ["Mobile", "Desktop", "Tablet", "POS"])
types = get_unique("Transaction_Type", ["Purchase", "Withdrawal", "Transfer"])

selected_city = st.sidebar.selectbox("City", cities)
selected_device = st.sidebar.selectbox("Transaction Device", devices)
selected_type = st.sidebar.selectbox("Transaction Type", types)

# =========================
# 4. Feature Engineering for Inference
# =========================
if st.button("Predict Fraud Probability"):
    if model is None:
        st.error("Model not loaded.")
        st.stop()

    # A. Initialize Input Dictionary with Default 0s for all model columns
    input_data = {col: 0 for col in columns_used}
    
    # B. Fill Numeric Features
    # Note: Ensure these names match EXACTLY what was in X_train
    input_data["Transaction_Amount"] = txn_amount
    input_data["Transaction_Hour"] = txn_hour
    input_data["Transaction_Weekday"] = txn_weekday
    input_data["Account_Balance"] = account_balance
    
    # C. Handle Frequency Encoding
    # If the model uses 'City_Freq', we must look up the frequency of the selected city
    if "City_Freq" in columns_used:
        # Look up in our map, default to 0.0 or a low probability if unknown
        city_freq = freq_maps.get("City", {}).get(selected_city, 0.001) 
        input_data["City_Freq"] = city_freq

    # D. Handle One-Hot Encoding
    # The model likely has columns like 'Transaction_Device_Mobile', 'Transaction_Device_POS'
    
    # Construct the expected column name
    device_col = f"Transaction_Device_{selected_device}"
    type_col = f"Transaction_Type_{selected_type}"
    
    # Set to 1 if it exists in the model features
    if device_col in input_data:
        input_data[device_col] = 1
    if type_col in input_data:
        input_data[type_col] = 1

    # =========================
    # 5. Prediction
    # =========================
    # Convert dict to DataFrame with correct order
    X_new = pd.DataFrame([input_data])
    X_new = X_new[columns_used] # Enforce column order

    # Predict
    try:
        fraud_prob = model.predict_proba(X_new)[0][1]
        prediction = model.predict(X_new)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # =========================
    # 6. Display Results
    # =========================
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if fraud_prob > 0.5:
            st.error("### 🚨 High Risk")
            st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
        else:
            st.success("### ✅ Low Risk")
            st.image("https://cdn-icons-png.flaticon.com/512/148/148767.png", width=100)
            
    with col2:
        st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        
        st.progress(float(fraud_prob))
        
        st.markdown("#### Input Summary")
        st.json({
            "Amount": f"${txn_amount}",
            "Location": selected_city,
            "Device": selected_device,
            "Type": selected_type
        })
else:
    st.info("Adjust settings in the sidebar and click Predict.")