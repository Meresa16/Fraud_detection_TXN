











# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # ==========================================
# # 1. SETUP & CONFIG
# # ==========================================
# st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "../model")

# # File Paths
# MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl")
# SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
# COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")
# FREQ_MAPS_PATH = os.path.join(MODEL_DIR, "freq_maps.pkl") 
# UNIQUE_VALS_PATH = os.path.join(MODEL_DIR, "unique_values.pkl")

# # ==========================================
# # 2. LOAD RESOURCES
# # ==========================================
# @st.cache_resource
# def load_resources():
#     required = [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, FREQ_MAPS_PATH, UNIQUE_VALS_PATH]
#     for f in required:
#         if not os.path.exists(f):
#             st.error(f"Missing critical file: {f}. Please run the training pipeline.")
#             st.stop()
            
#     try:
#         model = joblib.load(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)
#         cols = joblib.load(COLUMNS_PATH)
#         freq_maps = joblib.load(FREQ_MAPS_PATH)
#         unique_vals = joblib.load(UNIQUE_VALS_PATH)
#         return model, scaler, cols, freq_maps, unique_vals
#     except Exception as e:
#         st.error(f"Error loading files: {e}")
#         st.stop()

# model, scaler, model_features, freq_maps, unique_vals = load_resources()

# # ==========================================
# # 3. UI GENERATION (HUMAN FRIENDLY)
# # ==========================================
# st.title("🛡️ Transaction Fraud Inspector")
# st.markdown("Enter the transaction details below to assess risk.")

# with st.form("transaction_form"):
    
#     # --- SECTION 1: MONEY & ACCOUNT ---
#     st.subheader("1. Financial Information")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         # Explicit, clear input for Amount
#         txn_amount = st.number_input(
#             "Transaction Amount ($)", 
#             min_value=0.0, 
#             value=150.00, 
#             step=10.0,
#             help="The total value of the transaction."
#         )

#     with col2:
#         # Explicit, clear input for Balance
#         acc_balance = st.number_input(
#             "Current Account Balance ($)", 
#             min_value=0.0, 
#             value=5000.00, 
#             step=100.0,
#             help="Balance available before this transaction."
#         )
        
#     with col3:
#         # Explicit input for Age (Check if your model uses it)
#         # We check scaler.feature_names_in_ to see if Age is needed
#         cust_age = 30 # Default
#         if "Customer_Age" in scaler.feature_names_in_:
#             cust_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)

#     # --- SECTION 2: TIMING ---
#     st.subheader("2. Time & Date")
#     t1, t2 = st.columns(2)
#     with t1:
#         hour = st.slider("Hour of Day (24h format)", 0, 23, 14, help="0 = Midnight, 12 = Noon")
#     with t2:
#         weekday = st.selectbox("Day of Week", range(7), 
#                                format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])

#     # --- SECTION 3: CONTEXT (Location, Device, etc.) ---
#     st.subheader("3. Transaction Details")
#     cat1, cat2 = st.columns(2)
    
#     user_categoricals = {}

#     # We combine the maps to generate dropdowns automatically
#     all_categorical_keys = sorted(list(set(list(freq_maps.keys()) + list(unique_vals.keys()))))

#     for i, col_name in enumerate(all_categorical_keys):
#         # Clean up the label for the user (e.g. "Transaction_Type" -> "Transaction Type")
#         clean_label = col_name.replace("_", " ")
        
#         # Get options
#         options = []
#         if col_name in freq_maps:
#             options = sorted(list(freq_maps[col_name].keys()))
#         elif col_name in unique_vals:
#             options = unique_vals[col_name]

#         with [cat1, cat2][i % 2]:
#             user_categoricals[col_name] = st.selectbox(clean_label, options)

#     # --- SUBMIT BUTTON ---
#     st.markdown("---")
#     submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")


# # ==========================================
# # 4. EXECUTION LOGIC (MAPPING UI -> MODEL)
# # ==========================================
# if submitted:
    
#     # A. Initialize Input Vector with Zeros
#     input_vector = {col: 0 for col in model_features}
    
#     # B. Map the Manual Financial Inputs to the Scaler
#     # We create a dictionary that matches the EXACT column names used in training
#     raw_financials = {
#         "Transaction_Amount": txn_amount,
#         "Account_Balance": acc_balance,
#         # Only add age if the model actually expects it
#         "Customer_Age": cust_age if "Customer_Age" in scaler.feature_names_in_ else 0
#     }
    
#     try:
#         # 1. Convert to DataFrame
#         # We align this with scaler.feature_names_in_ to ensure correct order
#         raw_num_df = pd.DataFrame([raw_financials])
        
#         # Ensure we only pass columns the scaler knows about
#         # (This handles cases where Age might be missing from training)
#         final_raw_df = pd.DataFrame()
#         for col in scaler.feature_names_in_:
#             final_raw_df[col] = raw_num_df.get(col, 0) # Get value or 0
            
#         # 2. Scale values
#         scaled_values = scaler.transform(final_raw_df)
        
#         # 3. Put scaled values into the input vector
#         for idx, col in enumerate(scaler.feature_names_in_):
#             input_vector[col] = scaled_values[0][idx]
            
#     except Exception as e:
#         st.error(f"Error processing financial details: {e}")
#         st.stop()

#     # C. Map Time
#     if "Transaction_Hour" in input_vector: input_vector["Transaction_Hour"] = hour
#     if "Transaction_Weekday" in input_vector: input_vector["Transaction_Weekday"] = weekday

#     # D. Map Categoricals (The Hidden Logic)
#     for col_name, selected_val in user_categoricals.items():
#         # Case 1: Frequency Encoding (e.g., City)
#         if col_name in freq_maps:
#             target_col = f"{col_name}_Freq"
#             val = freq_maps[col_name].get(selected_val, 0)
#             if target_col in input_vector: input_vector[target_col] = val
                
#         # Case 2: One-Hot Encoding (e.g., Device)
#         else:
#             target_col = f"{col_name}_{selected_val}"
#             if target_col in input_vector: input_vector[target_col] = 1

#     # E. Predict
#     try:
#         # Convert to DF to ensure column order is perfect
#         X_final = pd.DataFrame([input_vector])[model_features]
        
#         prob = model.predict_proba(X_final)[0][1]
#         is_fraud = prob > 0.5
        
#         st.divider()
#         col_res1, col_res2 = st.columns([1, 2])
        
#         with col_res1:
#             st.metric("Risk Score", f"{prob:.1%}")
            
#         with col_res2:
#             if is_fraud:
#                 st.error("🚨 **POTENTIAL FRAUD DETECTED**")
#                 st.write("This transaction has been flagged for immediate review.")
#             else:
#                 st.success("✅ **TRANSACTION APPROVED**")
#                 st.write("This transaction appears legitimate.")
            
#             st.progress(prob)
            
#     except Exception as e:
#         st.error(f"Prediction Error: {e}")













import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import mlflow
import datetime

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="GuardianAI | Fraud Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "FinTech" look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0052cc;
        color: white;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #003d99;
        border: none;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #0f2a4a;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MLFLOW CONFIGURATION
# ==========================================
# This creates a local folder ./mlruns to store logs
# mlflow.set_tracking_uri("file:./mlruns")
# New way (Database)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Live_Dashboard_Inference")

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")

@st.cache_resource
def load_resources():
    paths = {
        "model": os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl"),
        "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
        "columns": os.path.join(MODEL_DIR, "columns_used.pkl"),
        "freq_maps": os.path.join(MODEL_DIR, "freq_maps.pkl"),
        "unique_vals": os.path.join(MODEL_DIR, "unique_values.pkl")
    }
    
    for name, p in paths.items():
        if not os.path.exists(p):
            st.error(f"🚨 Missing file: {p}")
            st.stop()
            
    try:
        resources = {k: joblib.load(v) for k, v in paths.items()}
        return resources
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

data = load_resources()
model = data["model"]
scaler = data["scaler"]
model_features = data["columns"]
freq_maps = data["freq_maps"]
unique_vals = data["unique_vals"]

# ==========================================
# 4. SIDEBAR - MODEL INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9322/9322127.png", width=80)
    st.title("GuardianAI")
    st.markdown("### 🟢 System Online")
    
    st.info(
        f"""
        **Model Architecture:** Random Forest  
        **Features:** {len(model_features)} inputs  
        **Tracking:** MLflow Active  
        **Environment:** Production  
        """
    )
    st.markdown("---")
    st.write("Usage Guide:")
    st.caption("1. Enter financial details.")
    st.caption("2. Verify transaction context.")
    st.caption("3. Click Analyze to run inference.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("Transaction Risk Inspector")
    st.markdown("Real-time fraud detection engine.")

# --- FORM START ---
with st.form("main_form"):
    
    # --- ROW 1: MONEY ---
    st.subheader("💸 Financial Context")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        txn_amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0, step=10.0, format="%.2f")
    with c2:
        acc_balance = st.number_input("Current Balance", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    with c3:
        # Check if Age is used by model
        cust_age = 0
        if "Customer_Age" in scaler.feature_names_in_:
            cust_age = st.number_input("Customer Age", 18, 100, 35)
        else:
            st.write("") # Spacer

    # --- ROW 2: TIME ---
    st.subheader("📅 Temporal Details")
    c4, c5 = st.columns(2)
    with c4:
        hour = st.slider("Hour of Transaction (24h)", 0, 23, 14)
    with c5:
        weekday = st.selectbox("Day of Week", range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    # --- ROW 3: CATEGORICALS (Dynamic) ---
    st.subheader("🌍 Transaction Metadata")
    
    # Combine keys for iteration
    all_keys = sorted(list(set(list(freq_maps.keys()) + list(unique_vals.keys()))))
    
    # Create dynamic columns (2 per row)
    cat_cols = st.columns(2)
    user_categoricals = {}

    for i, col_name in enumerate(all_keys):
        # Clean label (e.g., "Device_Type" -> "Device Type")
        label = col_name.replace("_", " ")
        
        # Get options
        options = []
        if col_name in freq_maps:
            options = sorted(list(freq_maps[col_name].keys()))
        elif col_name in unique_vals:
            options = unique_vals[col_name]
            
        with cat_cols[i % 2]:
            user_categoricals[col_name] = st.selectbox(label, options)

    st.markdown("---")
    submitted = st.form_submit_button("🔍 ANALYZE RISK")

# ==========================================
# 6. INFERENCE & MLFLOW TRACKING
# ==========================================
if submitted:
    with st.spinner("Processing transaction vector..."):
        
        # 1. Prepare Input Vector
        input_vector = {col: 0 for col in model_features}
        
        # 2. Process Numerics (Scaling)
        raw_fin = {
            "Transaction_Amount": txn_amount,
            "Account_Balance": acc_balance,
            "Customer_Age": cust_age if "Customer_Age" in scaler.feature_names_in_ else 0
        }
        
        try:
            # Create DF aligned with scaler
            raw_df = pd.DataFrame([raw_fin])
            # Align columns
            final_raw = pd.DataFrame()
            for c in scaler.feature_names_in_:
                final_raw[c] = raw_df.get(c, 0)
            
            scaled = scaler.transform(final_raw)
            for idx, c in enumerate(scaler.feature_names_in_):
                input_vector[c] = scaled[0][idx]
                
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

        # 3. Process Time
        if "Transaction_Hour" in input_vector: input_vector["Transaction_Hour"] = hour
        if "Transaction_Weekday" in input_vector: input_vector["Transaction_Weekday"] = weekday

        # 4. Process Categoricals
        for col, val in user_categoricals.items():
            # Frequency Map
            if col in freq_maps:
                target = f"{col}_Freq"
                if target in input_vector:
                    input_vector[target] = freq_maps[col].get(val, 0)
            # One-Hot
            else:
                target = f"{col}_{val}"
                if target in input_vector:
                    input_vector[target] = 1

        # 5. Predict
        X_final = pd.DataFrame([input_vector])[model_features]
        prob = model.predict_proba(X_final)[0][1]
        is_fraud = prob > 0.5

        # ==========================================
        # 7. LOG TO MLFLOW
        # ==========================================
        try:
            with mlflow.start_run():
                # Log Inputs (Raw values make more sense for auditing)
                mlflow.log_params(raw_fin)
                mlflow.log_param("Hour", hour)
                mlflow.log_params(user_categoricals)
                
                # Log Outputs
                mlflow.log_metric("fraud_probability", prob)
                mlflow.log_param("prediction_class", "Fraud" if is_fraud else "Legit")
                
                # Tag
                mlflow.set_tag("source", "streamlit_app")
                mlflow.set_tag("user", "bank_clerk_01")
                
        except Exception as e:
            st.warning(f"MLflow logging failed (Check server connection): {e}")

        # ==========================================
        # 8. DISPLAY RESULTS
        # ==========================================
        st.divider()
        
        # Banner Style Result
        if is_fraud:
            st.markdown("""
                <div style='background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 10px solid #f44336;'>
                    <h2 style='color: #b71c1c; margin:0;'>🚨 HIGH RISK TRANSACTION DETECTED</h2>
                    <p style='color: #b71c1c; margin:0;'>Do not approve. Escalation required.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 10px solid #4caf50;'>
                    <h2 style='color: #1b5e20; margin:0;'>✅ TRANSACTION APPROVED</h2>
                    <p style='color: #1b5e20; margin:0;'>Risk levels are within normal limits.</p>
                </div>
                """, unsafe_allow_html=True)

        # Metrics Column
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Fraud Probability", f"{prob:.1%}", delta=None)
        with m2:
            st.metric("Risk Level", "Critical" if prob > 0.8 else "Moderate" if prob > 0.5 else "Low")
        with m3:
            st.metric("Log ID", datetime.datetime.now().strftime("%H:%M:%S"))

        st.progress(prob)