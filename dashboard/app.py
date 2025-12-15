# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # =========================
# # 1. Configuration & Paths
# # =========================
# st.set_page_config(
#     page_title="Fraud Prediction",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # Note: Adjust these path names to match exactly where your main.py saved them
# # In the previous step, we saved to ../models/ and ../data/processed/
# DATA_DIR = os.path.join(BASE_DIR, "../data/processed")
# MODEL_DIR = os.path.join(BASE_DIR, "../model") 

# FEATURES_DATA_PATH = os.path.join(DATA_DIR, "bank_transactions_clean.csv") # Used for dropdown lists & freq maps
# MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl") # Or "rf_model_imbalanced.pkl"
# COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")

# # =========================
# # 2. Load Resources (Cached)
# # =========================
# @st.cache_resource
# def load_model_and_metadata():
#     if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
#         st.error(f"Model files not found. Please run the training pipeline first.\nExpected: {MODEL_PATH}")
#         return None, None
    
#     model = joblib.load(MODEL_PATH)
#     cols = joblib.load(COLUMNS_PATH)
#     return model, cols

# @st.cache_data
# def load_reference_data():
#     """Load clean data to get unique values for dropdowns and compute frequency maps."""
#     if not os.path.exists(FEATURES_DATA_PATH):
#         st.error(f"Data file not found: {FEATURES_DATA_PATH}")
#         return None, None, None
    
#     df = pd.read_csv(FEATURES_DATA_PATH)
    
#     # Generate Frequency Maps (Must match logic in preprocessing.py)
#     freq_maps = {}
#     for col in ["City", "Bank_Branch", "Transaction_Location"]:
#         if col in df.columns: # Clean data might already have these dropped if fully processed
#             # If the clean CSV has the original columns, we compute the map
#             freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
#         elif f"{col}_Freq" in df.columns:
#             # If clean CSV only has the _Freq column, we can't reconstruct the map easily 
#             # without the original labels. 
#             # Assuming 'bank_transactions_clean.csv' from step 3 (which dropped originals? 
#             # No, preprocessing step 3 usually drops them. 
#             # If they are dropped, we can't make a dropdown. 
#             # *Fallback*: We assume the CSV loaded here still has categorical columns 
#             # OR we rely on a separate mapping file.
#             pass

#     return df, freq_maps

# model, columns_used = load_model_and_metadata()
# df_clean, freq_maps = load_reference_data()

# # =========================
# # 3. Sidebar Inputs
# # =========================
# st.sidebar.header("📝 Transaction Details")

# # -- Numeric Inputs --
# txn_amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=250.0, step=10.0)
# account_balance = st.sidebar.number_input("Current Account Balance ($)", min_value=0.0, value=5000.0, step=100.0)

# # -- Time Inputs --
# txn_hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)
# txn_weekday = st.sidebar.selectbox("Day of Week", 
#     options=[0,1,2,3,4,5,6], 
#     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
#     index=2
# )

# # -- Categorical Inputs (Dynamic Dropdowns) --
# # We try to get unique lists from df_clean, otherwise fall back to static lists
# def get_unique(col_name, default_list):
#     if df_clean is not None and col_name in df_clean.columns:
#         return df_clean[col_name].unique().tolist()
#     return default_list

# cities = get_unique("City", ["New York", "London", "Paris", "Tokyo"])
# devices = get_unique("Transaction_Device", ["Mobile", "Desktop", "Tablet", "POS"])
# types = get_unique("Transaction_Type", ["Purchase", "Withdrawal", "Transfer"])

# selected_city = st.sidebar.selectbox("City", cities)
# selected_device = st.sidebar.selectbox("Transaction Device", devices)
# selected_type = st.sidebar.selectbox("Transaction Type", types)

# # =========================
# # 4. Feature Engineering for Inference
# # =========================
# if st.button("Predict Fraud Probability"):
#     if model is None:
#         st.error("Model not loaded.")
#         st.stop()

#     # A. Initialize Input Dictionary with Default 0s for all model columns
#     input_data = {col: 0 for col in columns_used}
    
#     # B. Fill Numeric Features
#     # Note: Ensure these names match EXACTLY what was in X_train
#     input_data["Transaction_Amount"] = txn_amount
#     input_data["Transaction_Hour"] = txn_hour
#     input_data["Transaction_Weekday"] = txn_weekday
#     input_data["Account_Balance"] = account_balance
    
#     # C. Handle Frequency Encoding
#     # If the model uses 'City_Freq', we must look up the frequency of the selected city
#     if "City_Freq" in columns_used:
#         # Look up in our map, default to 0.0 or a low probability if unknown
#         city_freq = freq_maps.get("City", {}).get(selected_city, 0.001) 
#         input_data["City_Freq"] = city_freq

#     # D. Handle One-Hot Encoding
#     # The model likely has columns like 'Transaction_Device_Mobile', 'Transaction_Device_POS'
    
#     # Construct the expected column name
#     device_col = f"Transaction_Device_{selected_device}"
#     type_col = f"Transaction_Type_{selected_type}"
    
#     # Set to 1 if it exists in the model features
#     if device_col in input_data:
#         input_data[device_col] = 1
#     if type_col in input_data:
#         input_data[type_col] = 1

#     # =========================
#     # 5. Prediction
#     # =========================
#     # Convert dict to DataFrame with correct order
#     X_new = pd.DataFrame([input_data])
#     X_new = X_new[columns_used] # Enforce column order

#     # Predict
#     try:
#         fraud_prob = model.predict_proba(X_new)[0][1]
#         prediction = model.predict(X_new)[0]
#     except Exception as e:
#         st.error(f"Prediction Error: {e}")
#         st.stop()

#     # =========================
#     # 6. Display Results
#     # =========================
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         if fraud_prob > 0.5:
#             st.error("### 🚨 High Risk")
#             st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
#         else:
#             st.success("### ✅ Low Risk")
#             st.image("https://cdn-icons-png.flaticon.com/512/148/148767.png", width=100)
            
#     with col2:
#         st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        
#         st.progress(float(fraud_prob))
        
#         st.markdown("#### Input Summary")
#         st.json({
#             "Amount": f"${txn_amount}",
#             "Location": selected_city,
#             "Device": selected_device,
#             "Type": selected_type
#         })
# else:
#     st.info("Adjust settings in the sidebar and click Predict.")


















# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # ==========================================
# # 1. SETUP & CONFIG
# # ==========================================
# st.set_page_config(page_title="Dynamic Fraud AI", page_icon="🧠", layout="wide")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "../model")

# # File Paths
# MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl")
# SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
# COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")
# FREQ_MAPS_PATH = os.path.join(MODEL_DIR, "freq_maps.pkl") # Contains mappings for City, etc.

# # ==========================================
# # 2. LOAD RESOURCES
# # ==========================================
# @st.cache_resource
# def load_resources():
#     required = [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, FREQ_MAPS_PATH]
#     for f in required:
#         if not os.path.exists(f):
#             st.error(f"Missing critical file: {f}")
#             st.stop()
            
#     try:
#         model = joblib.load(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)
#         cols = joblib.load(COLUMNS_PATH)
#         freq_maps = joblib.load(FREQ_MAPS_PATH)
#         return model, scaler, cols, freq_maps
#     except Exception as e:
#         st.error(f"Error loading files: {e}")
#         st.stop()

# model, scaler, model_features, freq_maps = load_resources()

# # ==========================================
# # 3. DYNAMIC FEATURE PARSING
# # ==========================================
# # We need to categorize the model's features to know what input widget to draw.

# def categorize_features(features, scaler, freq_maps):
#     """
#     Reverse-engineers the model columns into UI components.
#     """
#     specs = {
#         "numeric_scaled": [],
#         "numeric_raw": [],
#         "one_hot": {},      # { 'Transaction_Type': ['Debit', 'Credit', ...] }
#         "freq_encoded": []  # [ 'City', 'State' ] - derived from 'City_Freq'
#     }

#     # 1. Identify Scaled Numerics (from Scaler)
#     scaled_cols = list(scaler.feature_names_in_)
#     specs["numeric_scaled"] = scaled_cols

#     # 2. Loop through all model features to categorize the rest
#     for col in features:
#         # Skip if it's a scaled numeric (already handled)
#         if col in scaled_cols:
#             continue
            
#         # A. Check Frequency Encoding (endswith '_Freq')
#         if col.endswith("_Freq"):
#             base_name = col.replace("_Freq", "")
#             # Only add if we actually have the map for it
#             if base_name in freq_maps:
#                 specs["freq_encoded"].append(base_name)
#             else:
#                 # Fallback: Treat as raw numeric input if map is missing
#                 specs["numeric_raw"].append(col)
                
#         # B. Check One-Hot Encoding (contains underscore)
#         # Heuristic: We assume OHE columns look like "Category_Value"
#         elif "_" in col:
#             # We try to match prefixes against known raw data concepts or guessing
#             # This logic groups "Transaction_Type_Debit" and "Transaction_Type_Credit"
#             prefix, value = col.rsplit("_", 1)
            
#             # Simple heuristic: If the prefix appears multiple times or seems categorical
#             if prefix not in specs["one_hot"]:
#                 specs["one_hot"][prefix] = []
#             specs["one_hot"][prefix].append(value)
            
#         # C. Raw Numerics (e.g., Transaction_Hour if not scaled)
#         else:
#             specs["numeric_raw"].append(col)

#     return specs

# feature_specs = categorize_features(model_features, scaler, freq_maps)

# # ==========================================
# # 4. GENERATE DYNAMIC FORM
# # ==========================================
# st.title("🤖 Adaptive Model Dashboard")
# st.markdown("This form is generated **automatically** based on the features found in `columns_used.pkl`.")

# with st.form("prediction_form"):
#     st.subheader("1. Numeric Inputs")
#     col1, col2, col3 = st.columns(3)
#     user_inputs = {}

#     # --- Render Scaled Numerics ---
#     for i, col in enumerate(feature_specs["numeric_scaled"]):
#         with [col1, col2, col3][i % 3]:
#             user_inputs[col] = st.number_input(f"{col} (Scaled)", value=0.0)

#     # --- Render Raw Numerics ---
#     for i, col in enumerate(feature_specs["numeric_raw"]):
#         with [col1, col2, col3][i % 3]:
#             # Guessing reasonable defaults for time
#             val = 12 if "Hour" in col else 0
#             user_inputs[col] = st.number_input(col, value=val)

#     st.subheader("2. Categorical Inputs")
#     c1, c2 = st.columns(2)
    
#     # --- Render Frequency Encoded Inputs ---
#     # The model wants 'City_Freq', but we show 'City' dropdown using keys from freq_maps
#     for i, base_col in enumerate(feature_specs["freq_encoded"]):
#         options = sorted(list(freq_maps[base_col].keys()))
#         with [c1, c2][i % 2]:
#             selected = st.selectbox(f"{base_col}", options)
#             # Store the selected TEXT temporarily, we map it to float later
#             user_inputs[f"{base_col}_RAW"] = selected

#     # --- Render One-Hot Encoded Inputs ---
#     # The model wants 'Type_Debit'=1, but we show 'Type' dropdown
#     for i, (prefix, options) in enumerate(feature_specs["one_hot"].items()):
#         # We only show this group if it has > 1 option, otherwise it might be a boolean
#         if len(options) > 0:
#             with [c1, c2][i % 2]:
#                 selected = st.selectbox(f"{prefix}", sorted(options))
#                 user_inputs[f"{prefix}_RAW"] = selected

#     submit_btn = st.form_submit_button("🚀 Run Prediction")

# # ==========================================
# # 5. EXECUTION LOGIC
# # ==========================================
# if submit_btn:
#     # We construct the final vector initialized to zeros
#     input_vector = {col: 0 for col in model_features}

#     # --- A. Handle Scaled Numerics ---
#     try:
#         # Create DF for scaler
#         raw_scaled_df = pd.DataFrame([
#             {k: user_inputs[k] for k in feature_specs["numeric_scaled"]}
#         ])
#         # Transform
#         scaled_vals = scaler.transform(raw_scaled_df)
#         # Update vector
#         for idx, col in enumerate(feature_specs["numeric_scaled"]):
#             input_vector[col] = scaled_vals[0][idx]
#     except Exception as e:
#         st.error(f"Scaling failed: {e}")
#         st.stop()

#     # --- B. Handle Raw Numerics ---
#     for col in feature_specs["numeric_raw"]:
#         input_vector[col] = user_inputs[col]

#     # --- C. Handle Frequency Encoding ---
#     for base_col in feature_specs["freq_encoded"]:
#         # Get user selection (e.g., "New York")
#         selection = user_inputs[f"{base_col}_RAW"]
#         # Lookup float value (e.g., 0.45)
#         freq_val = freq_maps[base_col].get(selection, 0)
#         # Update vector (e.g., "City_Freq")
#         input_vector[f"{base_col}_Freq"] = freq_val

#     # --- D. Handle One-Hot Encoding ---
#     for prefix, options in feature_specs["one_hot"].items():
#         # Get user selection (e.g., "Mobile")
#         selection = user_inputs[f"{prefix}_RAW"]
#         # Construct specific column name (e.g., "Transaction_Device_Mobile")
#         target_col = f"{prefix}_{selection}"
        
#         # Set to 1 if it exists in the model
#         if target_col in input_vector:
#             input_vector[target_col] = 1

#     # --- E. Final Prediction ---
#     X_final = pd.DataFrame([input_vector])
#     # Enforce order
#     X_final = X_final[model_features]

#     try:
#         prediction = model.predict(X_final)[0]
#         prob = model.predict_proba(X_final)[0][1]

#         st.divider()
#         st.write("### 🔍 Model Decision")
        
#         c1, c2, c3 = st.columns([1,2,1])
#         with c2:
#             if prob > 0.5:
#                 st.error(f"**FRAUD DETECTED** ({prob:.2%})")
#             else:
#                 st.success(f"**LEGITIMATE TRANSACTION** ({prob:.2%})")
            
#             st.progress(prob)
            
#         with st.expander("See Technical Input Vector"):
#             st.dataframe(X_final)
            
#     except Exception as e:
#         st.error(f"Prediction Error: {e}")











# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # ==========================================
# # 1. SETUP & CONFIG
# # ==========================================
# st.set_page_config(page_title="Fraud Detection System", page_icon="🏦", layout="wide")

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
# # 3. UI GENERATION (LOOKS LIKE ORIGINAL DATA)
# # ==========================================
# st.title("🏦 Transaction Fraud Inspector")
# st.markdown("Enter the transaction details exactly as they appear in the banking system.")

# with st.form("transaction_form"):
    
#     # --- GROUP A: NUMERIC DETAILS ---
#     st.subheader("1. Financial Details")
#     c1, c2, c3 = st.columns(3)
    
#     # We dynamically find numeric columns that were scaled
#     scaled_cols = list(scaler.feature_names_in_)
    
#     user_numerics = {}
    
#     # Iterate through scaled columns (e.g., Amount, Balance, Age)
#     for i, col in enumerate(scaled_cols):
#         with [c1, c2, c3][i % 3]:
#             # Clean up the label (e.g., "Transaction_Amount" -> "Transaction Amount")
#             label = col.replace("_", " ")
#             user_numerics[col] = st.number_input(label, min_value=0.0, value=0.0)

#     # --- GROUP B: TIME DETAILS ---
#     st.subheader("2. Time Details")
#     t1, t2 = st.columns(2)
#     with t1:
#         hour = st.slider("Transaction Hour (24h)", 0, 23, 12)
#     with t2:
#         weekday = st.selectbox("Day of Week", range(7), 
#                                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

#     # --- GROUP C: CATEGORICAL DETAILS ---
#     st.subheader("3. Transaction Context")
#     cat1, cat2 = st.columns(2)
    
#     user_categoricals = {}

#     # 1. FREQUENCY ENCODED FIELDS (City, State, etc.)
#     # These come from 'freq_maps.pkl'
#     all_categorical_keys = list(freq_maps.keys()) + list(unique_vals.keys())
#     # Remove duplicates if any
#     all_categorical_keys = sorted(list(set(all_categorical_keys)))

#     for i, col_name in enumerate(all_categorical_keys):
#         # Determine the options list
#         options = []
#         if col_name in freq_maps:
#             options = sorted(list(freq_maps[col_name].keys()))
#         elif col_name in unique_vals:
#             options = unique_vals[col_name]
            
#         # Render Dropdown
#         with [cat1, cat2][i % 2]:
#             label = col_name.replace("_", " ")
#             user_categoricals[col_name] = st.selectbox(label, options)

#     # --- SUBMIT ---
#     st.markdown("---")
#     submitted = st.form_submit_button("🛡️ Analyze Transaction Risk", type="primary")


# # ==========================================
# # 4. EXECUTION LOGIC (MAP UI -> MODEL)
# # ==========================================
# if submitted:
    
#     # Initialize the input vector with Zeros for all 40+ model columns
#     input_vector = {col: 0 for col in model_features}
    
#     # --- STEP A: PROCESS NUMERICS ---
#     try:
#         # 1. Create a DataFrame for the Scaler
#         raw_num_df = pd.DataFrame([user_numerics])
#         # 2. Scale values
#         scaled_values = scaler.transform(raw_num_df)
#         # 3. Map back to input_vector
#         for idx, col in enumerate(scaled_cols):
#             input_vector[col] = scaled_values[0][idx]
#     except Exception as e:
#         st.error(f"Numeric Scaling Error: {e}")
#         st.stop()
        
#     # --- STEP B: PROCESS TIME ---
#     # These are usually raw numerics in the model
#     if "Transaction_Hour" in input_vector:
#         input_vector["Transaction_Hour"] = hour
#     if "Transaction_Weekday" in input_vector:
#         input_vector["Transaction_Weekday"] = weekday
        
#     # --- STEP C: PROCESS CATEGORICALS ---
#     for col_name, selected_val in user_categoricals.items():
        
#         # Case 1: Is it Frequency Encoded? (e.g., City -> City_Freq)
#         if col_name in freq_maps:
#             target_col = f"{col_name}_Freq"
#             # Lookup the frequency value (float)
#             freq_val = freq_maps[col_name].get(selected_val, 0)
            
#             if target_col in input_vector:
#                 input_vector[target_col] = freq_val
                
#         # Case 2: Is it One-Hot Encoded? (e.g., Device -> Device_Mobile)
#         # We need to construct the column name the model expects
#         else:
#             target_col_one_hot = f"{col_name}_{selected_val}"
            
#             if target_col_one_hot in input_vector:
#                 input_vector[target_col_one_hot] = 1

#     # --- STEP D: PREDICT ---
#     # Convert dict to DataFrame enforcing strict column order
#     X_final = pd.DataFrame([input_vector])
#     X_final = X_final[model_features]

#     try:
#         prob = model.predict_proba(X_final)[0][1]
#         is_fraud = prob > 0.5
        
#         st.divider()
#         r1, r2 = st.columns([1, 2])
        
#         with r1:
#             st.metric("Fraud Probability", f"{prob:.2%}")
        
#         with r2:
#             if is_fraud:
#                 st.error("🚨 **HIGH RISK TRANSACTION**")
#                 st.write("Reasoning: Pattern matches known fraud indicators.")
#             else:
#                 st.success("✅ **TRANSACTION APPROVED**")
#                 st.write("Reasoning: Pattern aligns with normal behavior.")
            
#             st.progress(prob)

#         # Debugging: Show what the model actually saw
#         with st.expander("View Processed Data (Internal Vector)"):
#             st.write("This is the mathematical representation sent to the Random Forest:")
#             st.dataframe(X_final)
            
#     except Exception as e:
#         st.error(f"Prediction Calculation Error: {e}")




















import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")

# File Paths
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_imbalanced.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns_used.pkl")
FREQ_MAPS_PATH = os.path.join(MODEL_DIR, "freq_maps.pkl") 
UNIQUE_VALS_PATH = os.path.join(MODEL_DIR, "unique_values.pkl")

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    required = [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, FREQ_MAPS_PATH, UNIQUE_VALS_PATH]
    for f in required:
        if not os.path.exists(f):
            st.error(f"Missing critical file: {f}. Please run the training pipeline.")
            st.stop()
            
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        cols = joblib.load(COLUMNS_PATH)
        freq_maps = joblib.load(FREQ_MAPS_PATH)
        unique_vals = joblib.load(UNIQUE_VALS_PATH)
        return model, scaler, cols, freq_maps, unique_vals
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

model, scaler, model_features, freq_maps, unique_vals = load_resources()

# ==========================================
# 3. UI GENERATION (HUMAN FRIENDLY)
# ==========================================
st.title("🛡️ Transaction Fraud Inspector")
st.markdown("Enter the transaction details below to assess risk.")

with st.form("transaction_form"):
    
    # --- SECTION 1: MONEY & ACCOUNT ---
    st.subheader("1. Financial Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Explicit, clear input for Amount
        txn_amount = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            value=150.00, 
            step=10.0,
            help="The total value of the transaction."
        )

    with col2:
        # Explicit, clear input for Balance
        acc_balance = st.number_input(
            "Current Account Balance ($)", 
            min_value=0.0, 
            value=5000.00, 
            step=100.0,
            help="Balance available before this transaction."
        )
        
    with col3:
        # Explicit input for Age (Check if your model uses it)
        # We check scaler.feature_names_in_ to see if Age is needed
        cust_age = 30 # Default
        if "Customer_Age" in scaler.feature_names_in_:
            cust_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)

    # --- SECTION 2: TIMING ---
    st.subheader("2. Time & Date")
    t1, t2 = st.columns(2)
    with t1:
        hour = st.slider("Hour of Day (24h format)", 0, 23, 14, help="0 = Midnight, 12 = Noon")
    with t2:
        weekday = st.selectbox("Day of Week", range(7), 
                               format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])

    # --- SECTION 3: CONTEXT (Location, Device, etc.) ---
    st.subheader("3. Transaction Details")
    cat1, cat2 = st.columns(2)
    
    user_categoricals = {}

    # We combine the maps to generate dropdowns automatically
    all_categorical_keys = sorted(list(set(list(freq_maps.keys()) + list(unique_vals.keys()))))

    for i, col_name in enumerate(all_categorical_keys):
        # Clean up the label for the user (e.g. "Transaction_Type" -> "Transaction Type")
        clean_label = col_name.replace("_", " ")
        
        # Get options
        options = []
        if col_name in freq_maps:
            options = sorted(list(freq_maps[col_name].keys()))
        elif col_name in unique_vals:
            options = unique_vals[col_name]

        with [cat1, cat2][i % 2]:
            user_categoricals[col_name] = st.selectbox(clean_label, options)

    # --- SUBMIT BUTTON ---
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")


# ==========================================
# 4. EXECUTION LOGIC (MAPPING UI -> MODEL)
# ==========================================
if submitted:
    
    # A. Initialize Input Vector with Zeros
    input_vector = {col: 0 for col in model_features}
    
    # B. Map the Manual Financial Inputs to the Scaler
    # We create a dictionary that matches the EXACT column names used in training
    raw_financials = {
        "Transaction_Amount": txn_amount,
        "Account_Balance": acc_balance,
        # Only add age if the model actually expects it
        "Customer_Age": cust_age if "Customer_Age" in scaler.feature_names_in_ else 0
    }
    
    try:
        # 1. Convert to DataFrame
        # We align this with scaler.feature_names_in_ to ensure correct order
        raw_num_df = pd.DataFrame([raw_financials])
        
        # Ensure we only pass columns the scaler knows about
        # (This handles cases where Age might be missing from training)
        final_raw_df = pd.DataFrame()
        for col in scaler.feature_names_in_:
            final_raw_df[col] = raw_num_df.get(col, 0) # Get value or 0
            
        # 2. Scale values
        scaled_values = scaler.transform(final_raw_df)
        
        # 3. Put scaled values into the input vector
        for idx, col in enumerate(scaler.feature_names_in_):
            input_vector[col] = scaled_values[0][idx]
            
    except Exception as e:
        st.error(f"Error processing financial details: {e}")
        st.stop()

    # C. Map Time
    if "Transaction_Hour" in input_vector: input_vector["Transaction_Hour"] = hour
    if "Transaction_Weekday" in input_vector: input_vector["Transaction_Weekday"] = weekday

    # D. Map Categoricals (The Hidden Logic)
    for col_name, selected_val in user_categoricals.items():
        # Case 1: Frequency Encoding (e.g., City)
        if col_name in freq_maps:
            target_col = f"{col_name}_Freq"
            val = freq_maps[col_name].get(selected_val, 0)
            if target_col in input_vector: input_vector[target_col] = val
                
        # Case 2: One-Hot Encoding (e.g., Device)
        else:
            target_col = f"{col_name}_{selected_val}"
            if target_col in input_vector: input_vector[target_col] = 1

    # E. Predict
    try:
        # Convert to DF to ensure column order is perfect
        X_final = pd.DataFrame([input_vector])[model_features]
        
        prob = model.predict_proba(X_final)[0][1]
        is_fraud = prob > 0.5
        
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric("Risk Score", f"{prob:.1%}")
            
        with col_res2:
            if is_fraud:
                st.error("🚨 **POTENTIAL FRAUD DETECTED**")
                st.write("This transaction has been flagged for immediate review.")
            else:
                st.success("✅ **TRANSACTION APPROVED**")
                st.write("This transaction appears legitimate.")
            
            st.progress(prob)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")