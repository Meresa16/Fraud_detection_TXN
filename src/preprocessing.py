



# import pandas as pd
# import numpy as np
# import logging
# from typing import List, Optional, Union

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def clean_missing(
#     df: pd.DataFrame, 
#     drop_threshold: float = 0.5, 
#     fill_method: Optional[str] = None, 
#     fill_value: Optional[Union[int, float, str]] = None
# ) -> pd.DataFrame:
#     """
#     Handle missing values by dropping columns with high missing rates 
#     and filling the rest.
#     """
#     df = df.copy()
    
#     # Drop columns with too many missing values
#     missing_mean = df.isnull().mean()
#     drop_cols = missing_mean[missing_mean > drop_threshold].index.tolist()
    
#     if drop_cols:
#         logging.info(f"Dropping columns with >{drop_threshold*100}% missing: {drop_cols}")
#         df.drop(columns=drop_cols, inplace=True)
    
#     # Fill remaining missing values
#     if fill_method == "ffill":
#         df = df.ffill()
#     elif fill_method == "bfill":
#         df = df.bfill()
#     elif fill_value is not None:
#         df = df.fillna(fill_value)
        
#     return df

# def drop_irrelevant_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
#     """Drop columns that are not needed for analysis."""
#     df = df.copy()
#     existing_cols = [c for c in columns_to_drop if c in df.columns]
#     if existing_cols:
#         logging.info(f"Dropping irrelevant columns: {existing_cols}")
#         df.drop(columns=existing_cols, inplace=True)
#     return df

# def feature_engineering_advanced(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Extract date/time features and perform frequency encoding on high-cardinality locations.
#     """
#     df = df.copy()
    
#     # Handle Transaction Time (assuming format "HH:MM:SS" or similar string)
#     if "Transaction_Time" in df.columns:
#         # Extract first 2 characters as hour
#         df["Transaction_Hour"] = pd.to_numeric(
#             df["Transaction_Time"].astype(str).str.split(':').str[0], 
#             errors="coerce"
#         ).fillna(0).astype(int)
#         df.drop(columns=["Transaction_Time"], inplace=True)
        
#     # Handle Transaction Date
#     if "Transaction_Date" in df.columns:
#         temp_date = pd.to_datetime(df["Transaction_Date"], errors="coerce")
#         df["Transaction_Day"] = temp_date.dt.day.fillna(0).astype(int)
#         df["Transaction_Weekday"] = temp_date.dt.dayofweek.fillna(0).astype(int)
#         df["Transaction_Month"] = temp_date.dt.month.fillna(0).astype(int)
#         df.drop(columns=["Transaction_Date"], inplace=True)
        
#     # Frequency encoding for specific high-cardinality columns
#     target_cols = ["City", "Bank_Branch", "Transaction_Location"]
#     for col in target_cols:
#         if col in df.columns:
#             freq_map = df[col].value_counts(normalize=True)
#             df[f"{col}_Freq"] = df[col].map(freq_map)
#             # We often drop the original if we are strictly using freq encoding, 
#             # but sometimes it's better to keep it if we plan to use other encoders later.
#             # Here we drop to match original logic.
#             df.drop(columns=[col], inplace=True)
            
#     return df

# def encode_categorical(
#     df: pd.DataFrame, 
#     columns: Optional[List[str]] = None, 
#     drop_first: bool = True, 
#     max_cardinality: int = 50
# ) -> pd.DataFrame:
#     """
#     One-Hot encode low cardinality categorical columns.
#     """
#     df = df.copy()
#     if columns is None:
#         columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
#     cols_to_encode = [c for c in columns if df[c].nunique() <= max_cardinality]
    
#     if cols_to_encode:
#         logging.info(f"One-hot encoding columns: {cols_to_encode}")
#         # dtype=int ensures outputs are 0/1 integers instead of True/False
#         df = pd.get_dummies(df, columns=cols_to_encode, drop_first=drop_first, dtype=int)
        
#     return df

















import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_missing(
    df: pd.DataFrame, 
    drop_threshold: float = 0.5, 
    fill_method: Optional[str] = None, 
    fill_value: Optional[Union[int, float, str]] = None
) -> pd.DataFrame:
    """
    Handle missing values by dropping columns with high missing rates 
    and filling the rest.
    """
    df = df.copy()
    
    # Drop columns with too many missing values
    missing_mean = df.isnull().mean()
    drop_cols = missing_mean[missing_mean > drop_threshold].index.tolist()
    
    if drop_cols:
        logging.info(f"Dropping columns with >{drop_threshold*100}% missing: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)
    
    # Fill remaining missing values
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_value is not None:
        df = df.fillna(fill_value)
        
    return df

def drop_irrelevant_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop columns that are not needed for analysis."""
    df = df.copy()
    existing_cols = [c for c in columns_to_drop if c in df.columns]
    if existing_cols:
        logging.info(f"Dropping irrelevant columns: {existing_cols}")
        df.drop(columns=existing_cols, inplace=True)
    return df

def feature_engineering_advanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Extract date/time features and perform frequency encoding.
    
    Returns:
        df: The transformed dataframe.
        freq_maps: A dictionary containing the frequency mappings (useful for inference).
    """
    df = df.copy()
    freq_maps = {} # We will store the maps here to return them
    
    # 1. Handle Transaction Time
    # Using to_datetime is safer than string splitting
    if "Transaction_Time" in df.columns:
        # Coerce errors allows handling inconsistent formats
        temp_time = pd.to_datetime(df["Transaction_Time"], format='%H:%M:%S', errors='coerce')
        # If format failed, try inferring
        if temp_time.isnull().all():
             temp_time = pd.to_datetime(df["Transaction_Time"], errors='coerce')
             
        df["Transaction_Hour"] = temp_time.dt.hour.fillna(0).astype(int)
        df.drop(columns=["Transaction_Time"], inplace=True)
        
    # 2. Handle Transaction Date
    if "Transaction_Date" in df.columns:
        # Try specific format first, fall back to auto-detect
        temp_date = pd.to_datetime(df["Transaction_Date"], format='%d-%m-%Y', errors='coerce')
        if temp_date.isnull().all():
            temp_date = pd.to_datetime(df["Transaction_Date"], errors='coerce')

        df["Transaction_Day"] = temp_date.dt.day.fillna(0).astype(int)
        df["Transaction_Weekday"] = temp_date.dt.dayofweek.fillna(0).astype(int)
        df["Transaction_Month"] = temp_date.dt.month.fillna(0).astype(int)
        df["Customer_Age"] = df["Age"] if "Age" in df.columns else 0 # Ensure Age is renamed/kept
        
        df.drop(columns=["Transaction_Date"], inplace=True)
        
    # 3. Frequency Encoding
    # We capture the map so we can save it later!
    target_cols = ["City", "Bank_Branch", "Transaction_Location", "State"]
    
    for col in target_cols:
        if col in df.columns:
            # Calculate the map (e.g., {'New York': 0.4, 'London': 0.1})
            freq_map = df[col].value_counts(normalize=True).to_dict()
            
            # Save it to our dictionary
            freq_maps[col] = freq_map
            
            # Map the column
            df[f"{col}_Freq"] = df[col].map(freq_map)
            
            # Drop the original text column
            df.drop(columns=[col], inplace=True)
            
    return df, freq_maps

def encode_categorical(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    drop_first: bool = True, 
    max_cardinality: int = 50
) -> pd.DataFrame:
    """
    One-Hot encode low cardinality categorical columns.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
    cols_to_encode = [c for c in columns if df[c].nunique() <= max_cardinality]
    
    if cols_to_encode:
        logging.info(f"One-hot encoding columns: {cols_to_encode}")
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=drop_first, dtype=int)
        
    return df