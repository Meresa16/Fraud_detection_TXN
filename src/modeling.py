

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features (X) and target (y)."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    X = df.drop(columns=[target])
    y = df[target]
    logging.info(f"Split data: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
    stratify: Optional[pd.Series] = None, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    logging.info(f"Train-test split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    method: str = "standard",
    target_columns: Optional[List[str]] = None
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric features and RETURN THE SCALER.
    """
    if target_columns is None:
        num_cols = X_train.select_dtypes(include=['float64', 'float32']).columns.tolist()
    else:
        num_cols = [c for c in target_columns if c in X_train.columns]

    if not num_cols:
        logging.warning("No suitable numeric columns found to scale.")
        # Return None for scaler if no scaling happened
        return None, X_train.copy(), X_test.copy()
    
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit on Train, Transform on Train and Test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    logging.info(f"Scaled {len(num_cols)} columns using {method} scaler.")
    
    # CRITICAL: Return 3 values (scaler, train, test)
    return scaler, X_train_scaled, X_test_scaled


def train_simple_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    problem: str = "classification",
    model_name: str = "logistic",
    random_state: int = 42,
    resample: bool = True,
    **kwargs
):
    """
    Train a model with optional SMOTE or Internal Balancing.
    """
    X_train_res, y_train_res = X_train.copy(), y_train.copy()
    
    # 1. SPECIAL CASE: Disable SMOTE for BalancedRandomForest
    if model_name == "balanced_random_forest":
        if resample:
            logging.info("Model is 'balanced_random_forest'; disabling external SMOTE.")
            resample = False 

    # 2. SMOTE Handling
    if problem == "classification" and resample:
        logging.info("Applying SMOTE to training data...")
        try:
            smote = SMOTE(random_state=random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied: New X_train={X_train_res.shape}")
        except ValueError as e:
            logging.warning(f"SMOTE failed, proceeding without: {e}")

    # 3. Model Initialization
    model = None
    
    if problem == "classification":
        if model_name == "logistic":
            kwargs.setdefault("class_weight", "balanced" if not resample else None)
            model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
        
        elif model_name == "random_forest":
            kwargs.setdefault("class_weight", "balanced" if not resample else None)
            model = RandomForestClassifier(random_state=random_state, **kwargs)
        
        elif model_name == "balanced_random_forest":
            kwargs.setdefault("sampling_strategy", "auto")
            kwargs.setdefault("replacement", True)
            model = BalancedRandomForestClassifier(random_state=random_state, **kwargs)
            
        elif model_name == "gbm":
            model = GradientBoostingClassifier(random_state=random_state, **kwargs)
            
    else:  # regression
        if model_name == "linear":
            model = LinearRegression(**kwargs)
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=random_state, **kwargs)
    
    # 4. Final Safety Check
    if model is None:
        raise ValueError(f"Unsupported model '{model_name}' for problem '{problem}'")
        
    model.fit(X_train_res, y_train_res)
    logging.info(f"Trained {model_name} model for {problem}")

    # 5. Log Importances
    if hasattr(model, "feature_importances_"):
        try:
            importances = pd.Series(model.feature_importances_, index=X_train_res.columns)
            logging.info("Top 5 feature importances:\n" + str(importances.sort_values(ascending=False).head(5)))
        except:
            pass

    return model

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, problem: str = "classification"
) -> Dict[str, float]:
    """Evaluate a trained model and return metrics."""
    preds = model.predict(X_test)
    results: Dict[str, float] = {}

    if problem == "classification":
        results = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        }
        
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)[:, 1]
                if len(np.unique(y_test)) > 1:
                    results["roc_auc"] = roc_auc_score(y_test, probs)
                else:
                    results["roc_auc"] = np.nan
            except Exception as e:
                logging.warning(f"Could not calculate ROC AUC: {e}")
                results["roc_auc"] = np.nan
    else:
        mse = mean_squared_error(y_test, preds)
        results = {
            "mse": mse, 
            "rmse": np.sqrt(mse), 
            "r2": r2_score(y_test, preds)
        }

    logging.info(f"Evaluation results: {results}")
    return results