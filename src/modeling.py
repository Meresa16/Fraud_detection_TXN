# # src/modeling.py
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
# import pandas as pd
# import numpy as np
# from typing import Tuple, Dict, Any, Optional

# def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
#     if target not in df.columns:
#         raise KeyError(f"target column '{target}' not found in DataFrame")
#     X = df.drop(columns=[target])
#     y = df[target]
#     return X, y

# def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, stratify: Optional[pd.Series] = None, random_state: int = 42):
#     return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

# def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
#     scaler = StandardScaler() if method == "standard" else MinMaxScaler()
#     num_cols = X_train.select_dtypes(include=[float, int]).columns
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()
#     if len(num_cols) > 0:
#         X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
#         X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
#     return X_train_scaled, X_test_scaled

# def train_simple_model(X_train: pd.DataFrame, y_train: pd.Series, problem="classification", model_name="logistic", random_state=42, **kwargs) -> Any:
#     if problem == "classification":
#         if model_name == "logistic":
#             if "class_weight" not in kwargs:
#                 kwargs["class_weight"] = "balanced"
#             model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
#         elif model_name == "random_forest":
#             if "class_weight" not in kwargs:
#                 kwargs["class_weight"] = "balanced"
#             model = RandomForestClassifier(random_state=random_state, **kwargs)
#     else:
#         if model_name == "linear":
#             model = LinearRegression(**kwargs)
#         elif model_name == "random_forest":
#             model = RandomForestRegressor(random_state=random_state, **kwargs)
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem="classification") -> Dict[str, float]:
#     preds = model.predict(X_test)
#     if problem == "classification":
#         return {
#             "accuracy": float(accuracy_score(y_test, preds)),
#             "precision": float(precision_score(y_test, preds, zero_division=0)),
#             "recall": float(recall_score(y_test, preds, zero_division=0)),
#             "f1": float(f1_score(y_test, preds, zero_division=0))
#         }
#     else:
#         mse = mean_squared_error(y_test, preds)
#         return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, preds))}













# src/modeling.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y)
    """
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
    """
    Split data into train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )
    logging.info(f"Train-test split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric features using StandardScaler or MinMaxScaler
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        logging.warning("No numeric columns to scale")
        return X_train.copy(), X_test.copy()
    
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    logging.info(f"Scaled numeric columns: {num_cols.tolist()}")
    return X_train_scaled, X_test_scaled


def train_simple_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    problem: str = "classification",
    model_name: str = "logistic",
    random_state: int = 42,
    resample: bool = True,
    **kwargs
):
    """
    Train a simple model (logistic, random forest, linear regression) with optional resampling for imbalance
    """
    X_train_res, y_train_res = X_train.copy(), y_train.copy()
    
    # Handle imbalance for classification
    if problem == "classification" and resample:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"Applied SMOTE: X_train_res={X_train_res.shape}, y_train_res={y_train_res.shape}")

    # Initialize model
    if problem == "classification":
        if model_name == "logistic":
            kwargs.setdefault("class_weight", "balanced")
            model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
        elif model_name == "random_forest":
            kwargs.setdefault("class_weight", "balanced")
            model = RandomForestClassifier(random_state=random_state, **kwargs)
    else:  # regression
        if model_name == "linear":
            model = LinearRegression(**kwargs)
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=random_state, **kwargs)
    
    model.fit(X_train_res, y_train_res)
    logging.info(f"Trained {model_name} model for {problem}")

    # Log feature importances if applicable
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train_res.columns)
        logging.info("Top 10 feature importances:\n" + str(importances.sort_values(ascending=False).head(10)))

    return model


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, problem: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate a trained model (classification or regression)
    """
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
                results["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except ValueError:
                results["roc_auc"] = np.nan
    else:
        mse = mean_squared_error(y_test, preds)
        results = {"mse": mse, "rmse": np.sqrt(mse), "r2": r2_score(y_test, preds)}

    logging.info(f"Evaluation results: {results}")
    return results
