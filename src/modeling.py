# src/modeling.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import pandas as pd
from typing import Tuple, Dict, Any, Optional

def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, stratify: Optional[pd.Series] = None, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    num_cols = X_train.select_dtypes(include=[float, int]).columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if len(num_cols) > 0:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    return X_train_scaled, X_test_scaled

def train_simple_model(X_train: pd.DataFrame, y_train: pd.Series, problem="classification", model_name="logistic", random_state=42, **kwargs) -> Any:
    if problem == "classification":
        if model_name == "logistic":
            if "class_weight" not in kwargs:
                kwargs["class_weight"] = "balanced"
            model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
        elif model_name == "random_forest":
            if "class_weight" not in kwargs:
                kwargs["class_weight"] = "balanced"
            model = RandomForestClassifier(random_state=random_state, **kwargs)
    else:
        if model_name == "linear":
            model = LinearRegression(**kwargs)
        elif model_name == "random_forest":
            model = RandomForestRegressor(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem="classification") -> Dict[str, float]:
    preds = model.predict(X_test)
    if problem == "classification":
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0))
        }
    else:
        mse = mean_squared_error(y_test, preds)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, preds))}
