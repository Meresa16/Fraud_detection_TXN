# """
# dataset_utils.py

# Utility functions for working with CSV datasets in a Jupyter notebook or script.

# Usage:
#     from dataset_utils import (
#         load_csv, overview, clean_missing, encode_categorical,
#         split_X_y, scale_features, train_simple_model, evaluate_model, split_train_test
#     )
# """

# from typing import Optional, List, Tuple, Dict, Any
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     mean_squared_error, r2_score
# )

# __all__ = [
#     "load_csv",
#     "overview",
#     "clean_missing",
#     "encode_categorical",
#     "split_X_y",
#     "scale_features",
#     "train_simple_model",
#     "evaluate_model",
#     "split_train_test",
# ]


# def load_csv(path: str, nrows: Optional[int] = None, **read_kwargs) -> pd.DataFrame:
#     """Load a CSV into a pandas DataFrame."""
#     return pd.read_csv(path, nrows=nrows, **read_kwargs)


# def overview(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
#     """Return a simple overview of a DataFrame."""
#     return {
#         "shape": df.shape,
#         "dtypes": df.dtypes.to_dict(),
#         "head": df.head(n),
#         "missing": df.isnull().sum().sort_values(ascending=False).head(20)
#     }


# def clean_missing(
#     df: pd.DataFrame,
#     drop_threshold: float = 0.5,
#     fill_method: Optional[str] = None,
#     fill_value: Optional[Any] = None
# ) -> pd.DataFrame:
#     """
#     Clean missing values in a DataFrame.

#     - Drops columns where fraction of missing values > drop_threshold.
#     - If fill_method is 'ffill' or 'bfill', uses DataFrame.fillna(method=...).
#     - If fill_value is provided, fills remaining NA with that value.
#     """
#     df = df.copy()
#     frac_missing = df.isnull().mean()
#     drop_cols = frac_missing[frac_missing > drop_threshold].index.tolist()
#     if drop_cols:
#         df.drop(columns=drop_cols, inplace=True)

#     if fill_method in ("ffill", "bfill"):
#         df.fillna(method=fill_method, inplace=True)

#     if fill_value is not None:
#         df.fillna(fill_value, inplace=True)

#     return df


# def encode_categorical(
#     df: pd.DataFrame,
#     columns: Optional[List[str]] = None,
#     drop_first: bool = True
# ) -> pd.DataFrame:
#     """One-hot encode categorical columns. If columns is None, auto-detect object/category dtype."""
#     df = df.copy()
#     if columns is None:
#         columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
#     if not columns:
#         return df
#     return pd.get_dummies(df, columns=columns, drop_first=drop_first)


# def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
#     """Split DataFrame into features X and target y."""
#     if target not in df.columns:
#         raise KeyError(f"target column '{target}' not found in DataFrame")
#     X = df.drop(columns=[target])
#     y = df[target]
#     return X, y


# def scale_features(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     method: str = "standard"
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Scale numeric features using StandardScaler or MinMaxScaler."""
#     scaler = StandardScaler() if method == "standard" else MinMaxScaler()
#     num_cols = X_train.select_dtypes(include=[np.number]).columns.intersection(
#         X_test.select_dtypes(include=[np.number]).columns
#     )
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()
#     if len(num_cols) == 0:
#         return X_train_scaled, X_test_scaled

#     X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
#     X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
#     return X_train_scaled, X_test_scaled


# def train_simple_model(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     problem: str = "classification",
#     model_name: str = "logistic",
#     random_state: int = 42,
#     **kwargs
# ):
#     """Train a simple model and return the fitted estimator."""
#     if problem == "classification":
#         if model_name == "logistic":
#             model = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
#         elif model_name == "random_forest":
#             model = RandomForestClassifier(random_state=random_state, **kwargs)
#         else:
#             raise ValueError("Unsupported classification model_name")
#     else:
#         if model_name == "linear":
#             model = LinearRegression(**kwargs)
#         elif model_name == "random_forest":
#             model = RandomForestRegressor(random_state=random_state, **kwargs)
#         else:
#             raise ValueError("Unsupported regression model_name")

#     model.fit(X_train, y_train)
#     return model


# def evaluate_model(
#     model,
#     X_test: pd.DataFrame,
#     y_test: pd.Series,
#     problem: str = "classification"
# ) -> Dict[str, float]:
#     """Evaluate model and return common metrics."""
#     preds = model.predict(X_test)
#     if problem == "classification":
#         unique_vals = np.unique(y_test)
#         binary = unique_vals.size == 2
#         if binary:
#             avg = "binary"
#         else:
#             avg = "macro"

#         return {
#             "accuracy": float(accuracy_score(y_test, preds)),
#             "precision": float(precision_score(y_test, preds, average=avg, zero_division=0)),
#             "recall": float(recall_score(y_test, preds, average=avg, zero_division=0)),
#             "f1": float(f1_score(y_test, preds, average=avg, zero_division=0)),
#         }
#     else:
#         mse = mean_squared_error(y_test, preds)
#         return {
#             "mse": float(mse),
#             "rmse": float(np.sqrt(mse)),
#             "r2": float(r2_score(y_test, preds)),
#         }


# def split_train_test(
#     X: pd.DataFrame,
#     y: pd.Series,
#     test_size: float = 0.2,
#     random_state: int = 42,
#     stratify: Optional[pd.Series] = None
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     """Convenience wrapper around sklearn.model_selection.train_test_split."""
#     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)