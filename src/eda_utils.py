# # src/eda_utils.py
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Optional, Dict, Any, List

# try:
#     from IPython.display import display
# except ImportError:
#     def display(obj):
#         print(obj)

# def overview(df: pd.DataFrame) -> Dict[str, Any]:
#     """Quick overview of a dataframe"""
#     return {
#         "shape": df.shape,
#         "dtypes": df.dtypes.value_counts(),
#         "sample_data": df.head(),
#         "missing": df.isnull().sum().sum()
#     }

# def eda_descriptive(df: pd.DataFrame, n_head: int = 5):
#     """Print descriptive EDA summaries"""
#     print(f"Shape: {df.shape}")
#     print("\nData Sample:", df.head(n_head))
#     print("\nColumn Data Types:\n", df.dtypes.value_counts())
    
#     missing = df.isnull().sum()
#     missing = missing[missing > 0].sort_values(ascending=False)
#     if not missing.empty:
#         print("\nColumns with Missing Values:\n", missing)
#     else:
#         print("\nNo missing values found.")

#     if "Is_Fraud" in df.columns:
#         print("\nTarget Class Balance (Is_Fraud):\n", df["Is_Fraud"].value_counts(normalize=True))

# def plot_target_distribution(df: pd.DataFrame, target: str = "Is_Fraud"):
#     if target not in df.columns:
#         return
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x=df[target])
#     plt.title(f"Target Distribution: {target}")
#     plt.show()

# def numeric_eda_plots(df: pd.DataFrame, target: str = "Is_Fraud", top_corr: int = 12):
#     nums = df.select_dtypes(include=[np.number])
#     if nums.shape[1] == 0:
#         print("No numeric columns to analyze.")
#         return

#     if target in nums.columns:
#         corr = nums.corr()[target].abs().sort_values(ascending=False).head(top_corr)
#         print("Top correlations with target:")
#         display(corr.to_frame())
#     else:
#         corrmat = nums.corr()
#         top_pairs = corrmat.abs().unstack().sort_values(ascending=False).drop_duplicates().head(top_corr)
#         print("Top numeric correlations (pairs):")
#         display(top_pairs.to_frame())

#     top_cols = nums.columns.drop([target]) if target in nums.columns else nums.columns
#     top_cols = top_cols[:6]

#     for c in top_cols:
#         plt.figure(figsize=(8, 4))
#         if target in df.columns:
#             ax = sns.boxplot(x=df[target].astype(str), y=df[c])
#             plt.title(f"{c} by {target}")
#             counts = df[target].astype(str).value_counts()
#             x_labels = [label.get_text() for label in ax.get_xticklabels()]
#             new_labels = [f"{lbl}\n(n={counts.get(lbl, 0)})" for lbl in x_labels]
#             ax.set_xticklabels(new_labels)
#         else:
#             ax = sns.histplot(df[c].dropna(), kde=True)
#             plt.title(c)
#             for container in ax.containers:
#                 ax.bar_label(container)
#         plt.tight_layout()
#         plt.show()

# def categorical_eda(df: pd.DataFrame, max_cols: int = 10):
#     cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
#     if not cats:
#         print("No categorical columns.")
#         return
#     sample = cats[:max_cols]
#     for c in sample:
#         vc = df[c].value_counts(dropna=False).head(15)
#         print(f"\nColumn: {c} (n_unique={df[c].nunique()})")
#         display(vc)
#         plt.figure(figsize=(8, 3))
#         sns.barplot(x=vc.values, y=[str(x) for x in vc.index])
#         plt.title(c)
#         plt.show()

# def plot_missing_heatmap(df: pd.DataFrame, sample_frac: float = 0.2):
#     if df.isnull().sum().sum() == 0:
#         print("No missing values found.")
#         return
#     df_plot = df.sample(frac=sample_frac, random_state=42) if len(df) > 50000 else df
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(df_plot.isnull(), cbar=False, yticklabels=False, cmap='viridis')
#     plt.title("Missing Values Map (Yellow = Missing)")
#     plt.xlabel("Columns")
#     plt.show()



# # ======================================================
# # Fraud-Specific EDA Utilities
# # ======================================================

# def class_imbalance_report(df: pd.DataFrame, target: str = "Is_Fraud"):
#     """Print class imbalance metrics"""
#     if target not in df.columns:
#         print(f"{target} not found.")
#         return
#     vc = df[target].value_counts()
#     ratio = vc.min() / vc.max()
#     print("Class Distribution:")
#     display(vc)
#     print(f"Minority/Majority Ratio: {ratio:.4f}")


# def fraud_rate_by_feature(
#     df: pd.DataFrame,
#     feature: str,
#     target: str = "Is_Fraud",
#     top_n: int = 10
# ):
#     """Fraud rate by categorical feature"""
#     if feature not in df.columns or target not in df.columns:
#         return
#     stats = (
#         df.groupby(feature)[target]
#         .agg(["mean", "count"])
#         .sort_values("mean", ascending=False)
#         .head(top_n)
#     )
#     print(f"Fraud rate by {feature}")
#     display(stats)

#     plt.figure(figsize=(8, 4))
#     sns.barplot(x=stats["mean"], y=stats.index)
#     plt.xlabel("Fraud Rate")
#     plt.title(f"Fraud Rate by {feature}")
#     plt.show()


# def fraud_by_time(
#     df: pd.DataFrame,
#     hour_col: str = "Transaction_Hour",
#     weekday_col: str = "Transaction_Weekday",
#     target: str = "Is_Fraud"
# ):
#     """Fraud patterns by time"""
#     if hour_col in df.columns:
#         plt.figure(figsize=(8, 3))
#         df.groupby(hour_col)[target].mean().plot(kind="bar")
#         plt.title("Fraud Rate by Hour")
#         plt.ylabel("Fraud Rate")
#         plt.show()

#     if weekday_col in df.columns:
#         plt.figure(figsize=(8, 3))
#         df.groupby(weekday_col)[target].mean().plot(kind="bar")
#         plt.title("Fraud Rate by Weekday (0=Mon)")
#         plt.ylabel("Fraud Rate")
#         plt.show()


# def amount_outlier_analysis(
#     df: pd.DataFrame,
#     amount_col: str = "Transaction_Amount",
#     target: str = "Is_Fraud"
# ):
#     """Analyze transaction amount outliers"""
#     if amount_col not in df.columns:
#         return

#     plt.figure(figsize=(8, 4))
#     if target in df.columns:
#         sns.boxplot(x=df[target].astype(str), y=df[amount_col])
#         plt.title("Transaction Amount by Fraud Label")
#     else:
#         sns.boxplot(y=df[amount_col])
#         plt.title("Transaction Amount Distribution")

#     plt.yscale("log")
#     plt.show()

#     q1 = df[amount_col].quantile(0.25)
#     q3 = df[amount_col].quantile(0.75)
#     iqr = q3 - q1
#     threshold = q3 + 1.5 * iqr

#     print(f"High-value threshold (IQR): {threshold:.2f}")
#     print("Fraud rate above threshold:")
#     display(df[df[amount_col] > threshold][target].mean())


# def feature_cardinality(df: pd.DataFrame, max_unique: int = 30):
#     """Check categorical feature cardinality"""
#     cats = df.select_dtypes(include=["object", "category"]).columns
#     report = {
#         c: df[c].nunique()
#         for c in cats
#         if df[c].nunique() <= max_unique
#     }
#     print("Categorical Feature Cardinality:")
#     display(pd.Series(report).sort_values(ascending=False))










# src/eda_utils_prod.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# General EDA
# -----------------------------

def overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a quick overview of a dataframe."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "sample_data": df.head(5),
        "missing_total": df.isnull().sum().sum()
    }


def eda_descriptive(df: pd.DataFrame, n_head: int = 5) -> Dict[str, Any]:
    """Return descriptive statistics for the dataframe."""
    desc = {
        "shape": df.shape,
        "sample_data": df.head(n_head),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing": df.isnull().sum().to_dict()
    }
    if "Is_Fraud" in df.columns:
        desc["target_distribution"] = df["Is_Fraud"].value_counts(normalize=True).to_dict()
    return desc


def numeric_eda(df: pd.DataFrame, target: Optional[str] = "Is_Fraud", top_corr: int = 12
               ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Analyze numeric columns. Return top correlations with target (if available)
    and correlation matrix for numeric features.
    """
    nums = df.select_dtypes(include=[np.number])
    if nums.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    corr_matrix = nums.corr()
    target_corr = None
    if target in nums.columns:
        target_corr = corr_matrix[target].abs().sort_values(ascending=False).head(top_corr).to_frame()
    return target_corr, corr_matrix


def categorical_summary(df: pd.DataFrame, max_cols: int = 10) -> Dict[str, pd.Series]:
    """
    Return value counts for top categorical columns.
    """
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary = {}
    for c in cats[:max_cols]:
        summary[c] = df[c].value_counts(dropna=False)
    return summary


def missing_heatmap(df: pd.DataFrame, sample_frac: float = 0.2, plot: bool = True) -> Optional[plt.Figure]:
    """
    Return a missing value heatmap figure.
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found.")
        return None
    
    df_plot = df.sample(frac=sample_frac, random_state=42) if len(df) > 50000 else df
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_plot.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
        ax.set_title("Missing Values Map (Yellow = Missing)")
        ax.set_xlabel("Columns")
        return fig
    return None


# -----------------------------
# Fraud-Specific EDA
# -----------------------------

def class_imbalance(df: pd.DataFrame, target: str = "Is_Fraud") -> Dict[str, Any]:
    """
    Return class distribution metrics.
    """
    if target not in df.columns:
        return {}
    vc = df[target].value_counts()
    ratio = vc.min() / vc.max()
    return {"counts": vc.to_dict(), "minority_ratio": ratio}


def fraud_rate_by_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "Is_Fraud",
    top_n: int = 10,
    plot: bool = True
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Compute fraud rate by a categorical feature.
    """
    if feature not in df.columns or target not in df.columns:
        return pd.DataFrame(), None
    
    stats = (
        df.groupby(feature)[target]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=stats["mean"], y=stats.index, ax=ax)
        ax.set_xlabel("Fraud Rate")
        ax.set_title(f"Fraud Rate by {feature}")
    
    return stats, fig


def fraud_by_time(
    df: pd.DataFrame,
    hour_col: str = "Transaction_Hour",
    weekday_col: str = "Transaction_Weekday",
    target: str = "Is_Fraud",
    plot: bool = True
) -> Dict[str, Optional[plt.Figure]]:
    """
    Return fraud rate over hours and weekdays.
    """
    figs = {"hour": None, "weekday": None}
    
    if hour_col in df.columns and plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        df.groupby(hour_col)[target].mean().plot(kind="bar", ax=ax)
        ax.set_title("Fraud Rate by Hour")
        ax.set_ylabel("Fraud Rate")
        figs["hour"] = fig

    if weekday_col in df.columns and plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        df.groupby(weekday_col)[target].mean().plot(kind="bar", ax=ax)
        ax.set_title("Fraud Rate by Weekday (0=Mon)")
        ax.set_ylabel("Fraud Rate")
        figs["weekday"] = fig
    
    return figs


def amount_outlier_analysis(
    df: pd.DataFrame,
    amount_col: str = "Transaction_Amount",
    target: str = "Is_Fraud",
    plot: bool = True
) -> Dict[str, Any]:
    """
    Analyze outliers in transaction amount and return statistics.
    """
    if amount_col not in df.columns:
        return {}
    
    q1 = df[amount_col].quantile(0.25)
    q3 = df[amount_col].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    high_fraud_rate = df[df[amount_col] > threshold][target].mean() if target in df.columns else None
    
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        if target in df.columns:
            sns.boxplot(x=df[target].astype(str), y=df[amount_col], ax=ax)
            ax.set_title("Transaction Amount by Fraud Label")
        else:
            sns.boxplot(y=df[amount_col], ax=ax)
            ax.set_title("Transaction Amount Distribution")
        ax.set_yscale("log")
    
    return {"iqr_threshold": threshold, "high_value_fraud_rate": high_fraud_rate, "figure": fig}


def feature_cardinality(df: pd.DataFrame, max_unique: int = 30) -> Dict[str, int]:
    """
    Return the cardinality of categorical features.
    """
    cats = df.select_dtypes(include=["object", "category"]).columns
    return {c: df[c].nunique() for c in cats if df[c].nunique() <= max_unique}
