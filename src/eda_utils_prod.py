
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# General EDA
# -----------------------------

def overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a quick overview of a dataframe metadata."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "duplicates": df.duplicated().sum(),
        "missing_total": df.isnull().sum().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 ** 2
    }

def eda_descriptive(df: pd.DataFrame, n_head: int = 5) -> Dict[str, Any]:
    """Return descriptive statistics and sample data."""
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
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze numeric columns. Return top correlations with target and full matrix.
    """
    nums = df.select_dtypes(include=[np.number])
    if nums.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    corr_matrix = nums.corr()
    target_corr = pd.DataFrame()
    
    if target in nums.columns:
        target_corr = corr_matrix[[target]].sort_values(by=target, ascending=False, key=abs).head(top_corr)
    
    return target_corr, corr_matrix

def categorical_summary(df: pd.DataFrame, max_cols: int = 10) -> Dict[str, pd.Series]:
    """Return value counts for top categorical columns."""
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    summary = {}
    for c in cats[:max_cols]:
        summary[c] = df[c].value_counts(dropna=False)
    return summary

def missing_heatmap(df: pd.DataFrame, sample_frac: float = 0.2) -> Optional[plt.Figure]:
    """Generate a heatmap of missing values."""
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values found.")
        return None
    
    # Downsample for plotting performance if dataframe is large
    df_plot = df.sample(frac=sample_frac, random_state=42) if len(df) > 50000 else df
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_plot.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Map (Yellow = Missing)")
    ax.set_xlabel("Columns")
    plt.tight_layout()
    return fig

# -----------------------------
# Fraud-Specific EDA
# -----------------------------

def class_imbalance(df: pd.DataFrame, target: str = "Is_Fraud") -> Dict[str, Any]:
    """Return class distribution metrics."""
    if target not in df.columns:
        return {}
    vc = df[target].value_counts()
    ratio = vc.min() / vc.max() if not vc.empty else 0
    return {"counts": vc.to_dict(), "minority_ratio": ratio}

def fraud_rate_by_feature(
    df: pd.DataFrame,
    feature: str,
    target: str = "Is_Fraud",
    top_n: int = 10,
    plot: bool = True
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """Compute and plot fraud rate by a specific feature."""
    if feature not in df.columns or target not in df.columns:
        return pd.DataFrame(), None
    
    stats = (
        df.groupby(feature)[target]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    stats.rename(columns={"mean": "Fraud Rate", "count": "Volume"}, inplace=True)
    
    fig = None
    if plot and not stats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=stats["Fraud Rate"], y=stats.index.astype(str), ax=ax, palette="Reds_r")
        ax.set_xlabel("Fraud Rate")
        ax.set_title(f"Top Fraud Rates by {feature}")
        plt.tight_layout()
    
    return stats, fig

def fraud_by_time(
    df: pd.DataFrame,
    hour_col: str = "Transaction_Hour",
    weekday_col: str = "Transaction_Weekday",
    target: str = "Is_Fraud"
) -> Dict[str, plt.Figure]:
    """Return fraud rate plots over hours and weekdays."""
    figs = {}
    
    if hour_col in df.columns and target in df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        hourly_data = df.groupby(hour_col)[target].mean()
        hourly_data.plot(kind="bar", ax=ax1, color='skyblue')
        ax1.set_title("Fraud Rate by Hour")
        ax1.set_ylabel("Fraud Rate")
        plt.tight_layout()
        figs["hour"] = fig1

    if weekday_col in df.columns and target in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        weekly_data = df.groupby(weekday_col)[target].mean()
        weekly_data.plot(kind="barh", ax=ax2, color='salmon')
        ax2.set_title("Fraud Rate by Weekday (0=Mon)")
        ax2.set_ylabel("Fraud Rate")
        plt.tight_layout()
        figs["weekday"] = fig2
    
    return figs

def amount_outlier_analysis(
    df: pd.DataFrame,
    amount_col: str = "Transaction_Amount",
    target: str = "Is_Fraud",
    plot: bool = True
) -> Dict[str, Any]:
    """Analyze outliers in transaction amount."""
    if amount_col not in df.columns:
        return {}
    
    q1 = df[amount_col].quantile(0.25)
    q3 = df[amount_col].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    
    high_fraud_rate = 0.0
    if target in df.columns:
        outliers = df[df[amount_col] > threshold]
        if not outliers.empty:
            high_fraud_rate = outliers[target].mean()
    
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        if target in df.columns:
            sns.boxplot(x=df[target].astype(str), y=df[amount_col], ax=ax)
            ax.set_title(f"{amount_col} Distribution by Fraud Label")
        else:
            sns.boxplot(y=df[amount_col], ax=ax)
            ax.set_title(f"{amount_col} Distribution")
        ax.set_yscale("log")
        plt.tight_layout()
    
    return {
        "iqr_threshold": threshold, 
        "high_value_fraud_rate": high_fraud_rate, 
        "figure": fig
    }

def feature_cardinality(df: pd.DataFrame, max_unique: int = 30) -> Dict[str, int]:
    """Return cardinality for categorical features with reasonable unique counts."""
    cats = df.select_dtypes(include=["object", "category"]).columns
    return {c: df[c].nunique() for c in cats if df[c].nunique() <= max_unique}