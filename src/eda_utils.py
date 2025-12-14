# src/eda_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

def overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick overview of a dataframe"""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts(),
        "missing": df.isnull().sum().sum()
    }

def eda_descriptive(df: pd.DataFrame, n_head: int = 5):
    """Print descriptive EDA summaries"""
    print(f"Shape: {df.shape}")
    print("\nColumn Data Types:\n", df.dtypes.value_counts())
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print("\nColumns with Missing Values:\n", missing)
    else:
        print("\nNo missing values found.")

    if "Is_Fraud" in df.columns:
        print("\nTarget Class Balance (Is_Fraud):\n", df["Is_Fraud"].value_counts(normalize=True))

def plot_target_distribution(df: pd.DataFrame, target: str = "Is_Fraud"):
    if target not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[target])
    plt.title(f"Target Distribution: {target}")
    plt.show()

def numeric_eda_plots(df: pd.DataFrame, target: str = "Is_Fraud", top_corr: int = 12):
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] == 0:
        print("No numeric columns to analyze.")
        return

    if target in nums.columns:
        corr = nums.corr()[target].abs().sort_values(ascending=False).head(top_corr)
        print("Top correlations with target:")
        display(corr.to_frame())
    else:
        corrmat = nums.corr()
        top_pairs = corrmat.abs().unstack().sort_values(ascending=False).drop_duplicates().head(top_corr)
        print("Top numeric correlations (pairs):")
        display(top_pairs.to_frame())

    top_cols = nums.columns.drop([target]) if target in nums.columns else nums.columns
    top_cols = top_cols[:6]

    for c in top_cols:
        plt.figure(figsize=(8, 4))
        if target in df.columns:
            ax = sns.boxplot(x=df[target].astype(str), y=df[c])
            plt.title(f"{c} by {target}")
            counts = df[target].astype(str).value_counts()
            x_labels = [label.get_text() for label in ax.get_xticklabels()]
            new_labels = [f"{lbl}\n(n={counts.get(lbl, 0)})" for lbl in x_labels]
            ax.set_xticklabels(new_labels)
        else:
            ax = sns.histplot(df[c].dropna(), kde=True)
            plt.title(c)
            for container in ax.containers:
                ax.bar_label(container)
        plt.tight_layout()
        plt.show()

def categorical_eda(df: pd.DataFrame, max_cols: int = 10):
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cats:
        print("No categorical columns.")
        return
    sample = cats[:max_cols]
    for c in sample:
        vc = df[c].value_counts(dropna=False).head(15)
        print(f"\nColumn: {c} (n_unique={df[c].nunique()})")
        display(vc)
        plt.figure(figsize=(8, 3))
        sns.barplot(x=vc.values, y=[str(x) for x in vc.index])
        plt.title(c)
        plt.show()

def plot_missing_heatmap(df: pd.DataFrame, sample_frac: float = 0.2):
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
        return
    df_plot = df.sample(frac=sample_frac, random_state=42) if len(df) > 50000 else df
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_plot.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Values Map (Yellow = Missing)")
    plt.xlabel("Columns")
    plt.show()
