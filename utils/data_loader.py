"""
Data loading, preprocessing, and feature engineering utilities.
"""

import pandas as pd
import numpy as np
import os


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "European_Bank.csv")


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(DATA_PATH)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Drop non-informative columns
    2. One-hot encode categoricals
    3. Feature engineering
    """
    df = df.copy()

    # Drop columns not used for modeling
    df.drop(columns=["Year", "CustomerId", "Surname"], inplace=True, errors="ignore")

    # One-hot encode Geography and Gender
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

    # ── Feature Engineering ──────────────────────────────────
    df["Balance_Salary_Ratio"]         = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Product_Density"]              = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Engagement_Product"]           = df["IsActiveMember"] * df["NumOfProducts"]
    df["Age_Tenure_Interaction"]       = df["Age"] * df["Tenure"]
    df["Zero_Balance"]                 = (df["Balance"] == 0).astype(int)
    df["Is_Senior"]                    = (df["Age"] > 50).astype(int)
    df["High_Balance"]                 = (df["Balance"] > df["Balance"].median()).astype(int)
    df["CreditScore_Band"]             = pd.cut(
        df["CreditScore"], bins=[300, 450, 550, 650, 750, 850],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(3)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (everything except Exited)."""
    return [c for c in df.columns if c != "Exited"]


def get_eda_dataframe() -> pd.DataFrame:
    """Return raw dataframe with extra derived columns for EDA."""
    df = load_raw_data()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )
    df["BalanceBucket"] = pd.cut(
        df["Balance"],
        bins=[-1, 0, 50_000, 100_000, 150_000, 300_000],
        labels=["Zero", "Low", "Medium", "High", "Very High"]
    )
    df["CreditBand"] = pd.cut(
        df["CreditScore"],
        bins=[300, 450, 550, 650, 750, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
    )
    df["ChurnLabel"] = df["Exited"].map({0: "Retained", 1: "Churned"})
    return df
