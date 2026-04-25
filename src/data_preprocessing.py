"""
data_preprocessing.py
---------------------------------------------------------------
Customer Risk Prediction
Handles: loading, cleaning, encoding, feature engineering,
         scaling, and train/test splitting.
---------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Load ──────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV from disk and print basic shape info."""
    df = pd.read_csv(filepath)
    print(f"[load]  {df.shape[0]:,} rows  |  {df.shape[1]} columns")
    return df


# ── 2. Clean ─────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop exact duplicate rows
    - Fill numeric NaNs with column median
    - Fill categorical NaNs with column mode
    """
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] Dropped {before - len(df)} duplicate rows.")

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[clean] {df.shape[0]:,} rows remain after cleaning.")
    return df


# ── 3. Encode ────────────────────────────────────────────────

def encode_features(df: pd.DataFrame, target: str = "risk_label") -> pd.DataFrame:
    """
    Label-encode every categorical column except the target.
    Each column gets its own LabelEncoder (fit on that column only).
    """
    for col in df.select_dtypes(include=["object"]).columns:
        if col == target:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[encode] {col}  →  {le.classes_.tolist()[:6]} ...")
    return df


# ── 4. Feature Engineering ───────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain-relevant ratio features.
    Only added when the source columns are present so the function
    is safe to run on any version of the dataset.
    """
    if {"credit_limit", "credit_used"}.issubset(df.columns):
        df["utilisation_rate"] = df["credit_used"] / (df["credit_limit"] + 1)
        print("[feat]  Added: utilisation_rate")

    if {"missed_payments", "total_payments"}.issubset(df.columns):
        df["payment_failure_rate"] = (
            df["missed_payments"] / (df["total_payments"] + 1)
        )
        print("[feat]  Added: payment_failure_rate")

    if {"transaction_amount", "avg_monthly_spend"}.issubset(df.columns):
        df["spend_deviation"] = (
            df["transaction_amount"] - df["avg_monthly_spend"]
        ) / (df["avg_monthly_spend"] + 1)
        print("[feat]  Added: spend_deviation")

    return df


# ── 5. Split & Scale ─────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    target: str = "risk_label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    1. Separate features from target
    2. Fit StandardScaler on train set only (no leakage)
    3. Stratified split to preserve class balance

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_scaler
    """
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[split] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[split] Class balance (train):\n{y_train.value_counts(normalize=True).round(3)}")
    return X_train, X_test, y_train, y_test, scaler


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    RAW_PATH = "data/raw/customer_data.csv"
    OUT_PATH = "data/processed/customer_data_clean.csv"
    os.makedirs("data/processed", exist_ok=True)

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = encode_features(df)
    df = engineer_features(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[done]  Cleaned data saved to {OUT_PATH}")
