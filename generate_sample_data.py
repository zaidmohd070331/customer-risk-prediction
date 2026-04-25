"""
generate_sample_data.py
---------------------------------------------------------------
Generates a reproducible synthetic dataset for the
Customer Risk Prediction project and saves it to
data/raw/customer_data.csv
---------------------------------------------------------------
Run from the project root:
    python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 500
OUTPUT_PATH = "data/raw/customer_data.csv"

rng = np.random.default_rng(RANDOM_STATE)
os.makedirs("data/raw", exist_ok=True)


def generate_dataset(n: int) -> pd.DataFrame:

    # ── Demographics ───────────────────────────────────────────
    age               = rng.integers(20, 65, size=n)
    gender            = rng.choice(["M", "F"], size=n)
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed"],
        size=n, p=[0.60, 0.25, 0.15]
    )
    region            = rng.choice(["North", "South", "East", "West"], size=n)

    # ── Financials ─────────────────────────────────────────────
    income = np.where(
        employment_status == "Employed",
        rng.integers(40000, 120000, size=n),
        np.where(
            employment_status == "Self-Employed",
            rng.integers(20000, 80000, size=n),
            rng.integers(10000, 30000, size=n),
        )
    )

    credit_limit    = (income * rng.uniform(0.15, 0.55, size=n)).astype(int)
    credit_used     = np.clip(
        (credit_limit * rng.uniform(0.30, 1.05, size=n)).astype(int),
        0, credit_limit
    )

    account_age_months  = rng.integers(3, 180, size=n)
    total_payments      = np.maximum(account_age_months // 1, 1)
    missed_payments     = np.clip(
        rng.integers(0, 12, size=n), 0, total_payments
    )
    num_late_fees       = missed_payments + rng.integers(0, 4, size=n)
    num_products        = rng.integers(1, 8, size=n)
    loan_outstanding    = np.where(
        rng.random(n) < 0.6,
        rng.integers(0, 30000, size=n),
        0
    )

    # ── Transactions ───────────────────────────────────────────
    avg_monthly_spend   = (income / 12 * rng.uniform(0.10, 0.60, size=n)).astype(int)
    transaction_amount  = np.clip(
        (avg_monthly_spend * rng.uniform(0.50, 1.80, size=n)).astype(int),
        100, None
    )

    # ── Target: risk_label ────────────────────────────────────
    # High-risk conditions (each adds to a score):
    #   utilisation > 90 %
    #   missed_payments >= 4
    #   income < 30 000
    #   employment = Unemployed
    #   loan_outstanding > 15 000
    risk_score = (
        ((credit_used / (credit_limit + 1)) > 0.90).astype(int) * 2
        + (missed_payments >= 4).astype(int) * 2
        + (income < 30000).astype(int)
        + (employment_status == "Unemployed").astype(int)
        + (loan_outstanding > 15000).astype(int)
    )
    risk_label = (risk_score >= 3).astype(int)

    customer_ids = [f"C{str(i+1).zfill(4)}" for i in range(n)]

    df = pd.DataFrame({
        "customer_id":          customer_ids,
        "age":                  age,
        "gender":               gender,
        "income":               income,
        "credit_limit":         credit_limit,
        "credit_used":          credit_used,
        "missed_payments":      missed_payments,
        "total_payments":       total_payments,
        "num_products":         num_products,
        "account_age_months":   account_age_months,
        "transaction_amount":   transaction_amount,
        "avg_monthly_spend":    avg_monthly_spend,
        "num_late_fees":        num_late_fees,
        "loan_outstanding":     loan_outstanding,
        "employment_status":    employment_status,
        "region":               region,
        "risk_label":           risk_label,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset(N_SAMPLES)

    print("=== Dataset Summary ===")
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {df.columns.tolist()}")
    print(f"\nClass balance:")
    print(df["risk_label"].value_counts())
    print(f"\nClass ratio  : {df['risk_label'].mean():.1%} high-risk")
    print(f"\nSample rows  :")
    print(df.head(5).to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[saved] {OUTPUT_PATH}")
