"""
train_model.py
---------------------------------------------------------------
Customer Risk Prediction
Trains Logistic Regression and Random Forest classifiers with
5-fold GridSearchCV hyperparameter tuning.
Saves fitted models and a JSON results summary.
---------------------------------------------------------------
"""

import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

from data_preprocessing import split_data


# ── Hyperparameter Grids ─────────────────────────────────────

LR_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"],
}

RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"],
}


# ── Training Functions ───────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Grid-search over C, solver, and class_weight. Optimises ROC-AUC."""
    gs = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        LR_GRID,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"\n[LR]  Best params : {gs.best_params_}")
    print(f"[LR]  CV ROC-AUC  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Grid-search over n_estimators, max_depth, min_samples_split. Optimises ROC-AUC."""
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_GRID,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"\n[RF]  Best params : {gs.best_params_}")
    print(f"[RF]  CV ROC-AUC  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ── Evaluation Helper ────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str):
    """
    Print classification report and compute ROC-AUC.

    Returns
    -------
    report : dict  (from classification_report output_dict=True)
    auc    : float
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    if auc:
        print(f"  ROC-AUC : {auc:.4f}")

    return report, auc


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    PROCESSED = "data/processed/customer_data_clean.csv"
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(PROCESSED)
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    # ── Train ──
    print("\n── Logistic Regression ──────────────────────────")
    lr_model = train_logistic_regression(X_train, y_train)

    print("\n── Random Forest ────────────────────────────────")
    rf_model = train_random_forest(X_train, y_train)

    # ── Evaluate ──
    lr_report, lr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_report, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ── Save models ──
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")
    joblib.dump(scaler,   "models/scaler.pkl")
    print("\n[saved] logistic_regression.pkl  |  random_forest.pkl  |  scaler.pkl")

    # ── Save results summary ──
    results = {
        "logistic_regression": {"roc_auc": lr_auc, "report": lr_report},
        "random_forest":       {"roc_auc": rf_auc, "report": rf_report},
    }
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] models/results.json")
