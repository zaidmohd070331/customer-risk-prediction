"""
evaluate.py
---------------------------------------------------------------
Customer Risk Prediction
Generates and saves:
  - Confusion matrix heatmap (per model)
  - ROC curve comparison chart
  - Feature importance bar chart (Random Forest)
---------------------------------------------------------------
Run after train_model.py has saved the .pkl files.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from data_preprocessing import split_data

os.makedirs("reports/figures", exist_ok=True)

# ── Plot style ────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── 1. Confusion Matrix ───────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """
    Saves a labelled confusion matrix heatmap to reports/figures/.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_")
    path = f"reports/figures/confusion_matrix_{fname}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ── 2. ROC Curve Comparison ───────────────────────────────────

def plot_roc_curve(models_dict: dict, X_test, y_test):
    """
    Overlays ROC curves for every model in models_dict on one chart.
    models_dict = {"Model Name": fitted_model, ...}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Model Comparison", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = "reports/figures/roc_curve_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ── 3. Feature Importance ─────────────────────────────────────

def plot_feature_importance(rf_model, feature_names: list, top_n: int = 15):
    """
    Bar chart of top-N features by mean impurity decrease (Random Forest).
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(top_n), top_values, color="#2196F3", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Mean impurity decrease")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, pad=12)

    # value labels on bars
    for bar, val in zip(bars, top_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    path = "reports/figures/feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ── 4. Class Distribution ─────────────────────────────────────

def plot_class_distribution(y_train, y_test):
    """
    Side-by-side bar chart showing class balance in train vs test sets.
    Useful for confirming stratified split worked correctly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, y, title in zip(axes, [y_train, y_test], ["Train set", "Test set"]):
        counts = y.value_counts().sort_index()
        ax.bar(["Low Risk", "High Risk"], counts.values, color=["#4CAF50", "#F44336"])
        ax.set_title(title)
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=10)
    fig.suptitle("Class Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    path = "reports/figures/class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load processed data and rebuild the same split used during training
    df = pd.read_csv("data/processed/customer_data_clean.csv")
    X_train, X_test, y_train, y_test, _ = split_data(df)
    feature_names = [c for c in df.columns if c != "risk_label"]

    # Load saved models
    lr = joblib.load("models/logistic_regression.pkl")
    rf = joblib.load("models/random_forest.pkl")

    # Generate all plots
    plot_class_distribution(y_train, y_test)

    plot_confusion_matrix(y_test, lr.predict(X_test), "Logistic Regression")
    plot_confusion_matrix(y_test, rf.predict(X_test), "Random Forest")

    plot_roc_curve({"Logistic Regression": lr, "Random Forest": rf}, X_test, y_test)

    plot_feature_importance(rf, feature_names, top_n=15)

    print("\n[done]  All plots saved to reports/figures/")
