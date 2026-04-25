"""
app.py
---------------------------------------------------------------
Customer Risk Prediction — Streamlit Dashboard
---------------------------------------------------------------
Run with:
    streamlit run app.py
---------------------------------------------------------------
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

sys.path.append("src")
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #f8f9fb; }
    [data-testid="stSidebar"] { background-color: #1a1f36; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #4f46e5;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .metric-card.green  { border-left-color: #10b981; }
    .metric-card.red    { border-left-color: #ef4444; }
    .metric-card.amber  { border-left-color: #f59e0b; }
    .metric-label { font-size: 12px; color: #6b7280; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #111827; margin-top: 2px; }
    .metric-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }
    .section-title {
        font-size: 16px; font-weight: 700; color: #1e293b;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 14px;
    }
    .risk-high { background:#fef2f2; color:#b91c1c; padding:4px 12px;
                 border-radius:20px; font-weight:700; font-size:14px; }
    .risk-low  { background:#f0fdf4; color:#15803d; padding:4px 12px;
                 border-radius:20px; font-weight:700; font-size:14px; }
    .stButton>button {
        background: #4f46e5; color: white; border: none;
        border-radius: 8px; padding: 0.5rem 1.5rem;
        font-weight: 600; width: 100%;
    }
    .stButton>button:hover { background: #4338ca; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────

@st.cache_data
def load_data():
    path = "data/processed/customer_data_clean.csv"
    if not os.path.exists(path):
        raw = "data/raw/customer_data.csv"
        if not os.path.exists(raw):
            st.error("Dataset not found. Run `python generate_sample_data.py` first.")
            st.stop()
        from data_preprocessing import clean_data, encode_features, engineer_features
        df = pd.read_csv(raw)
        df = clean_data(df)
        df = encode_features(df)
        df = engineer_features(df)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    models = {}
    scaler = None
    if os.path.exists("models/random_forest.pkl"):
        models["Random Forest"] = joblib.load("models/random_forest.pkl")
    if os.path.exists("models/logistic_regression.pkl"):
        models["Logistic Regression"] = joblib.load("models/logistic_regression.pkl")
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
    return models, scaler


@st.cache_data
def get_split(df):
    from data_preprocessing import split_data
    return split_data(df)


def metric_card(label, value, sub="", color=""):
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 Risk Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 EDA", "🤖 Model Performance", "🎯 Predict Risk"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Project:** Customer Risk Prediction")
    st.markdown("**Author:** Mohammad Zaid")
    st.markdown("**Stack:** Python · scikit-learn · Streamlit")
    st.markdown("---")
    st.markdown("**Models available:**")
    models, scaler = load_models()
    if models:
        for m in models:
            st.markdown(f"✅ {m}")
    else:
        st.markdown("⚠️ No trained models found.")
        st.markdown("Run `python src/train_model.py` first.")


# ── Load data ─────────────────────────────────────────────────

df = load_data()
raw_df = pd.read_csv("data/raw/customer_data.csv") if os.path.exists("data/raw/customer_data.csv") else df


# ══════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════

if page == "📊 Overview":

    st.title("📊 Customer Risk Prediction Dashboard")
    st.markdown("End-to-end ML pipeline for identifying high-risk customers using Logistic Regression and Random Forest.")
    st.markdown("---")

    # ── KPI cards ──
    total     = len(raw_df)
    high_risk = int(raw_df["risk_label"].sum())
    low_risk  = total - high_risk
    risk_pct  = high_risk / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Customers", f"{total:,}", "in dataset")
    with col2:
        metric_card("High Risk", f"{high_risk:,}", f"{risk_pct:.1f}% of total", "red")
    with col3:
        metric_card("Low Risk", f"{low_risk:,}", f"{100-risk_pct:.1f}% of total", "green")
    with col4:
        avg_income = f"₹{int(raw_df['income'].mean()):,}"
        metric_card("Avg Income", avg_income, "across all customers", "amber")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Risk label distribution
    with col_left:
        st.markdown('<div class="section-title">Risk Label Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = raw_df["risk_label"].value_counts().sort_index()
        colors = ["#10b981", "#ef4444"]
        bars = ax.bar(["Low Risk (0)", "High Risk (1)"], counts.values, color=colors, width=0.5, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Employment vs Risk
    with col_right:
        st.markdown('<div class="section-title">Employment Status vs Risk</div>', unsafe_allow_html=True)
        emp_risk = raw_df.groupby("employment_status")["risk_label"].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(emp_risk.index, emp_risk.values,
                       color=["#ef4444" if v > 30 else "#f59e0b" if v > 15 else "#10b981" for v in emp_risk.values])
        for bar, val in zip(bars, emp_risk.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("High Risk %")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Region breakdown
    with col_a:
        st.markdown('<div class="section-title">Risk by Region</div>', unsafe_allow_html=True)
        region_risk = raw_df.groupby("region")["risk_label"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        region_risk.sort_values().plot(kind="barh", ax=ax, color="#4f46e5", edgecolor="white")
        ax.set_xlabel("High Risk %")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Age distribution by risk
    with col_b:
        st.markdown('<div class="section-title">Age Distribution by Risk Label</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color, name in [(0, "#10b981", "Low Risk"), (1, "#ef4444", "High Risk")]:
            ax.hist(raw_df[raw_df["risk_label"] == label]["age"],
                    bins=20, alpha=0.65, color=color, label=name, edgecolor="white")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════

elif page == "🔍 EDA":

    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    # Dataset preview
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(raw_df):,}")
    col2.metric("Columns", len(raw_df.columns))
    col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    corr = raw_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=10)
    fig.patch.set_facecolor("#f8f9fb")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature vs risk
    st.markdown('<div class="section-title">Feature Distribution vs Risk Label</div>', unsafe_allow_html=True)
    feat = st.selectbox("Select a feature to explore", numeric_cols)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    for label, color, name in [(0, "#10b981", "Low Risk"), (1, "#ef4444", "High Risk")]:
        axes[0].hist(raw_df[raw_df["risk_label"] == label][feat],
                     bins=25, alpha=0.65, color=color, label=name, edgecolor="white")
    axes[0].set_title(f"{feat} — Distribution by Risk")
    axes[0].set_xlabel(feat)
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Boxplot
    raw_df.boxplot(column=feat, by="risk_label", ax=axes[1],
                   boxprops=dict(color="#4f46e5"),
                   medianprops=dict(color="#ef4444", linewidth=2))
    axes[1].set_title(f"{feat} — Boxplot by Risk L
