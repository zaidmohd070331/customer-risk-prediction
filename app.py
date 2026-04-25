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
    axes[1].set_title(f"{feat} — Boxplot by Risk Label")
    axes[1].set_xlabel("Risk Label (0=Low, 1=High)")
    plt.suptitle("")
    for ax in axes:
        ax.set_facecolor("#f8f9fb")
    fig.patch.set_facecolor("#f8f9fb")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Descriptive stats
    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")
    st.markdown("---")

    if not models:
        st.warning("No trained models found. Run `python src/train_model.py` first.")
        st.stop()

    X_train, X_test, y_train, y_test, _ = get_split(df)

    # Model selector
    model_name = st.selectbox("Select model", list(models.keys()))
    model = models[model_name]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_proba)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy",  f"{report['accuracy']:.1%}", model_name)
    with col2:
        metric_card("Precision", f"{report['1']['precision']:.1%}", "High risk class", "amber")
    with col3:
        metric_card("Recall",    f"{report['1']['recall']:.1%}", "High risk class", "red")
    with col4:
        metric_card("ROC-AUC",   f"{auc:.3f}", "Higher is better", "green")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low Risk", "High Risk"],
                    yticklabels=["Low Risk", "High Risk"],
                    linewidths=0.5, ax=ax)
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ROC curve
    with col_right:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors_map = {"Random Forest": "#4f46e5", "Logistic Regression": "#10b981"}
        for name, mdl in models.items():
            fp, tp, _ = roc_curve(y_test, mdl.predict_proba(X_test)[:, 1])
            a = roc_auc_score(y_test, mdl.predict_proba(X_test)[:, 1])
            ax.plot(fp, tp, label=f"{name} (AUC={a:.3f})",
                    color=colors_map.get(name, "#6366f1"), lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Feature importance (RF only)
    if model_name == "Random Forest":
        st.markdown('<div class="section-title">Feature Importances — Random Forest</div>', unsafe_allow_html=True)
        feature_names = [c for c in df.columns if c != "risk_label"]
        importances   = model.feature_importances_
        indices       = np.argsort(importances)[::-1][:15]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(range(15), importances[indices], color="#4f46e5", edgecolor="white")
        ax.set_xticks(range(15))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=40, ha="right")
        ax.set_ylabel("Mean impurity decrease")
        ax.set_title("Top 15 Feature Importances")
        for bar, val in zip(bars, importances[indices]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8f9fb")
        fig.patch.set_facecolor("#f8f9fb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Full classification report
    with st.expander("📋 Full Classification Report"):
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — Predict Risk
# ══════════════════════════════════════════════════════════════

elif page == "🎯 Predict Risk":

    st.title("🎯 Predict Customer Risk")
    st.markdown("Enter a customer's details below to get an instant risk prediction.")
    st.markdown("---")

    if not models or scaler is None:
        st.warning("Models not loaded. Run `python src/train_model.py` first.")
        st.stop()

    # Model choice
    model_choice = st.selectbox("Model", list(models.keys()))
    st.markdown("---")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age    = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["M", "F"])
        employment_status = st.selectbox(
            "Employment Status", ["Employed", "Self-Employed", "Unemployed"]
        )
        region = st.selectbox("Region", ["North", "South", "East", "West"])

    with col2:
        st.markdown("**Financials**")
        income           = st.number_input("Annual Income (₹)", 10000, 200000, 55000, step=1000)
        credit_limit     = st.number_input("Credit Limit (₹)",  1000, 100000, 20000, step=500)
        credit_used      = st.number_input("Credit Used (₹)",   0,    100000, 8000,  step=500)
        loan_outstanding = st.number_input("Loan Outstanding (₹)", 0, 100000, 5000, step=500)

    with col3:
        st.markdown("**Account & Transactions**")
        missed_payments    = st.slider("Missed Payments",     0, 20, 1)
        total_payments     = st.slider("Total Payments",      1, 200, 36)
        num_products       = st.slider("Number of Products",  1, 10, 3)
        account_age_months = st.slider("Account Age (months)", 1, 200, 48)
        transaction_amount = st.number_input("Last Transaction (₹)", 100, 50000, 1500, step=100)
        avg_monthly_spend  = st.number_input("Avg Monthly Spend (₹)", 100, 50000, 1400, step=100)
        num_late_fees      = st.slider("Late Fees", 0, 20, 2)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk")

    if predict_btn:
        # ── Encode categoricals the same way as training ──
        gender_enc     = 1 if gender == "M" else 0
        emp_map        = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
        region_map     = {"East": 0, "North": 1, "South": 2, "West": 3}
        emp_enc        = emp_map[employment_status]
        region_enc     = region_map[region]

        utilisation_rate    = credit_used / (credit_limit + 1)
        payment_failure_rate = missed_payments / (total_payments + 1)
        spend_deviation      = (transaction_amount - avg_monthly_spend) / (avg_monthly_spend + 1)

        input_data = np.array([[
            age, gender_enc, income, credit_limit, credit_used,
            missed_payments, total_payments, num_products,
            account_age_months, transaction_amount, avg_monthly_spend,
            num_late_fees, loan_outstanding, emp_enc, region_enc,
            utilisation_rate, payment_failure_rate, spend_deviation
        ]])

        # ── Scale & predict ──
        try:
            input_scaled = scaler.transform(input_data)
            model    = models[model_choice]
            pred     = model.predict(input_scaled)[0]
            proba    = model.predict_proba(input_scaled)[0]
            risk_pct = proba[1] * 100

            st.markdown("---")

            # Result banner
            if pred == 1:
                st.markdown(f"""
                <div style="background:#fef2f2; border:1.5px solid #fca5a5; border-radius:12px;
                            padding:1.5rem 2rem; margin-bottom:1rem;">
                    <div style="font-size:22px; font-weight:700; color:#b91c1c;">
                        🚨 HIGH RISK CUSTOMER
                    </div>
                    <div style="font-size:14px; color:#6b7280; margin-top:4px;">
                        Model: {model_choice} &nbsp;|&nbsp; Confidence: {risk_pct:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:12px;
                            padding:1.5rem 2rem; margin-bottom:1rem;">
                    <div style="font-size:22px; font-weight:700; color:#15803d;">
                        ✅ LOW RISK CUSTOMER
                    </div>
                    <div style="font-size:14px; color:#6b7280; margin-top:4px;">
                        Model: {model_choice} &nbsp;|&nbsp; Confidence: {100-risk_pct:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("**Risk Probability Breakdown**")
                fig, ax = plt.subplots(figsize=(5, 2.5))
                ax.barh(["Low Risk", "High Risk"],
                        [proba[0]*100, proba[1]*100],
                        color=["#10b981", "#ef4444"], edgecolor="white")
                for i, v in enumerate([proba[0]*100, proba[1]*100]):
                    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold")
                ax.set_xlim(0, 110)
                ax.set_xlabel("Probability (%)")
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_facecolor("#f8f9fb")
                fig.patch.set_facecolor("#f8f9fb")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Key risk factors summary
            with col_p2:
                st.markdown("**Key Risk Factors Detected**")
                flags = []
                if utilisation_rate > 0.9:
                    flags.append("🔴 Credit utilisation > 90%")
                if missed_payments >= 4:
                    flags.append("🔴 4+ missed payments")
                if income < 30000:
                    flags.append("🟡 Low income (< ₹30,000)")
                if employment_status == "Unemployed":
                    flags.append("🟡 Unemployed")
                if loan_outstanding > 15000:
                    flags.append("🟡 High outstanding loan")
                if payment_failure_rate > 0.2:
                    flags.append("🟠 High payment failure rate")
                if not flags:
                    flags.append("🟢 No major risk flags detected")
                for f in flags:
                    st.markdown(f)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure the scaler was trained on the same features. "
                    "Re-run `python src/train_model.py` if needed.")



       
