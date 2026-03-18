"""Home Overview Page."""
import streamlit as st
import pandas as pd
import numpy as np
from utils.charts import churn_donut, churn_by_category, churn_prob_histogram


def render(state: dict):
    df  = state["df_raw"]
    res = state["results"]

    st.markdown("# 🏦 Bank Customer Churn Intelligence")
    st.markdown(
        "<p style='color:#a0a8c0; font-size:15px;'>"
        "Predictive Modeling & Risk Scoring Dashboard — European Bank Dataset (2025)"
        "</p>", unsafe_allow_html=True
    )
    st.markdown("---")

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers",  f"{len(df):,}")
    c2.metric("🔴 Churned",          f"{df['Exited'].sum():,}",
              delta=f"{df['Exited'].mean():.1%}",
              delta_color="inverse")
    c3.metric("✅ Retained",         f"{(df['Exited']==0).sum():,}")
    c4.metric("💶 Avg Balance",      f"€{df['Balance'].mean():,.0f}")
    c5.metric("🏆 Best ROC-AUC",
              f"{max(v['roc_auc'] for v in res.values()):.4f}")

    st.markdown("---")

    # ── Row 1: Donut + Geography bar + Gender bar ──────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.4, 1.4])

    with col1:
        st.markdown("#### Churn Distribution")
        st.pyplot(churn_donut(df), use_container_width=True)

    with col2:
        st.markdown("#### Churn Rate by Geography")
        st.pyplot(churn_by_category(df, "Geography",
                  "Churn Rate by Geography"), use_container_width=True)

    with col3:
        st.markdown("#### Churn Rate by Products Held")
        st.pyplot(churn_by_category(df, "NumOfProducts",
                  "Churn Rate by # Products"), use_container_width=True)

    st.markdown("---")

    # ── Row 2: Model summary table + probability histogram ────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("#### Model Leaderboard")
        rows = []
        for name, m in res.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "CV-AUC": f"{m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.3f}",
            })
        df_tb = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        df_tb.index = df_tb.index + 1
        st.dataframe(df_tb, use_container_width=True, height=210)

    with col2:
        st.markdown("#### Predicted Probability Distribution (Best Model)")
        y_prob = res["Gradient Boosting"]["y_prob"]
        y_true = state["y_test"].values
        st.pyplot(churn_prob_histogram(y_prob, y_true), use_container_width=True)

    st.markdown("---")

    # ── Key Insights ─────────────────────────────────────────────────────────
    st.markdown("#### 🔍 Key Insights from the Data")
    insights = [
        ("🌍", "Germany", f"Highest churn geography at **{df[df['Geography']=='Germany']['Exited'].mean():.1%}**"),
        ("♀️", "Gender", f"Female customers churn at **{df[df['Gender']=='Female']['Exited'].mean():.1%}** vs {df[df['Gender']=='Male']['Exited'].mean():.1%} males"),
        ("📦", "Products", f"Customers with **3+ products** churn at >80% — a critical signal"),
        ("⚡", "Activity", f"Inactive members churn at **{df[df['IsActiveMember']==0]['Exited'].mean():.1%}** vs {df[df['IsActiveMember']==1]['Exited'].mean():.1%} active"),
        ("🎂", "Age",     f"Customers aged **50+** are the highest-risk age cohort"),
        ("💰", "Balance",  f"Zero-balance customers represent a distinct churn risk group"),
    ]
    c1, c2, c3 = st.columns(3)
    for i, (icon, title, msg) in enumerate(insights):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(
                f"""<div style='background:#1A1F2E; border-left:4px solid #4F8BF9;
                border-radius:8px; padding:12px 14px; margin-bottom:10px;'>
                <span style='font-size:20px'>{icon}</span>
                <b style='color:#4F8BF9;'> {title}</b><br>
                <span style='color:#d1d5db; font-size:13px;'>{msg}</span>
                </div>""",
                unsafe_allow_html=True
            )
