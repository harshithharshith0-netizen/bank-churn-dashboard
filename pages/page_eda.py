"""Exploratory Data Analysis Page."""
import streamlit as st
import pandas as pd
from utils.charts import (age_distribution, balance_distribution, correlation_heatmap,
                           geo_gender_heatmap, tenure_churn_line, product_age_heatmap,
                           churn_by_category)


def render(state: dict):
    df = state["df_eda"]   # enriched raw df with AgeGroup, BalanceBucket etc.

    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("<p style='color:#a0a8c0;'>Deep-dive into patterns, distributions, and churn drivers.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔢 Distributions",
        "🌍 Geography & Demographics",
        "📦 Products & Engagement",
        "🔗 Correlations",
    ])

    # ─────────────────────────── TAB 1 ───────────────────────────────────────
    with tab1:
        st.markdown("### Age & Balance Distributions")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(age_distribution(df), use_container_width=True)
            st.caption("Older customers (45–60) show noticeably higher churn density.")
        with col2:
            st.pyplot(balance_distribution(df), use_container_width=True)
            st.caption("Churned customers cluster around mid-high balances, not zero.")

        st.markdown("---")
        st.markdown("### Credit Score Band Analysis")
        cs_churn = df.groupby("CreditBand", observed=True)["Exited"].mean().mul(100)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.bar_chart(cs_churn, color="#4F8BF9")
        with col2:
            cs_df = df.groupby("CreditBand", observed=True)["Exited"].agg(["mean","count"])
            cs_df.columns = ["Churn Rate", "Count"]
            cs_df["Churn Rate"] = cs_df["Churn Rate"].map("{:.1%}".format)
            st.dataframe(cs_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Tenure Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(tenure_churn_line(df), use_container_width=True)
        with col2:
            tenure_df = df.groupby("Tenure")["Exited"].agg(["mean","count"]).reset_index()
            tenure_df.columns = ["Tenure (yrs)", "Churn Rate", "Customers"]
            tenure_df["Churn Rate"] = tenure_df["Churn Rate"].map("{:.1%}".format)
            st.dataframe(tenure_df, use_container_width=True, height=320)

    # ─────────────────────────── TAB 2 ───────────────────────────────────────
    with tab2:
        st.markdown("### Geography × Gender Cross-Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(geo_gender_heatmap(df), use_container_width=True)
        with col2:
            st.pyplot(churn_by_category(df, "Geography", "Churn Rate by Country"),
                      use_container_width=True)

        st.markdown("---")
        st.markdown("### Demographics Deep-Dive")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Churn by Gender**")
            g = df.groupby("Gender")["Exited"].agg(["mean","sum","count"])
            g.columns = ["Rate", "Churned", "Total"]
            g["Rate"] = g["Rate"].map("{:.1%}".format)
            st.dataframe(g, use_container_width=True)
        with col2:
            st.markdown("**Churn by Age Group**")
            a = df.groupby("AgeGroup", observed=True)["Exited"].agg(["mean","sum","count"])
            a.columns = ["Rate", "Churned", "Total"]
            a["Rate"] = a["Rate"].map("{:.1%}".format)
            st.dataframe(a, use_container_width=True)
        with col3:
            st.markdown("**Churn by Balance Bucket**")
            b = df.groupby("BalanceBucket", observed=True)["Exited"].agg(["mean","sum","count"])
            b.columns = ["Rate", "Churned", "Total"]
            b["Rate"] = b["Rate"].map("{:.1%}".format)
            st.dataframe(b, use_container_width=True)

    # ─────────────────────────── TAB 3 ───────────────────────────────────────
    with tab3:
        st.markdown("### Products × Engagement Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(churn_by_category(df, "NumOfProducts",
                      "Churn Rate by # Products"), use_container_width=True)
            st.caption("⚠️ Products = 3 or 4 is an extreme churn signal (>80%)")
        with col2:
            st.pyplot(churn_by_category(df, "IsActiveMember",
                      "Churn: Active vs Inactive"), use_container_width=True)

        st.markdown("---")
        st.markdown("### Age Group × Products — Churn Rate Heatmap")
        st.pyplot(product_age_heatmap(df), use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Churn by Has Credit Card**")
            hcc = df.groupby("HasCrCard")["Exited"].agg(["mean","count"])
            hcc.index = hcc.index.map({0: "No Card", 1: "Has Card"})
            hcc.columns = ["Churn Rate", "Count"]
            hcc["Churn Rate"] = hcc["Churn Rate"].map("{:.1%}".format)
            st.dataframe(hcc, use_container_width=True)
        with col2:
            st.markdown("**Combined: Active × Products**")
            pivot = df.groupby(["IsActiveMember","NumOfProducts"])["Exited"].mean().mul(100).unstack()
            pivot.index = pivot.index.map({0: "Inactive", 1: "Active"})
            st.dataframe(pivot.style.format("{:.1f}%").background_gradient(cmap="RdYlGn_r"),
                         use_container_width=True)

    # ─────────────────────────── TAB 4 ───────────────────────────────────────
    with tab4:
        st.markdown("### Correlation Analysis")
        st.pyplot(correlation_heatmap(df), use_container_width=True)

        st.markdown("---")
        st.markdown("### Correlation with Target (Exited)")
        num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
                    "HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
        corr_target = df[num_cols].corr()["Exited"].drop("Exited").sort_values(key=abs, ascending=False)
        corr_df = corr_target.reset_index()
        corr_df.columns = ["Feature", "Correlation with Exited"]
        corr_df["Correlation with Exited"] = corr_df["Correlation with Exited"].round(4)
        corr_df["Direction"] = corr_df["Correlation with Exited"].apply(
            lambda x: "🔺 Positive" if x > 0 else "🔻 Negative"
        )
        st.dataframe(corr_df, use_container_width=True, height=340)
