"""Batch Customer Risk Report Page."""
import streamlit as st
import pandas as pd
import numpy as np
from utils.model_trainer import risk_tier, risk_color
from utils.charts import risk_tier_pie, churn_prob_histogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(state: dict):
    trained = state["trained"]
    X_test  = state["X_test"]
    y_test  = state["y_test"]
    df_raw  = state["df_raw"]
    feat    = state["feat"]

    st.markdown("# 📋 Batch Customer Risk Report")
    st.markdown("<p style='color:#a0a8c0;'>Risk scoring for all 2,000 test-set customers.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Score entire test set ─────────────────────────────────────────────────
    model_sel = st.selectbox("Scoring Model", list(trained.keys()), index=3)
    model = trained[model_sel]
    probs = model.predict_proba(X_test)[:, 1]

    tiers      = [risk_tier(p) for p in probs]
    tier_counts = {"HIGH": tiers.count("HIGH"),
                   "MEDIUM": tiers.count("MEDIUM"),
                   "LOW": tiers.count("LOW")}

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High Risk",   tier_counts["HIGH"],
              help="Churn probability ≥ 70%")
    c2.metric("🟡 Medium Risk", tier_counts["MEDIUM"],
              help="Churn probability 40-70%")
    c3.metric("🟢 Low Risk",    tier_counts["LOW"],
              help="Churn probability < 40%")
    c4.metric("Expected Churners",
              f"{int(probs.sum()):,}",
              help="Sum of all churn probabilities")

    st.markdown("---")

    # ── Visualisations row ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Risk Tier Distribution")
        st.pyplot(risk_tier_pie(tier_counts), use_container_width=True)
    with col2:
        st.markdown("#### Probability Distribution")
        st.pyplot(churn_prob_histogram(probs, y_test.values), use_container_width=True)

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("### 🔎 Customer Risk Table")
    col1, col2, col3 = st.columns(3)
    filter_tier  = col1.multiselect("Filter by Risk Tier", ["HIGH", "MEDIUM", "LOW"],
                                    default=["HIGH", "MEDIUM", "LOW"])
    min_prob     = col2.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)
    sort_by      = col3.selectbox("Sort by", ["Churn Probability ↓", "Churn Probability ↑"])

    # Build display dataframe
    report_df = pd.DataFrame({
        "Churn Probability": probs,
        "Risk Tier":         tiers,
        "Actual Churn":      y_test.values,
    }).reset_index(drop=True)
    report_df["Churn Probability %"] = (report_df["Churn Probability"] * 100).round(2)
    report_df["Correct Prediction"]  = (
        (report_df["Churn Probability"] >= 0.5) == (report_df["Actual Churn"] == 1)
    )

    filtered = report_df[
        (report_df["Risk Tier"].isin(filter_tier)) &
        (report_df["Churn Probability"] >= min_prob)
    ]
    filtered = filtered.sort_values(
        "Churn Probability",
        ascending=(sort_by == "Churn Probability ↑")
    ).reset_index(drop=True)

    st.markdown(f"Showing **{len(filtered):,}** customers (from {len(report_df):,} total)")

    def highlight_tier(row):
        colors = {"HIGH": "background-color:#3d1515",
                  "MEDIUM": "background-color:#3d2e10",
                  "LOW": "background-color:#0f2d1a"}
        return [colors.get(row["Risk Tier"], "")] * len(row)

    st.dataframe(
        filtered[["Risk Tier", "Churn Probability %", "Actual Churn", "Correct Prediction"]]
        .head(500)
        .style.apply(highlight_tier, axis=1)
        .format({"Churn Probability %": "{:.2f}%"}),
        use_container_width=True,
        height=400
    )

    # ── Download ──────────────────────────────────────────────────────────────
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Risk Report (CSV)",
        data=csv,
        file_name="customer_risk_report.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Segment analysis ──────────────────────────────────────────────────────
    st.markdown("### 📊 High-Risk Customer Profile Summary")
    high_mask = np.array(tiers) == "HIGH"
    X_test_np = X_test.values if hasattr(X_test, "values") else X_test

    if high_mask.sum() > 0:
        hr_df = X_test[high_mask].copy()
        all_df = X_test.copy()

        metrics_compare = {}
        for col in ["Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "CreditScore"]:
            if col in hr_df.columns:
                metrics_compare[col] = {
                    "High Risk Mean": round(hr_df[col].mean(), 2),
                    "All Customers Mean": round(all_df[col].mean(), 2),
                    "Difference": round(hr_df[col].mean() - all_df[col].mean(), 2),
                }

        compare_df = pd.DataFrame(metrics_compare).T.reset_index()
        compare_df.columns = ["Feature", "High Risk Mean", "All Customers Mean", "Difference"]
        st.dataframe(compare_df.style.background_gradient(
            subset=["Difference"], cmap="RdYlGn_r"), use_container_width=True)
