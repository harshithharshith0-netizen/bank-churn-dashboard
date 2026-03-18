"""Individual Customer Risk Score Calculator Page."""
import streamlit as st
import numpy as np
from utils.model_trainer import predict_single, risk_tier, risk_color
from utils.charts import probability_gauge, feature_importance_plot
import pandas as pd


def _build_customer(credit_score, geography, gender, age, tenure,
                    balance, num_products, has_cc, is_active, salary,
                    feature_cols):
    """Convert UI inputs → feature dict aligned with model columns."""
    return {
        col: {
            "CreditScore":                credit_score,
            "Age":                        age,
            "Tenure":                     tenure,
            "Balance":                    balance,
            "NumOfProducts":              num_products,
            "HasCrCard":                  has_cc,
            "IsActiveMember":             is_active,
            "EstimatedSalary":            salary,
            "Geography_France":           1 if geography == "France"   else 0,
            "Geography_Germany":          1 if geography == "Germany"  else 0,
            "Geography_Spain":            1 if geography == "Spain"    else 0,
            "Gender_Female":              1 if gender == "Female" else 0,
            "Gender_Male":               1 if gender == "Male"   else 0,
            "Balance_Salary_Ratio":       balance / (salary + 1),
            "Product_Density":            num_products / (tenure + 1),
            "Engagement_Product":         is_active * num_products,
            "Age_Tenure_Interaction":     age * tenure,
            "Zero_Balance":               1 if balance == 0 else 0,
            "Is_Senior":                  1 if age > 50 else 0,
            "High_Balance":               1 if balance > 97000 else 0,
            "CreditScore_Band":           (
                1 if credit_score < 450 else
                2 if credit_score < 550 else
                3 if credit_score < 650 else
                4 if credit_score < 750 else 5
            ),
        }.get(col, 0)
        for col in feature_cols
    }


def render(state: dict):
    trained = state["trained"]
    scaler  = state["scaler"]
    feat    = state["feat"]

    st.markdown("# 🎯 Churn Risk Score Calculator")
    st.markdown("<p style='color:#a0a8c0;'>Enter customer details to get an instant churn probability and risk tier.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("customer_form"):
        st.markdown("### 👤 Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal Details**")
            geography   = st.selectbox("Geography",  ["France", "Germany", "Spain"])
            gender      = st.selectbox("Gender",     ["Male", "Female"])
            age         = st.slider("Age",           18, 92, 38)
            tenure      = st.slider("Tenure (yrs)", 0,  10,  5)

        with col2:
            st.markdown("**Financial Info**")
            credit_score = st.slider("Credit Score",          350, 850, 650)
            balance      = st.number_input("Account Balance (€)",   0.0, 300000.0, 75000.0, step=5000.0)
            salary       = st.number_input("Est. Annual Salary (€)", 0.0, 200000.0, 60000.0, step=5000.0)

        with col3:
            st.markdown("**Banking Behaviour**")
            num_products = st.selectbox("# Products",       [1, 2, 3, 4])
            has_cc       = st.radio("Has Credit Card",      [1, 0], format_func=lambda x: "Yes" if x else "No",
                                    horizontal=True)
            is_active    = st.radio("Active Member",        [1, 0], format_func=lambda x: "Yes" if x else "No",
                                    horizontal=True)
            model_sel    = st.selectbox("Prediction Model", list(trained.keys()))

        submitted = st.form_submit_button("🔍  Calculate Risk Score", use_container_width=True, type="primary")

    if submitted:
        customer = _build_customer(credit_score, geography, gender, age, tenure,
                                   balance, num_products, has_cc, is_active,
                                   salary, feat)

        use_scaled = (model_sel == "Logistic Regression")
        prob = predict_single(trained[model_sel], scaler, feat, customer, use_scaled)
        tier = risk_tier(prob)
        color = risk_color(tier)

        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Result")

        col_gauge, col_detail = st.columns([1, 1.5])

        with col_gauge:
            st.pyplot(probability_gauge(prob), use_container_width=True)

        with col_detail:
            # Risk badge
            badge_bg = {"HIGH": "#3d1515", "MEDIUM": "#3d2e10", "LOW": "#0f2d1a"}[tier]
            st.markdown(
                f"""<div style='background:{badge_bg}; border:2px solid {color};
                border-radius:12px; padding:18px 22px; margin-bottom:12px;'>
                <span style='font-size:28px; font-weight:800; color:{color};'>
                    {"🔴" if tier=="HIGH" else "🟡" if tier=="MEDIUM" else "🟢"} {tier} RISK
                </span><br>
                <span style='color:#d1d5db; font-size:15px;'>
                    Churn Probability: <b style='color:{color};'>{prob:.2%}</b>
                </span>
                </div>""",
                unsafe_allow_html=True
            )

            # All model predictions
            st.markdown("**Predictions from all models:**")
            rows = []
            for mname, model in trained.items():
                uscl = (mname == "Logistic Regression")
                p = predict_single(model, scaler, feat, customer, uscl)
                rows.append({"Model": mname,
                             "Churn Probability": f"{p:.2%}",
                             "Risk Tier": risk_tier(p)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=205)

        st.markdown("---")

        # ── Recommended Action ───────────────────────────────────────────────
        st.markdown("### 💡 Recommended Retention Action")
        if tier == "HIGH":
            st.error("""
**Immediate Intervention Required**
- 📞 Priority outreach within 24 hours
- 💳 Offer premium account upgrade / loyalty bonus
- 🎁 Personalised retention package (fee waiver, interest rate boost)
- 👤 Assign dedicated relationship manager
            """)
        elif tier == "MEDIUM":
            st.warning("""
**Proactive Engagement Recommended**
- 📧 Send personalised product recommendations
- 📱 Trigger in-app engagement campaign
- 📊 Review product fit — consider cross-sell / upsell
- 🔔 Schedule follow-up call in next 30 days
            """)
        else:
            st.success("""
**Low Priority — Maintenance Mode**
- ✅ Continue standard engagement programmes
- 📊 Monitor for sudden behavioural changes
- 🎂 Acknowledge loyalty milestones
            """)

        # ── Feature contributions ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Key Factors Driving This Score")
        col_a, col_b = st.columns(2)
        factors = {
            "Age":          f"{age} yrs {'(HIGH risk age group)' if age > 50 else '(lower risk age group)'}",
            "Num Products": f"{num_products} {'(⚠️ critical risk if >2)' if num_products > 2 else ''}",
            "Active Member": "No (increases churn risk)" if not is_active else "Yes (reduces churn risk)",
            "Geography":    f"{geography} {'(highest risk country)' if geography=='Germany' else ''}",
            "Balance":      f"€{balance:,.0f} {'(zero balance flag)' if balance==0 else ''}",
            "Tenure":       f"{tenure} yrs",
        }
        for i, (k, v) in enumerate(factors.items()):
            with (col_a if i % 2 == 0 else col_b):
                st.markdown(
                    f"<div style='background:#1A1F2E; border-radius:8px; padding:8px 12px; "
                    f"margin-bottom:8px; font-size:13px;'>"
                    f"<b style='color:#4F8BF9;'>{k}:</b> <span style='color:#d1d5db;'>{v}</span></div>",
                    unsafe_allow_html=True
                )
