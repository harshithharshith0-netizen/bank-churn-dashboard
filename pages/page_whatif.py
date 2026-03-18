"""What-If Scenario Simulator Page."""
import streamlit as st
import numpy as np
import pandas as pd
from utils.model_trainer import predict_single, risk_tier, risk_color
from utils.charts import scenario_comparison


def _predict(trained, scaler, feat, **kwargs):
    defaults = dict(credit_score=650, geography="France", gender="Male",
                    age=38, tenure=5, balance=75000, num_products=1,
                    has_cc=1, is_active=1, salary=60000)
    defaults.update(kwargs)
    cs  = defaults["credit_score"]
    geo = defaults["geography"]
    gen = defaults["gender"]
    customer = {
        col: {
            "CreditScore":            cs,
            "Age":                    defaults["age"],
            "Tenure":                 defaults["tenure"],
            "Balance":                defaults["balance"],
            "NumOfProducts":          defaults["num_products"],
            "HasCrCard":              defaults["has_cc"],
            "IsActiveMember":         defaults["is_active"],
            "EstimatedSalary":        defaults["salary"],
            "Geography_France":       1 if geo=="France"  else 0,
            "Geography_Germany":      1 if geo=="Germany" else 0,
            "Geography_Spain":        1 if geo=="Spain"   else 0,
            "Gender_Female":          1 if gen=="Female" else 0,
            "Gender_Male":           1 if gen=="Male"   else 0,
            "Balance_Salary_Ratio":   defaults["balance"] / (defaults["salary"]+1),
            "Product_Density":        defaults["num_products"]/(defaults["tenure"]+1),
            "Engagement_Product":     defaults["is_active"]*defaults["num_products"],
            "Age_Tenure_Interaction": defaults["age"]*defaults["tenure"],
            "Zero_Balance":           1 if defaults["balance"]==0 else 0,
            "Is_Senior":              1 if defaults["age"]>50     else 0,
            "High_Balance":           1 if defaults["balance"]>97000 else 0,
            "CreditScore_Band":       (1 if cs<450 else 2 if cs<550 else 3 if cs<650 else 4 if cs<750 else 5),
        }.get(col, 0)
        for col in feat
    }
    return predict_single(trained["Gradient Boosting"], scaler, feat, customer)


def render(state: dict):
    trained = state["trained"]
    scaler  = state["scaler"]
    feat    = state["feat"]

    st.markdown("# 🔬 What-If Scenario Simulator")
    st.markdown("<p style='color:#a0a8c0;'>Adjust parameters and see how churn risk changes in real-time.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🎛️ Parameter Explorer", "📊 Scenario Comparison", "📈 Sensitivity Analysis"])

    # ── TAB 1: Live parameter explorer ───────────────────────────────────────
    with tab1:
        st.markdown("### Adjust Parameters — Churn Risk Updates Instantly")
        col1, col2 = st.columns(2)

        with col1:
            age         = st.slider("Age",                   18, 92, 38, key="wi_age")
            num_products= st.slider("Number of Products",    1,  4,  1,  key="wi_np")
            tenure      = st.slider("Tenure (years)",        0,  10, 5,  key="wi_ten")
            credit_score= st.slider("Credit Score",          350,850,650, key="wi_cs")
        with col2:
            balance     = st.slider("Account Balance (€k)",  0,  250, 75, key="wi_bal")
            salary      = st.slider("Salary (€k)",           10, 200, 60, key="wi_sal")
            geography   = st.selectbox("Geography",  ["France","Germany","Spain"], key="wi_geo")
            is_active   = st.toggle("Active Member", value=True, key="wi_act")

        prob = _predict(trained, scaler, feat,
                        age=age, num_products=num_products, tenure=tenure,
                        credit_score=credit_score, balance=balance*1000,
                        salary=salary*1000, geography=geography,
                        is_active=int(is_active))

        tier  = risk_tier(prob)
        color = risk_color(tier)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("Churn Probability", f"{prob:.2%}")
        with c2:
            st.metric("Risk Tier", f"{'🔴' if tier=='HIGH' else '🟡' if tier=='MEDIUM' else '🟢'} {tier}")
        with c3:
            retention_action = {
                "HIGH":   "🚨 Immediate Call",
                "MEDIUM": "📧 Email Campaign",
                "LOW":    "✅ Monitor Only"
            }[tier]
            st.metric("Suggested Action", retention_action)

        # Progress bar
        st.markdown(f"""
        <div style='margin-top:10px;'>
            <div style='display:flex; justify-content:space-between; color:#a0a8c0; font-size:12px;'>
                <span>Low Risk</span><span>Medium Risk</span><span>High Risk</span>
            </div>
            <div style='background:#1A1F2E; border-radius:8px; height:16px; overflow:hidden; border:1px solid #2d3147;'>
                <div style='background:linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c);
                    width:{prob*100:.1f}%; height:100%; border-radius:8px; transition:width 0.3s;'></div>
            </div>
            <div style='text-align:right; color:{color}; font-weight:700; font-size:16px; margin-top:4px;'>{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Preset scenario comparison ────────────────────────────────────
    with tab2:
        st.markdown("### Pre-Built Scenario Comparison")
        scenarios_def = [
            {"label": "Young Active (28, Active, 2 Products)",      "age": 28, "num_products": 2, "is_active": 1},
            {"label": "Senior Inactive (58, Inactive, 1 Product)",  "age": 58, "num_products": 1, "is_active": 0},
            {"label": "Germany Inactive (42, DE, Inactive)",        "age": 42, "geography": "Germany", "is_active": 0},
            {"label": "Germany Inactive + 3 Products",              "age": 45, "geography": "Germany", "is_active": 0, "num_products": 3},
            {"label": "Zero Balance + Inactive",                    "balance": 0, "is_active": 0},
            {"label": "Highly Engaged (35, Active, 2 Products)",    "age": 35, "num_products": 2, "is_active": 1},
            {"label": "Premium Active (4 products, active)",        "num_products": 4, "is_active": 1},
            {"label": "Long Tenure Active (10 yrs, active)",        "tenure": 10, "is_active": 1},
        ]

        scenarios_with_probs = []
        for s in scenarios_def:
            kw = {k: v for k, v in s.items() if k != "label"}
            p  = _predict(trained, scaler, feat, **kw)
            scenarios_with_probs.append({"label": s["label"], "prob": p})

        st.pyplot(scenario_comparison(scenarios_with_probs), use_container_width=True)

        # Table view
        tbl = pd.DataFrame([
            {"Scenario": s["label"],
             "Churn Probability": f"{s['prob']:.2%}",
             "Risk Tier": ("🔴 HIGH" if s["prob"]>=0.7 else "🟡 MEDIUM" if s["prob"]>=0.4 else "🟢 LOW")}
            for s in scenarios_with_probs
        ])
        st.dataframe(tbl, use_container_width=True, height=310)

    # ── TAB 3: Sensitivity analysis ───────────────────────────────────────────
    with tab3:
        st.markdown("### Feature Sensitivity — How Each Feature Affects Churn Probability")

        base_age     = st.slider("Base Age",          18, 80, 40, key="sa_age")
        base_products= st.slider("Base Products",      1,  4,  1, key="sa_np")
        base_active  = st.toggle("Active Member",  value=True, key="sa_act")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0F1117")
        BG2 = "#1A1F2E"

        # Sensitivity: Age
        ages  = list(range(18, 85, 2))
        probs_age = [_predict(trained, scaler, feat, age=a,
                              num_products=base_products, is_active=int(base_active)) for a in ages]
        axes[0].plot(ages, [p*100 for p in probs_age], color="#4F8BF9", lw=2.5, marker="o", ms=3)
        axes[0].axvline(base_age, color="white", ls="--", alpha=0.6)
        axes[0].axhline(40, color="#f39c12", ls="--", lw=1.2)
        axes[0].axhline(70, color="#e74c3c", ls="--", lw=1.2)
        axes[0].set_xlabel("Age", color="white"); axes[0].set_ylabel("Churn %", color="white")
        axes[0].set_title("Age Sensitivity", color="white", fontweight="bold")
        axes[0].set_facecolor(BG2); axes[0].tick_params(colors="white")
        for sp in axes[0].spines.values(): sp.set_edgecolor("#2d3147")

        # Sensitivity: Balance
        bals  = list(range(0, 260_000, 10_000))
        probs_bal = [_predict(trained, scaler, feat, balance=b,
                              age=base_age, num_products=base_products,
                              is_active=int(base_active)) for b in bals]
        axes[1].plot([b/1000 for b in bals], [p*100 for p in probs_bal],
                     color="#2ecc71", lw=2.5, marker="o", ms=3)
        axes[1].axhline(40, color="#f39c12", ls="--", lw=1.2)
        axes[1].axhline(70, color="#e74c3c", ls="--", lw=1.2)
        axes[1].set_xlabel("Balance (€k)", color="white"); axes[1].set_ylabel("Churn %", color="white")
        axes[1].set_title("Balance Sensitivity", color="white", fontweight="bold")
        axes[1].set_facecolor(BG2); axes[1].tick_params(colors="white")
        for sp in axes[1].spines.values(): sp.set_edgecolor("#2d3147")

        # Sensitivity: Credit Score
        css   = list(range(350, 855, 20))
        probs_cs = [_predict(trained, scaler, feat, credit_score=c,
                             age=base_age, num_products=base_products,
                             is_active=int(base_active)) for c in css]
        axes[2].plot(css, [p*100 for p in probs_cs], color="#e74c3c", lw=2.5, marker="o", ms=3)
        axes[2].axhline(40, color="#f39c12", ls="--", lw=1.2)
        axes[2].axhline(70, color="#e74c3c", ls="--", lw=1.2)
        axes[2].set_xlabel("Credit Score", color="white"); axes[2].set_ylabel("Churn %", color="white")
        axes[2].set_title("Credit Score Sensitivity", color="white", fontweight="bold")
        axes[2].set_facecolor(BG2); axes[2].tick_params(colors="white")
        for sp in axes[2].spines.values(): sp.set_edgecolor("#2d3147")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
