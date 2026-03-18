"""Project Summary & Methodology Page."""
import streamlit as st
import pandas as pd


def render(state: dict):
    res  = state["results"]
    feat = state["feat"]

    st.markdown("# 📖 Project Summary & Methodology")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Objectives", "⚙️ Methodology", "📊 Results", "💡 Recommendations"
    ])

    # ── TAB 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Primary Objectives")
            for item in [
                "Predict customer churn with high accuracy (ROC-AUC > 0.85)",
                "Generate individual churn probability scores (0–1)",
                "Identify key churn drivers via feature importance",
                "Assign actionable risk tiers: LOW / MEDIUM / HIGH",
            ]:
                st.markdown(f"✅ {item}")

        with col2:
            st.markdown("### Secondary Objectives")
            for item in [
                "Reduce false positives (Precision focus)",
                "Improve model interpretability (tree-based + SHAP-ready)",
                "Enable scenario-based what-if analysis",
                "Provide downloadable batch risk reports",
                "Build an interactive Streamlit dashboard",
            ]:
                st.markdown(f"🔵 {item}")

        st.markdown("---")
        st.markdown("### Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",      "10,000")
        c2.metric("Features",  f"{len(feat)}")
        c3.metric("Churn Rate", f"{state['df_raw']['Exited'].mean():.1%}")
        c4.metric("Geography", "France / Germany / Spain")

        st.markdown("""
        **Target Variable:** `Exited`
        - `1` → Customer churned
        - `0` → Customer retained

        **Original Features:** CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        """)

    # ── TAB 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Data Preprocessing")
        preprocessing_steps = {
            "Step": ["1", "2", "3", "4", "5"],
            "Action": [
                "Drop non-informative columns",
                "One-hot encode categoricals",
                "Scale numerical features",
                "Feature engineering",
                "Stratified 80/20 train-test split",
            ],
            "Detail": [
                "Removed: Year, CustomerId, Surname",
                "Geography (France/Germany/Spain), Gender (Male/Female)",
                "StandardScaler applied for Logistic Regression",
                "8 new derived features created (see below)",
                "Stratify=Exited to preserve 20.4% churn class ratio",
            ]
        }
        st.dataframe(pd.DataFrame(preprocessing_steps), use_container_width=True)

        st.markdown("---")
        st.markdown("### Engineered Features")
        eng_features = {
            "Feature": [
                "Balance_Salary_Ratio", "Product_Density",
                "Engagement_Product", "Age_Tenure_Interaction",
                "Zero_Balance", "Is_Senior", "High_Balance", "CreditScore_Band"
            ],
            "Formula / Definition": [
                "Balance / (EstimatedSalary + 1)",
                "NumOfProducts / (Tenure + 1)",
                "IsActiveMember × NumOfProducts",
                "Age × Tenure",
                "1 if Balance == 0 else 0",
                "1 if Age > 50 else 0",
                "1 if Balance > median(Balance) else 0",
                "Ordinal bins: 1–5 (Poor → Excellent)",
            ],
            "Rationale": [
                "Relative financial dependency",
                "Product engagement relative to relationship length",
                "Combined engagement signal",
                "Relationship depth indicator",
                "Potential disengagement signal",
                "Higher churn risk in older demographic",
                "High balance + churn = revenue risk",
                "Non-linear credit quality encoding",
            ]
        }
        st.dataframe(pd.DataFrame(eng_features), use_container_width=True)

        st.markdown("---")
        st.markdown("### Models Used")
        models_info = {
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest",
                      "Gradient Boosting", "Voting Ensemble"],
            "Type": ["Baseline (linear)", "Tree-based", "Ensemble (bagging)",
                     "Ensemble (boosting)", "Soft-voting ensemble"],
            "Key Hyperparameters": [
                "max_iter=1000, C=1.0",
                "max_depth=6, min_samples_leaf=20",
                "n_estimators=200, max_depth=10, min_samples_leaf=10",
                "n_estimators=200, learning_rate=0.05, max_depth=4",
                "LR + RF + GB with soft voting",
            ],
            "Purpose": [
                "Interpretability benchmark",
                "Simple tree structure, explainable rules",
                "Reduces variance via bagging",
                "Best predictive performance",
                "Combines strength of all three",
            ]
        }
        st.dataframe(pd.DataFrame(models_info), use_container_width=True)

        st.markdown("---")
        st.markdown("### Evaluation Metrics")
        metrics_info = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV-AUC (5-fold)"],
            "Purpose": [
                "Overall correctness",
                "Controls false positive churn alarms",
                "Captures actual churners (sensitivity)",
                "Harmonic mean — balances Precision & Recall",
                "Model discrimination capability",
                "Generalisation stability estimate",
            ]
        }
        st.dataframe(pd.DataFrame(metrics_info), use_container_width=True)

    # ── TAB 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Final Model Results")
        rows = []
        for name, m in res.items():
            rows.append({
                "Model": name,
                "Accuracy":   f"{m['accuracy']:.4f}",
                "Precision":  f"{m['precision']:.4f}",
                "Recall":     f"{m['recall']:.4f}",
                "F1-Score":   f"{m['f1']:.4f}",
                "ROC-AUC":    f"{m['roc_auc']:.4f}",
                "CV-AUC":     f"{m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.3f}",
            })
        df_res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        df_res.index += 1
        st.dataframe(df_res, use_container_width=True)

        best = max(res, key=lambda n: res[n]["roc_auc"])
        st.success(f"""
        **🏆 Best Model: {best}**

        - ROC-AUC: `{res[best]['roc_auc']:.4f}`
        - F1-Score: `{res[best]['f1']:.4f}`
        - CV-AUC:   `{res[best]['cv_auc_mean']:.4f} ± {res[best]['cv_auc_std']:.3f}`

        Top churn drivers: **Age**, **NumOfProducts**, **IsActiveMember**, **Geography_Germany**, **Balance**
        """)

    # ── TAB 4 ──────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 💼 Business Recommendations")

        recs = [
            ("🔴 High-Risk Segment (70%+ probability)",
             "Assign dedicated relationship managers. Launch immediate personalised retention offer. "
             "Consider fee waivers, interest rate adjustments, or premium tier upgrades.",
             "#3d1515", "#e74c3c"),
            ("🌍 Germany Market Priority",
             "Germany has 32.4% churn vs 16% France/Spain. Investigate service quality, "
             "competitor offerings, and local product gaps. Consider Germany-specific retention campaigns.",
             "#2d1f0f", "#f39c12"),
            ("📦 3–4 Products = Danger Zone",
             "Counter-intuitively, customers with 3-4 products churn nearly 100%. "
             "Investigate product complexity and cross-sell alignment. Simplify product bundles.",
             "#1a1530", "#9b59b6"),
            ("⚡ Inactive Member Activation",
             "Inactive members churn at 2× the rate of active members. Launch digital engagement "
             "campaigns, in-app nudges, and financial health check-ins within 90 days of inactivity.",
             "#0f2d1a", "#2ecc71"),
            ("🎂 Senior Customer Programme (50+)",
             "Age is the #1 churn predictor. Create a dedicated senior banking programme with "
             "personalised support, simplified digital interfaces, and exclusive senior benefits.",
             "#141824", "#4F8BF9"),
            ("📊 Predictive Score Integration",
             "Integrate the Gradient Boosting model's probability scores into CRM workflows. "
             "Auto-flag HIGH-risk customers weekly for proactive outreach. "
             "Target precision: reduce churn rate from 20.4% → <15% within 12 months.",
             "#0f1f2d", "#4F8BF9"),
        ]

        for title, desc, bg, border in recs:
            st.markdown(
                f"""<div style='background:{bg}; border-left:4px solid {border};
                border-radius:10px; padding:16px 18px; margin-bottom:12px;'>
                <b style='color:{border}; font-size:15px;'>{title}</b><br>
                <span style='color:#d1d5db; font-size:13px; line-height:1.7;'>{desc}</span>
                </div>""",
                unsafe_allow_html=True
            )
