"""Model Performance Page."""
import streamlit as st
import pandas as pd
import numpy as np
from utils.charts import (model_comparison_bar, roc_curves, confusion_matrix_plot,
                           feature_importance_plot, cv_scores_plot, precision_recall_plot)
from utils.model_trainer import get_feature_importance


def render(state: dict):
    res     = state["results"]
    trained = state["trained"]
    feat    = state["feat"]
    y_test  = state["y_test"]

    st.markdown("# 🤖 Model Performance Analysis")
    st.markdown("<p style='color:#a0a8c0;'>Comparison of 5 classifiers trained on real bank data.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Summary table ────────────────────────────────────────────────────────
    rows = []
    for name, m in res.items():
        rows.append({
            "Model": name,
            "Accuracy":  round(m["accuracy"],  4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"],    4),
            "F1-Score":  round(m["f1"],        4),
            "ROC-AUC":   round(m["roc_auc"],   4),
            "CV-AUC":    f"{m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.3f}",
        })
    df_tb = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    df_tb.index += 1
    st.dataframe(
        df_tb.style
             .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"], color="#1e3a5f")
             .format({"Accuracy":"{:.4f}","Precision":"{:.4f}",
                      "Recall":"{:.4f}","F1-Score":"{:.4f}","ROC-AUC":"{:.4f}"}),
        use_container_width=True, height=220
    )
    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Metrics", "📈 ROC & PR Curves", "🔲 Confusion Matrices", "🌲 Feature Importance"
    ])

    with tab1:
        st.markdown("### Grouped Metric Comparison")
        st.pyplot(model_comparison_bar(res), use_container_width=True)
        st.markdown("---")
        st.markdown("### Cross-Validation Stability (5-Fold AUC)")
        st.pyplot(cv_scores_plot(res), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ROC Curves")
            st.pyplot(roc_curves(res, y_test), use_container_width=True)
        with col2:
            st.markdown("### Precision-Recall Curves")
            st.pyplot(precision_recall_plot(res), use_container_width=True)

    with tab3:
        st.markdown("### Confusion Matrices")
        cols = st.columns(3)
        for i, (name, m) in enumerate(res.items()):
            with cols[i % 3]:
                st.pyplot(confusion_matrix_plot(m["conf_matrix"], name),
                          use_container_width=True)

    with tab4:
        st.markdown("### Feature Importance")
        model_sel = st.selectbox(
            "Select model",
            [n for n in trained.keys() if n not in ["Logistic Regression", "Voting Ensemble"]]
        )
        fi = get_feature_importance(trained[model_sel], feat)
        if not fi.empty:
            top_n = st.slider("Show top N features", 5, len(fi), 15)
            st.pyplot(feature_importance_plot(fi, model_sel, top_n=top_n),
                      use_container_width=True)

            st.markdown("#### Feature Importance Table")
            fi_df = fi.reset_index()
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Importance %"] = fi_df["Importance"].div(fi_df["Importance"].sum()).mul(100).round(2)
            st.dataframe(fi_df.head(top_n), use_container_width=True)

    st.markdown("---")
    # ── Model insights callouts ───────────────────────────────────────────────
    best = max(res, key=lambda n: res[n]["roc_auc"])
    st.markdown("### 📌 Model Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**Best Overall Model:** {best}\n\nROC-AUC: `{res[best]['roc_auc']:.4f}`\nF1-Score: `{res[best]['f1']:.4f}`")
    with c2:
        best_rec = max(res, key=lambda n: res[n]["recall"])
        st.warning(f"**Best Recall (catches most churners):** {best_rec}\n\nRecall: `{res[best_rec]['recall']:.4f}`")
    with c3:
        best_prec = max(res, key=lambda n: res[n]["precision"])
        st.success(f"**Best Precision (fewer false alarms):** {best_prec}\n\nPrecision: `{res[best_prec]['precision']:.4f}`")
