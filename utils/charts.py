"""
Centralised chart-building functions.
All functions return a matplotlib Figure so Streamlit can render them.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

# ── Global style ──────────────────────────────────────────────────────────────
BG      = "#0F1117"
CARD_BG = "#1A1F2E"
TEXT    = "#FFFFFF"
ACCENT  = "#4F8BF9"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
ORANGE  = "#f39c12"
PURPLE  = "#9b59b6"
COLORS5 = [ACCENT, GREEN, RED, ORANGE, PURPLE]

def _style(fig, ax_list=None):
    """Apply dark theme to a figure."""
    fig.patch.set_facecolor(BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3147")
    return fig


# ── EDA Charts ────────────────────────────────────────────────────────────────

def churn_donut(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    vals   = df["Exited"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    colors = [GREEN, RED]
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
        textprops=dict(color=TEXT, fontsize=11)
    )
    for at in autotexts:
        at.set_color(TEXT); at.set_fontsize(11)
    ax.set_title("Overall Churn Distribution", color=TEXT, fontsize=13, fontweight="bold", pad=15)
    return _style(fig)


def churn_by_category(df: pd.DataFrame, col: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    grp = df.groupby(col)["Exited"].mean().mul(100).sort_values(ascending=False)
    bars = ax.bar(grp.index.astype(str), grp.values,
                  color=[RED if v > 20 else ACCENT for v in grp.values],
                  edgecolor=BG, linewidth=0.8, zorder=3)
    ax.set_ylabel("Churn Rate (%)", color=TEXT)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#2d3147", zorder=0)
    ax.set_ylim(0, grp.max() * 1.25)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", color=TEXT, fontsize=9)
    return _style(fig)


def age_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[df["Exited"] == 0]["Age"], bins=35, alpha=0.7,
            color=GREEN, label="Retained", density=True)
    ax.hist(df[df["Exited"] == 1]["Age"], bins=35, alpha=0.7,
            color=RED, label="Churned", density=True)
    ax.set_xlabel("Age"); ax.set_ylabel("Density")
    ax.set_title("Age Distribution by Churn Status", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT)
    return _style(fig)


def balance_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    retained = df[df["Exited"] == 0]["Balance"]
    churned  = df[df["Exited"] == 1]["Balance"]
    ax.hist(retained, bins=40, alpha=0.7, color=GREEN, label="Retained", density=True)
    ax.hist(churned,  bins=40, alpha=0.7, color=RED,   label="Churned",  density=True)
    ax.set_xlabel("Account Balance (€)"); ax.set_ylabel("Density")
    ax.set_title("Balance Distribution by Churn", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT)
    return _style(fig)


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, mask=mask,
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 8, "color": TEXT},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    return _style(fig)


def geo_gender_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.groupby(["Geography", "Gender"])["Exited"].mean().mul(100).unstack()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax, linewidths=1, linecolor=BG,
                annot_kws={"size": 12, "color": TEXT},
                cbar_kws={"label": "Churn Rate (%)", "shrink": 0.8})
    ax.set_title("Churn Rate (%) — Geography × Gender",
                 color=TEXT, fontsize=12, fontweight="bold")
    return _style(fig)


def tenure_churn_line(df: pd.DataFrame) -> plt.Figure:
    tenure_churn = df.groupby("Tenure")["Exited"].mean().mul(100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tenure_churn.index, tenure_churn.values,
            marker="o", color=ACCENT, linewidth=2.5, markersize=7, zorder=3)
    ax.fill_between(tenure_churn.index, tenure_churn.values, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Tenure (Years)"); ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Customer Tenure", color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(color="#2d3147", zorder=0)
    return _style(fig)


def product_age_heatmap(df: pd.DataFrame) -> plt.Figure:
    df2 = df.copy()
    df2["AgeGroup"] = pd.cut(df2["Age"],
                              bins=[17, 30, 40, 50, 60, 100],
                              labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    pivot = df2.groupby(["AgeGroup", "NumOfProducts"], observed=True)["Exited"].mean().mul(100).unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, linecolor=BG,
                annot_kws={"size": 10},
                cbar_kws={"label": "Churn Rate (%)"})
    ax.set_title("Churn Rate % — Age Group × Products", color=TEXT, fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Products"); ax.set_ylabel("Age Group")
    return _style(fig)


# ── Model Performance Charts ───────────────────────────────────────────────────

def model_comparison_bar(results: dict) -> plt.Figure:
    names    = list(results.keys())
    metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, l) in enumerate(zip(metrics, labels)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=l,
                      color=COLORS5[i], alpha=0.88, edgecolor=BG)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], color=TEXT, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison — All Metrics",
                 color=TEXT, fontsize=13, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
    ax.grid(axis="y", color="#2d3147", zorder=0)
    return _style(fig)


def roc_curves(results: dict, y_test) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, res), color in zip(results.items(), COLORS5):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2.5,
                label=f"{name}  AUC={res['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8, loc="lower right")
    ax.grid(color="#2d3147")
    return _style(fig)


def confusion_matrix_plot(cm: np.ndarray, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    group_names  = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = [f"{v:,}" for v in cm.flatten()]
    group_pct    = [f"{v:.1%}" for v in cm.flatten() / cm.sum()]
    annot_labels = [f"{n}\n{c}\n{p}" for n, c, p in
                    zip(group_names, group_counts, group_pct)]
    annot_arr    = np.array(annot_labels).reshape(2, 2)
    sns.heatmap(cm, annot=annot_arr, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
                linewidths=2, linecolor=BG,
                annot_kws={"size": 9})
    ax.set_title(f"Confusion Matrix\n{model_name}", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    return _style(fig)


def feature_importance_plot(feat_imp: pd.Series, model_name: str, top_n=15) -> plt.Figure:
    fi = feat_imp.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [RED if i >= len(fi) - 3 else ACCENT for i in range(len(fi))]
    ax.barh(fi.index, fi.values, color=colors, edgecolor=BG, height=0.7)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#2d3147")
    for i, (val, name) in enumerate(zip(fi.values, fi.index)):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", color=TEXT, fontsize=8)
    return _style(fig)


def cv_scores_plot(results: dict) -> plt.Figure:
    names  = list(results.keys())
    means  = [results[n]["cv_auc_mean"] for n in names]
    stds   = [results[n]["cv_auc_std"]  for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, means, color=COLORS5, yerr=stds, capsize=5,
                  edgecolor=BG, alpha=0.88, error_kw=dict(color=TEXT, lw=1.5))
    ax.set_ylim(0.7, 0.95); ax.set_ylabel("CV ROC-AUC")
    ax.set_title("5-Fold Cross-Validation AUC (Mean ± Std)",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", color="#2d3147")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{mean:.4f}", ha="center", color=TEXT, fontsize=9)
    return _style(fig)


def precision_recall_plot(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, res), color in zip(results.items(), COLORS5):
        ax.plot(res["rec_curve"], res["prec_curve"], color=color, lw=2,
                label=f"{name}  F1={res['f1']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
    ax.grid(color="#2d3147")
    return _style(fig)


# ── Risk / Prediction Charts ───────────────────────────────────────────────────

def probability_gauge(prob: float) -> plt.Figure:
    """Semi-circle gauge showing churn probability."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.3, 1.3)

    # Draw background arcs
    theta = np.linspace(np.pi, 0, 300)
    for i, (lo, hi, col) in enumerate([(0, 0.4, GREEN), (0.4, 0.7, ORANGE), (0.7, 1.0, RED)]):
        t_lo = np.pi * (1 - lo); t_hi = np.pi * (1 - hi)
        t    = np.linspace(t_lo, t_hi, 100)
        ax.plot(np.cos(t), np.sin(t), color=col, lw=18, alpha=0.3, solid_capstyle="round")

    # Filled arc up to prob
    t_fill = np.linspace(np.pi, np.pi * (1 - prob), 200)
    col = RED if prob >= 0.7 else (ORANGE if prob >= 0.4 else GREEN)
    ax.plot(np.cos(t_fill), np.sin(t_fill), color=col, lw=18, solid_capstyle="round")

    # Needle
    angle = np.pi * (1 - prob)
    ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.5))
    ax.plot(0, 0, "o", color=TEXT, ms=8, zorder=5)

    # Labels
    ax.text(-1.1, -0.2, "0%",   color=TEXT, ha="center", fontsize=9)
    ax.text(0,   -0.2, "50%",   color=TEXT, ha="center", fontsize=9)
    ax.text(1.1, -0.2, "100%",  color=TEXT, ha="center", fontsize=9)
    tier = "HIGH RISK" if prob >= 0.7 else ("MEDIUM RISK" if prob >= 0.4 else "LOW RISK")
    ax.text(0, 0.45, f"{prob:.1%}", color=TEXT, ha="center", fontsize=22, fontweight="bold")
    ax.text(0, 0.22, tier, color=col, ha="center", fontsize=13, fontweight="bold")
    ax.axis("off")
    return _style(fig)


def churn_prob_histogram(y_prob, y_true) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.7, color=GREEN, label="Retained", density=True)
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.7, color=RED,   label="Churned",  density=True)
    ax.axvline(0.40, color=ORANGE, ls="--", lw=1.8, label="Medium threshold (0.40)")
    ax.axvline(0.70, color=RED,    ls="--", lw=1.8, label="High threshold (0.70)")
    ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution by Actual Class",
                 color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=9)
    return _style(fig)


def risk_tier_pie(risk_counts: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = list(risk_counts.keys())
    vals   = list(risk_counts.values())
    colors = [GREEN if l == "LOW" else (ORANGE if l == "MEDIUM" else RED) for l in labels]
    wedges, texts, autos = ax.pie(vals, labels=labels, colors=colors,
                                   autopct="%1.1f%%", startangle=90,
                                   wedgeprops=dict(edgecolor=BG, linewidth=2),
                                   textprops=dict(color=TEXT, fontsize=11))
    for at in autos:
        at.set_color(TEXT)
    ax.set_title("Risk Tier Distribution", color=TEXT, fontsize=12, fontweight="bold")
    return _style(fig)


def scenario_comparison(scenarios: list) -> plt.Figure:
    """Bar chart comparing churn probabilities across what-if scenarios."""
    names  = [s["label"] for s in scenarios]
    probs  = [s["prob"]  for s in scenarios]
    colors = [RED if p >= 0.7 else (ORANGE if p >= 0.4 else GREEN) for p in probs]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, [p * 100 for p in probs], color=colors,
                   edgecolor=BG, height=0.55)
    ax.axvline(40, color=ORANGE, ls="--", lw=1.5, label="Medium threshold")
    ax.axvline(70, color=RED,    ls="--", lw=1.5, label="High threshold")
    ax.set_xlabel("Churn Probability (%)")
    ax.set_title("Scenario Comparison", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
    ax.grid(axis="x", color="#2d3147")
    ax.set_xlim(0, 110)
    for bar, val in zip(bars, probs):
        ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", color=TEXT, fontsize=9)
    return _style(fig)
