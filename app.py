"""
╔══════════════════════════════════════════════════════════════╗
║   BANK CUSTOMER CHURN — PREDICTIVE MODELING DASHBOARD        ║
║   Entry point: streamlit run app.py                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
from streamlit import _config
st.set_page_config(
    page_title="Churn Intelligence | Bank Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #0F1117; color: #FFFFFF; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #141824 !important; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1F2E 0%, #1e2540 100%);
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #a0a8c0 !important; font-size: 13px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 28px !important; font-weight: 700;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Headers */
h1 { color: #4F8BF9 !important; font-weight: 800; }
h2, h3 { color: #FFFFFF !important; font-weight: 700; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4F8BF9, #764ba2);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: transform 0.15s;
}
.stButton > button:hover { transform: scale(1.04); }

/* Info / warning / success boxes */
.stAlert { border-radius: 10px; }

/* Selectbox / Slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #a0a8c0 !important; font-size: 13px;
}

/* Divider */
hr { border-color: #2d3147; }

/* Tab styling */
.stTabs [role="tab"] {
    background: #1A1F2E; border-radius: 8px 8px 0 0;
    color: #a0a8c0; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #4F8BF9 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data & models ONCE and cache in session_state ───────────────────────
import pandas as pd
import numpy as np
from utils.data_loader   import load_raw_data, preprocess, get_feature_columns, get_eda_dataframe
from utils.model_trainer import train_and_evaluate, load_artifacts, risk_tier

@st.cache_resource(show_spinner="⚙️  Training models on real data…")
def initialise():
    df_raw  = load_raw_data()
    df_eda  = get_eda_dataframe()
    df_proc = preprocess(df_raw)
    feat    = get_feature_columns(df_proc)
    X       = df_proc[feat]
    y       = df_proc["Exited"]
    results, trained, split, scaler = train_and_evaluate(X, y, feat)
    X_train, X_test, y_train, y_test = split
    probs = trained["Gradient Boosting"].predict_proba(X_test)[:, 1]
    return dict(
        df_raw=df_raw, df_eda=df_eda, df_proc=df_proc,
        feat=feat, X=X, y=y,
        results=results, trained=trained,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        scaler=scaler, best_probs=probs,
    )

state = initialise()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-size:38px;'>🏦</span><br>
        <span style='font-size:20px; font-weight:800; color:#4F8BF9;'>Churn Intelligence</span><br>
        <span style='font-size:11px; color:#6b7280;'>European Bank · 10,000 Customers</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    NAV = {
        "🏠  Home Overview":          "home",
        "📊  Exploratory Analysis":   "eda",
        "🤖  Model Performance":      "models",
        "🎯  Risk Score Calculator":  "calc",
        "🔬  What-If Simulator":      "whatif",
        "📋  Batch Risk Report":      "report",
        "📖  Project Summary":        "summary",
    }

    choice = st.radio("", list(NAV.keys()),
                      label_visibility="collapsed")
    page = NAV[choice]

    st.markdown("---")
    best_auc = max(v["roc_auc"] for v in state["results"].values())
    n_high   = sum(1 for p in state["best_probs"] if p >= 0.70)
    churn_rt = state["df_raw"]["Exited"].mean()

    st.markdown(f"""
    <div style='font-size:12px; color:#a0a8c0; line-height:2;'>
    📈 Best ROC-AUC: <b style='color:#4F8BF9'>{best_auc:.4f}</b><br>
    🔴 High-Risk (test): <b style='color:#e74c3c'>{n_high}</b><br>
    📉 Churn Rate: <b style='color:#f39c12'>{churn_rt:.1%}</b><br>
    🗂️ Rows: <b>10,000</b> | Features: <b>{len(state['feat'])}</b>
    </div>
    """, unsafe_allow_html=True)

# ── Route to pages ────────────────────────────────────────────────────────────
if   page == "home":    from pages.page_home    import render; render(state)
elif page == "eda":     from pages.page_eda     import render; render(state)
elif page == "models":  from pages.page_models  import render; render(state)
elif page == "calc":    from pages.page_calc    import render; render(state)
elif page == "whatif":  from pages.page_whatif  import render; render(state)
elif page == "report":  from pages.page_report  import render; render(state)
elif page == "summary": from pages.page_summary import render; render(state)
