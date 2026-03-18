"""
Model training, evaluation, and persistence utilities.
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, precision_recall_curve)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42
TEST_SIZE    = 0.2


def build_models():
    """Return dict of untrained model instances."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, solver="saga", random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                            max_depth=4, random_state=RANDOM_STATE),
        "Voting Ensemble":     VotingClassifier(
                                   estimators=[
                                       ("lr",  LogisticRegression(max_iter=3000, solver="saga", random_state=RANDOM_STATE)),
                                       ("rf",  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
                                       ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
                                   ],
                                   voting="soft"
                               ),
    }


def train_and_evaluate(X, y, feature_cols):
    """
    Train all models, return:
        results   – dict[model_name] -> metrics dict
        trained   – dict[model_name] -> fitted model object
        split     – (X_train, X_test, y_train, y_test)
        scaler    – fitted StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models  = build_models()
    results = {}
    trained = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        use_scaled = (name == "Logistic Regression")
        Xtr = X_train_sc if use_scaled else X_train.values
        Xte = X_test_sc  if use_scaled else X_test.values

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        cv_auc = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        prec_c, rec_c, _    = precision_recall_curve(y_test, y_prob)

        results[name] = {
            "accuracy":   accuracy_score(y_test, y_pred),
            "precision":  precision_score(y_test, y_pred),
            "recall":     recall_score(y_test, y_pred),
            "f1":         f1_score(y_test, y_pred),
            "roc_auc":    roc_auc_score(y_test, y_prob),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std":  cv_auc.std(),
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "y_pred":      y_pred,
            "y_prob":      y_prob,
            "fpr":         fpr,
            "tpr":         tpr,
            "prec_curve":  prec_c,
            "rec_curve":   rec_c,
        }
        trained[name] = model

    # Save to disk
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(trained, os.path.join(MODELS_DIR, "trained_models.pkl"))
    joblib.dump(scaler,  os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump({"X_test": X_test, "y_test": y_test,
                 "X_train": X_train, "y_train": y_train,
                 "feature_cols": feature_cols},
                os.path.join(MODELS_DIR, "split_data.pkl"))

    return results, trained, (X_train, X_test, y_train, y_test), scaler


def load_artifacts():
    """Load previously saved model artifacts."""
    trained = joblib.load(os.path.join(MODELS_DIR, "trained_models.pkl"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    split   = joblib.load(os.path.join(MODELS_DIR, "split_data.pkl"))
    return trained, scaler, split


def predict_single(model, scaler, feature_cols, customer_dict, use_scaled=False):
    """Predict churn probability for a single customer dict."""
    row = {col: customer_dict.get(col, 0) for col in feature_cols}
    df  = pd.DataFrame([row])
    X   = scaler.transform(df) if use_scaled else df.values
    return float(model.predict_proba(X)[0][1])


def risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def risk_color(tier: str) -> str:
    return {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}[tier]


def get_feature_importance(model, feature_cols):
    """Return sorted feature importance series for tree-based models."""
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        return fi.sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        fi = pd.Series(np.abs(model.coef_[0]), index=feature_cols)
        return fi.sort_values(ascending=False)
    return pd.Series(dtype=float)
