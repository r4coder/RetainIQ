import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                             use_label_encoder=False, eval_metric="logloss",
                                             random_state=42, n_jobs=-1),
    }


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0, 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba) if y_proba is not None else ([], [], [])
    return metrics, cm, fpr, tpr


def train_all(X_train, X_test, y_train, y_test, feature_names):
    models     = get_models()
    results    = {}
    best_name  = None
    best_score = 0.0

    for name, model in models.items():
        print(f"  Training {name} …")
        model.fit(X_train, y_train)
        metrics, cm, fpr, tpr = evaluate(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics, "cm": cm, "fpr": fpr, "tpr": tpr}
        if metrics["ROC-AUC"] > best_score:
            best_score = metrics["ROC-AUC"]
            best_name  = name

    best_model = results[best_name]["model"]

    # Feature importance (XGBoost preferred, else RF, else None)
    feat_imp = None
    for preferred in ["XGBoost", "Random Forest"]:
        if preferred in results:
            m = results[preferred]["model"]
            if hasattr(m, "feature_importances_"):
                feat_imp = pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)
                break

    # Persist artifacts
    joblib.dump(best_model,  MODELS_DIR / "best_model.pkl")
    joblib.dump(results,     MODELS_DIR / "all_results.pkl")
    joblib.dump(feat_imp,    MODELS_DIR / "feature_importance.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    print(f"\n  Best model: {best_name}  (ROC-AUC = {best_score:.4f})")
    return results, best_name, feat_imp


def churn_risk_label(prob: float) -> str:
    if prob < 0.35:   return "Low Risk"
    if prob < 0.65:   return "Medium Risk"
    return "High Risk"


def retention_suggestions(row: dict, prob: float) -> list[str]:
    tips = []
    if row.get("days_since_last_login", 0) > 30:
        tips.append("📧 Send a personalized re-engagement email with curated top picks.")
    if row.get("monthly_watch_hours", 99) < 10:
        tips.append("🎬 Recommend a binge-worthy series based on their favorite genre.")
    if row.get("support_tickets", 0) >= 2:
        tips.append("🛠️ Proactively reach out to resolve open support issues.")
    if not row.get("auto_renew_enabled", 1):
        tips.append("🔄 Prompt the user to enable auto-renewal with a limited-time discount.")
    if row.get("subscription_type") == "Basic":
        tips.append("⬆️ Offer a free 14-day Standard/Premium trial upgrade.")
    if prob > 0.65:
        tips.append("💎 Send a personalized retention offer: 20% off next 3 months.")
    if not tips:
        tips.append("✅ User appears healthy. Engage with loyalty rewards or referral programs.")
    return tips
