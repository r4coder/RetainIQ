import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["gender", "subscription_type", "favorite_genre",
                    "payment_method", "ads_tolerance", "binge_watch_frequency"]

NUMERIC_COLS = ["age", "monthly_watch_hours", "devices_used", "avg_session_time",
                "monthly_subscription_cost", "support_tickets", "days_since_last_login",
                "number_of_profiles", "content_rating_given", "account_age_months", "auto_renew_enabled"]

TARGET = "churn"
DROP_COLS = ["customer_id"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, fit: bool = True,
               encoders: dict = None, scaler: StandardScaler = None):
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore", inplace=True)

    # Fill missing values
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns and encoders and col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]
                    if x in le.classes_ else -1
                )

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df[TARGET] if TARGET in df.columns else None

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    else:
        X_scaled = scaler.transform(X) if scaler else X.values

    return X_scaled, y, encoders, scaler, list(X.columns)


def get_train_test(df: pd.DataFrame):
    X_scaled, y, encoders, scaler, feature_names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names, encoders, scaler
