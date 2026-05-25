"""
OTT Churn Prediction — Training Pipeline
Run once to generate data, train models, and persist artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data import generate_ott_dataset
from src.preprocessing import get_train_test
from src.model_training import train_all

DATA_PATH = Path("data/ott_churn_data.csv")


def main():
    print("=" * 60)
    print("  OTT CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Generate / load data
    if not DATA_PATH.exists():
        print("\n[1/3] Generating synthetic OTT dataset …")
        df = generate_ott_dataset(5000)
        df.to_csv(DATA_PATH, index=False)
        print(f"      Saved → {DATA_PATH}  ({len(df):,} rows)")
    else:
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        print(f"\n[1/3] Loaded existing dataset  ({len(df):,} rows)")

    print(f"      Churn rate: {df['churn'].mean():.2%}")

    # 2. Preprocess
    print("\n[2/3] Preprocessing …")
    X_train, X_test, y_train, y_test, feature_names, encoders, scaler = get_train_test(df)
    print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 3. Train
    print("\n[3/3] Training models …")
    results, best_name, feat_imp = train_all(X_train, X_test, y_train, y_test, feature_names)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    for name, res in results.items():
        m = res["metrics"]
        marker = "  ★ BEST" if name == best_name else ""
        print(f"  {name:<22}  Acc={m['Accuracy']:.3f}  F1={m['F1-Score']:.3f}  AUC={m['ROC-AUC']:.3f}{marker}")

    print("\n  Artifacts saved in models/")
    print("  Run `streamlit run app/streamlit_app.py` to launch the dashboard.\n")


if __name__ == "__main__":
    main()
