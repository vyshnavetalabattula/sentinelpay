"""
SentinelPay - Fraud Detection Model Training
==============================================
Trains a RandomForest classifier on engineered transaction features.
Evaluated on precision/recall/PR-AUC rather than accuracy, since fraud
is a rare-event problem (~2% positive class) where accuracy is misleading.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score
)

FEATURE_COLUMNS = [
    "amount", "hour_of_day", "velocity_1h", "account_age_days",
    "amount_ratio", "geo_mismatch", "device_mismatch", "is_odd_hour",
    "merchant_category_enc",
]


def engineer_features(df):
    """Turn raw transaction fields into model-ready signals."""
    df = df.copy()

    # Ratio of this transaction to the user's typical spend - catches amount anomalies
    df["amount_ratio"] = df["amount"] / df["avg_user_amount"].replace(0, 1)

    # Binary flags the model can learn sharp thresholds around
    df["geo_mismatch"] = (df["txn_city"] != df["home_city"]).astype(int)
    df["device_mismatch"] = (df["device_type"] != df["primary_device"]).astype(int)
    df["is_odd_hour"] = df["hour_of_day"].apply(lambda h: 1 if h < 5 else 0)

    le = LabelEncoder()
    df["merchant_category_enc"] = le.fit_transform(df["merchant_category"])

    return df, le


def train():
    print("Loading transactions...")
    df = pd.read_csv("data/transactions.csv")
    df, merchant_encoder = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Fraud rate - train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")

    # class_weight="balanced" matters a lot here: without it, a model can get
    # 98% accuracy by just predicting "not fraud" every time.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== Classification Report (threshold=0.5) ===")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Feature importance - useful for the agent's explanations later
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\n=== Feature Importances ===")
    print(importances)

    joblib.dump(model, "models/fraud_model.joblib")
    joblib.dump(merchant_encoder, "models/merchant_encoder.joblib")
    importances.to_csv("models/feature_importances.csv")
    print("\nModel saved to models/fraud_model.joblib")

    return model, merchant_encoder


if __name__ == "__main__":
    train()
