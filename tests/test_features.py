"""
Basic unit tests for SentinelPay's feature engineering and mock agent.
Run with: python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from train_model import engineer_features
from fraud_agent import mock_agent_response


def make_txn(**overrides):
    base = {
        "transaction_id": "t1",
        "user_id": "U00001",
        "amount": 1000.0,
        "merchant_category": "grocery",
        "txn_city": "Mumbai",
        "home_city": "Mumbai",
        "device_type": "mobile_app",
        "primary_device": "mobile_app",
        "hour_of_day": 14,
        "velocity_1h": 0,
        "account_age_days": 500,
        "avg_user_amount": 1000.0,
        "is_fraud": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_geo_mismatch_flagged():
    df = make_txn(txn_city="Delhi", home_city="Mumbai")
    engineered, _ = engineer_features(df)
    assert engineered.iloc[0]["geo_mismatch"] == 1


def test_no_geo_mismatch_when_same_city():
    df = make_txn(txn_city="Mumbai", home_city="Mumbai")
    engineered, _ = engineer_features(df)
    assert engineered.iloc[0]["geo_mismatch"] == 0


def test_device_mismatch_flagged():
    df = make_txn(device_type="pos_terminal", primary_device="mobile_app")
    engineered, _ = engineer_features(df)
    assert engineered.iloc[0]["device_mismatch"] == 1


def test_amount_ratio_computed_correctly():
    df = make_txn(amount=5000.0, avg_user_amount=1000.0)
    engineered, _ = engineer_features(df)
    assert abs(engineered.iloc[0]["amount_ratio"] - 5.0) < 1e-6


def test_odd_hour_flagged_for_early_morning():
    df = make_txn(hour_of_day=3)
    engineered, _ = engineer_features(df)
    assert engineered.iloc[0]["is_odd_hour"] == 1


def test_odd_hour_not_flagged_for_afternoon():
    df = make_txn(hour_of_day=14)
    engineered, _ = engineer_features(df)
    assert engineered.iloc[0]["is_odd_hour"] == 0


def test_mock_agent_blocks_extreme_amount_anomaly():
    context = {
        "risk_score": 0.95,
        "feature_values": {
            "velocity_1h": 1, "amount_ratio": 12.0,
            "device_mismatch": 0, "geo_mismatch": 0,
        },
        "transaction": {}, "recent_history": [],
    }
    report = mock_agent_response(context)
    assert report["recommended_action"] == "BLOCK"


def test_mock_agent_approves_low_risk_transaction():
    context = {
        "risk_score": 0.1,
        "feature_values": {
            "velocity_1h": 0, "amount_ratio": 1.0,
            "device_mismatch": 0, "geo_mismatch": 0,
        },
        "transaction": {}, "recent_history": [],
    }
    report = mock_agent_response(context)
    assert report["recommended_action"] == "APPROVE"
