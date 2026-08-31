"""
SentinelPay - Synthetic Transaction Data Generator
====================================================
Generates realistic payment transaction data with injected fraud patterns
for training and evaluating the fraud detection model.

Fraud patterns simulated:
  1. Velocity fraud    - many transactions in a short window
  2. Geo mismatch      - transaction location far from user's usual location
  3. Device anomaly    - new/unrecognized device making high-value transaction
  4. Amount anomaly    - transaction size far outside user's normal spend
  5. Odd-hour fraud    - transactions at unusual hours for that user
  6. Card testing      - sequence of small amounts followed by a large one
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

np.random.seed(42)

N_USERS = 2000
N_TRANSACTIONS = 50000
FRAUD_RATE = 0.02  # ~2% of transactions are fraudulent (realistic for payments)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "food_delivery", "fashion",
    "utilities", "entertainment", "healthcare", "education", "gaming"
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"
]

DEVICE_TYPES = ["mobile_app", "web_desktop", "web_mobile", "pos_terminal"]


def generate_users(n_users):
    """Create a base population of users with a 'home' city, typical spend, etc."""
    users = []
    for i in range(n_users):
        users.append({
            "user_id": f"U{i:05d}",
            "home_city": np.random.choice(CITIES),
            "avg_txn_amount": np.random.gamma(shape=2.0, scale=800),  # INR
            "typical_hour_mean": np.random.normal(14, 4) % 24,       # typical hour of day
            "primary_device": np.random.choice(DEVICE_TYPES),
            "account_age_days": np.random.randint(10, 2000),
        })
    return pd.DataFrame(users)


def generate_transactions(users_df, n_transactions, fraud_rate):
    """Generate transactions, injecting fraud patterns for a subset."""
    rows = []
    start_date = datetime(2026, 1, 1)

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud
    fraud_flags = np.array([0] * n_legit + [1] * n_fraud)
    np.random.shuffle(fraud_flags)

    for i in range(n_transactions):
        user = users_df.sample(1).iloc[0]
        is_fraud = fraud_flags[i]
        txn_time = start_date + timedelta(
            days=np.random.randint(0, 240),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )

        if not is_fraud:
            amount = max(10, np.random.normal(user["avg_txn_amount"], user["avg_txn_amount"] * 0.3))
            city = user["home_city"] if np.random.rand() > 0.05 else np.random.choice(CITIES)
            device = user["primary_device"] if np.random.rand() > 0.05 else np.random.choice(DEVICE_TYPES)
            hour = int(txn_time.hour)
            velocity_1h = np.random.poisson(0.3)
            fraud_pattern = "none"
        else:
            pattern = np.random.choice(
                ["velocity", "geo_mismatch", "device_anomaly", "amount_anomaly", "odd_hour", "card_testing"]
            )
            fraud_pattern = pattern

            if pattern == "velocity":
                amount = max(10, np.random.normal(user["avg_txn_amount"], user["avg_txn_amount"] * 0.5))
                city = user["home_city"]
                device = user["primary_device"]
                hour = int(txn_time.hour)
                velocity_1h = np.random.poisson(8) + 5  # burst of transactions
            elif pattern == "geo_mismatch":
                amount = max(10, np.random.normal(user["avg_txn_amount"] * 1.5, user["avg_txn_amount"] * 0.5))
                other_cities = [c for c in CITIES if c != user["home_city"]]
                city = np.random.choice(other_cities)
                device = np.random.choice(DEVICE_TYPES)
                hour = int(txn_time.hour)
                velocity_1h = np.random.poisson(1)
            elif pattern == "device_anomaly":
                amount = max(10, user["avg_txn_amount"] * np.random.uniform(2, 6))
                city = user["home_city"]
                device = np.random.choice([d for d in DEVICE_TYPES if d != user["primary_device"]])
                hour = int(txn_time.hour)
                velocity_1h = np.random.poisson(1)
            elif pattern == "amount_anomaly":
                amount = user["avg_txn_amount"] * np.random.uniform(8, 25)
                city = user["home_city"]
                device = user["primary_device"]
                hour = int(txn_time.hour)
                velocity_1h = np.random.poisson(0.5)
            elif pattern == "odd_hour":
                amount = max(10, np.random.normal(user["avg_txn_amount"], user["avg_txn_amount"] * 0.4))
                city = user["home_city"]
                device = np.random.choice(DEVICE_TYPES)
                hour = np.random.choice([1, 2, 3, 4])  # 1-4 AM
                velocity_1h = np.random.poisson(2)
            else:  # card_testing
                amount = np.random.uniform(1, 50)  # tiny probing amounts
                city = user["home_city"]
                device = np.random.choice(DEVICE_TYPES)
                hour = int(txn_time.hour)
                velocity_1h = np.random.poisson(6) + 4

        rows.append({
            "transaction_id": str(uuid.uuid4())[:12],
            "user_id": user["user_id"],
            "timestamp": txn_time,
            "amount": round(amount, 2),
            "merchant_category": np.random.choice(MERCHANT_CATEGORIES),
            "txn_city": city,
            "home_city": user["home_city"],
            "device_type": device,
            "primary_device": user["primary_device"],
            "hour_of_day": hour,
            "velocity_1h": velocity_1h,
            "account_age_days": user["account_age_days"],
            "avg_user_amount": round(user["avg_txn_amount"], 2),
            "is_fraud": int(is_fraud),
            "fraud_pattern": fraud_pattern,
        })

    return pd.DataFrame(rows)


def main():
    print("Generating user base...")
    users_df = generate_users(N_USERS)

    print(f"Generating {N_TRANSACTIONS} transactions ({FRAUD_RATE*100:.1f}% fraud rate)...")
    txns_df = generate_transactions(users_df, N_TRANSACTIONS, FRAUD_RATE)
    txns_df = txns_df.sort_values("timestamp").reset_index(drop=True)

    users_df.to_csv("data/users.csv", index=False)
    txns_df.to_csv("data/transactions.csv", index=False)

    print(f"\nDone. {len(txns_df)} transactions written to data/transactions.csv")
    print(f"Fraud count: {txns_df['is_fraud'].sum()} ({txns_df['is_fraud'].mean()*100:.2f}%)")
    print("\nFraud pattern breakdown:")
    print(txns_df[txns_df.is_fraud == 1]["fraud_pattern"].value_counts())


if __name__ == "__main__":
    main()
