"""
SentinelPay - Agentic Fraud Investigation Layer
==================================================
When the ML model flags a transaction as high-risk, this agent gathers
context (user history, feature values, model reasoning) and produces a
structured, explainable investigation report using Claude.

This turns a black-box risk score into an auditable decision a human
risk analyst can act on immediately - which is the core value prop of
an "AI Risk Manager" vs. a plain classifier.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python src/fraud_agent.py --transaction_id <id>

    # Or run in mock mode (no API key needed, for demos/offline testing):
    python src/fraud_agent.py --transaction_id <id> --mock
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd

from train_model import engineer_features, FEATURE_COLUMNS

MODEL_PATH = "models/fraud_model.joblib"
ENCODER_PATH = "models/merchant_encoder.joblib"
DATA_PATH = "data/transactions.csv"

AGENT_SYSTEM_PROMPT = """You are a fraud risk investigation agent for a payments company.
You are given a flagged transaction, the ML model's risk score, the feature values that
drove that score, and the user's recent transaction history.

Your job:
1. Assess WHY this transaction looks risky, in plain language a human analyst can verify.
2. Weigh it against legitimate explanations (e.g. user is traveling, made a big purchase).
3. Recommend exactly one action: APPROVE, HOLD_FOR_REVIEW, or BLOCK.
4. Give a confidence level (low/medium/high) in your recommendation.

Respond ONLY as JSON with this exact shape, no other text:
{
  "risk_summary": "<2-3 sentence plain-language explanation>",
  "key_signals": ["<signal 1>", "<signal 2>", ...],
  "alternative_explanation": "<one plausible innocent explanation, or 'none apparent'>",
  "recommended_action": "APPROVE" | "HOLD_FOR_REVIEW" | "BLOCK",
  "confidence": "low" | "medium" | "high"
}"""


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, encoder, df


def get_transaction_context(txn_id, df, model):
    """Pull the transaction, its user's recent history, and the model's score/features."""
    txn = df[df.transaction_id == txn_id]
    if txn.empty:
        raise ValueError(f"Transaction {txn_id} not found")
    txn = txn.iloc[0]

    user_id = txn["user_id"]
    user_history = df[df.user_id == user_id].sort_values("timestamp").tail(10)

    engineered, _ = engineer_features(pd.DataFrame([txn]))
    X = engineered[FEATURE_COLUMNS]
    risk_score = float(model.predict_proba(X)[0, 1])

    feature_values = X.iloc[0].to_dict()

    return {
        "transaction": txn.to_dict(),
        "risk_score": risk_score,
        "feature_values": feature_values,
        "recent_history": user_history[
            ["timestamp", "amount", "merchant_category", "txn_city", "device_type"]
        ].to_dict(orient="records"),
    }


def build_user_prompt(context):
    return f"""Flagged transaction:
{json.dumps(context['transaction'], indent=2, default=str)}

ML model risk score: {context['risk_score']:.3f} (0=safe, 1=high risk)

Feature values that drove this score:
{json.dumps(context['feature_values'], indent=2, default=str)}

User's last 10 transactions (for context on normal behavior):
{json.dumps(context['recent_history'], indent=2, default=str)}

Investigate this transaction and respond with the JSON format specified in your instructions."""


def call_claude(context):
    """Real call to the Anthropic API. Requires ANTHROPIC_API_KEY in the environment."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=AGENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(context)}],
    )
    text = message.content[0].text
    return json.loads(text)


def mock_agent_response(context):
    """
    Deterministic offline stand-in for the LLM call, used for demos/testing
    without an API key. Mirrors the same reasoning structure the real agent
    would produce, using simple thresholds on the engineered features.
    """
    fv = context["feature_values"]
    signals = []
    action = "APPROVE"
    confidence = "medium"

    if fv["velocity_1h"] > 4:
        signals.append(f"Unusually high transaction velocity ({fv['velocity_1h']:.0f} txns in 1 hour)")
        action = "HOLD_FOR_REVIEW"
    if fv["amount_ratio"] > 5:
        signals.append(f"Transaction is {fv['amount_ratio']:.1f}x this user's typical spend")
        action = "BLOCK"
    if fv["device_mismatch"] == 1:
        signals.append("Made from a device not previously associated with this user")
        if action == "APPROVE":
            action = "HOLD_FOR_REVIEW"
    if fv["geo_mismatch"] == 1:
        signals.append("Transaction location differs from user's home city")

    if not signals:
        signals.append("No single dominant risk signal, but combined score exceeded threshold")

    if context["risk_score"] > 0.85:
        confidence = "high"
    elif context["risk_score"] < 0.6:
        confidence = "low"

    return {
        "risk_summary": (
            f"Model risk score {context['risk_score']:.2f}. "
            f"Flagged primarily due to: {signals[0].lower()}."
        ),
        "key_signals": signals,
        "alternative_explanation": (
            "User may be traveling or making a planned large purchase"
            if fv["geo_mismatch"] or fv["amount_ratio"] > 3
            else "none apparent"
        ),
        "recommended_action": action,
        "confidence": confidence,
    }


def investigate(txn_id, mock=False):
    model, encoder, df = load_artifacts()
    context = get_transaction_context(txn_id, df, model)

    if mock or not os.environ.get("ANTHROPIC_API_KEY"):
        if not mock:
            print("[info] No ANTHROPIC_API_KEY set - falling back to mock agent.", file=sys.stderr)
        report = mock_agent_response(context)
    else:
        report = call_claude(context)

    report["transaction_id"] = txn_id
    report["risk_score"] = round(context["risk_score"], 4)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction_id", required=True)
    parser.add_argument("--mock", action="store_true", help="Force offline mock agent (no API key needed)")
    args = parser.parse_args()

    result = investigate(args.transaction_id, mock=args.mock)
    print(json.dumps(result, indent=2))
