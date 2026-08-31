"""
SentinelPay - Risk Analyst Dashboard
=======================================
Streamlit app that lets a risk analyst review flagged transactions,
see the ML score and engineered features, and trigger the agentic
investigation to get an explainable recommendation.

Run with:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import joblib

from train_model import engineer_features, FEATURE_COLUMNS
from fraud_agent import get_transaction_context, mock_agent_response, call_claude
import os

st.set_page_config(page_title="SentinelPay | Risk Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions.csv")
    engineered, _ = engineer_features(df)
    model = load_model()
    engineered["risk_score"] = model.predict_proba(engineered[FEATURE_COLUMNS])[:, 1]
    return engineered


st.title("🛡️ SentinelPay — AI Risk Manager")
st.caption("Real-time transaction fraud scoring with agentic investigation")

model = load_model()
df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Flagged High-Risk (>0.5)", f"{(df.risk_score > 0.5).sum():,}")
col3.metric("Confirmed Fraud", f"{df.is_fraud.sum():,}")
col4.metric("Avg Risk Score", f"{df.risk_score.mean():.3f}")

st.divider()

threshold = st.slider("Risk score threshold", 0.0, 1.0, 0.5, 0.05)
flagged = df[df.risk_score >= threshold].sort_values("risk_score", ascending=False)

st.subheader(f"Flagged Transactions ({len(flagged)})")
st.dataframe(
    flagged[[
        "transaction_id", "user_id", "amount", "txn_city", "home_city",
        "device_type", "risk_score", "is_fraud", "fraud_pattern"
    ]].head(50),
    use_container_width=True,
    height=300,
)

st.divider()
st.subheader("🔍 Investigate a Transaction")

txn_id = st.text_input("Transaction ID", value=flagged.iloc[0]["transaction_id"] if len(flagged) else "")
use_live_agent = st.checkbox(
    "Enable live AI investigation",
    value=False,
    help="If unchecked, uses the offline rule-based mock agent for demo purposes."
)

if st.button("Run Investigation", type="primary") and txn_id:
    with st.spinner("Agent investigating..."):
        raw_df = pd.read_csv("data/transactions.csv")
        context = get_transaction_context(txn_id, raw_df, model)

        if use_live_agent and os.environ.get("ANTHROPIC_API_KEY"):
            report = call_claude(context)
        else:
            if use_live_agent:
                st.warning("No ANTHROPIC_API_KEY found in environment - using mock agent instead.")
            report = mock_agent_response(context)

    action_colors = {"APPROVE": "green", "HOLD_FOR_REVIEW": "orange", "BLOCK": "red"}
    action = report["recommended_action"]
    st.markdown(f"### Recommendation: :{action_colors.get(action, 'gray')}[{action}]")
    st.write(f"**Confidence:** {report['confidence']}")
    st.write(f"**Risk score:** {context['risk_score']:.3f}")

    st.write("**Summary:**")
    st.info(report["risk_summary"])

    st.write("**Key signals:**")
    for s in report["key_signals"]:
        st.write(f"- {s}")

    st.write(f"**Alternative explanation:** {report['alternative_explanation']}")

    with st.expander("Raw transaction context"):
        st.json(context)

st.divider()
with st.expander("📊 Model Feature Importances"):
    importances = pd.read_csv("models/feature_importances.csv", index_col=0)
    st.bar_chart(importances)
