# 🛡️ SentinelPay — Agentic AI Fraud Guardian

**Track 2: AI Risk Manager — Razorpay AI Internship 2026**

SentinelPay is a two-layer transaction fraud detection system that pairs a
trained ML risk model with an **agentic investigation layer**. Instead of
handing a risk analyst a bare probability score, SentinelPay explains *why*
a transaction is risky, checks it against plausible innocent explanations,
and recommends a concrete action — turning a black-box score into an
auditable decision.

## Why this approach

Most fraud-detection submissions stop at "train a classifier, report
accuracy." That's necessary but not sufficient in production: a risk
analyst still has to manually dig through logs to decide whether to act on
a flag, and every action needs to be justifiable in an audit trail. SentinelPay
closes that gap by having an LLM agent reason over the model's output and
the user's transaction context, then produce a structured recommendation
a human can immediately act on or override.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Transaction      │────▶│ ML Risk Model    │────▶│ Agentic Investi-   │
│ Stream           │     │ (RandomForest)   │     │ gation Layer       │
│                  │     │                  │     │ (Claude)           │
└─────────────────┘     └──────────────────┘     └──────────┬─────────┘
                                                              │
                                                              ▼
                                                  ┌────────────────────┐
                                                  │ Risk Analyst        │
                                                  │ Dashboard           │
                                                  │ (Streamlit)         │
                                                  └────────────────────┘
```

1. **Data layer** (`src/data_generator.py`) — synthetic but realistic payment
   transactions with 6 injected fraud patterns: velocity abuse, geo mismatch,
   device anomaly, amount anomaly, odd-hour activity, and card testing.
2. **ML layer** (`src/train_model.py`) — a `RandomForestClassifier` trained
   on engineered behavioral features, tuned for recall on the rare fraud
   class using `class_weight="balanced"`.
3. **Agent layer** (`src/fraud_agent.py`) — when a transaction crosses the
   risk threshold, the agent is given the score, the feature values that
   drove it, and the user's recent transaction history, and returns a
   structured JSON investigation report (summary, key signals, alternative
   explanation, recommended action, confidence).
4. **Dashboard** (`src/dashboard.py`) — a Streamlit app for browsing flagged
   transactions and triggering investigations interactively.

## Results

Trained on 50,000 synthetic transactions (2% fraud rate), evaluated on a
held-out 25% test split:

| Metric              | Value  |
|---------------------|--------|
| Precision (fraud)   | 0.77   |
| Recall (fraud)      | 0.88   |
| PR-AUC              | 0.92   |
| ROC-AUC             | 0.99   |

Top features by importance: transaction velocity (1hr window), amount vs.
user's typical spend, and device mismatch — consistent with how real
payment fraud typically presents.

*(Accuracy alone would be misleading here — a model that always predicts
"legit" scores ~98% accuracy on this data. Precision/recall/PR-AUC are the
metrics that actually matter for a 2%-positive-class problem.)*

## Setup

```bash
git clone <your-repo-url>
cd sentinelpay
pip install -r requirements.txt

# 1. Generate synthetic data
python src/data_generator.py

# 2. Train the model
python src/train_model.py

# 3. Investigate a flagged transaction (offline mock agent, no API key needed)
python src/fraud_agent.py --transaction_id <id> --mock

# 3b. Or with a live Claude agent
export ANTHROPIC_API_KEY=sk-ant-...
python src/fraud_agent.py --transaction_id <id>

# 4. Launch the dashboard
streamlit run src/dashboard.py
```

## Testing

```bash
python -m pytest tests/
```

8 unit tests cover feature engineering correctness and the mock agent's
decision logic.

## Project structure

```
sentinelpay/
├── src/
│   ├── data_generator.py   # synthetic transaction + fraud pattern generator
│   ├── train_model.py      # feature engineering + model training/eval
│   ├── fraud_agent.py      # agentic investigation layer (Claude + mock mode)
│   └── dashboard.py        # Streamlit risk analyst UI
├── tests/
│   └── test_features.py    # unit tests
├── data/                    # generated transaction data (gitignored in prod use)
├── models/                  # trained model artifacts
├── requirements.txt
└── README.md
```

## Limitations & honest caveats

- Trained on **synthetic data** — real transaction data has messier, more
  correlated fraud patterns and adversarial fraudsters actively adapting
  to detection. This is a proof of concept, not production-ready.
- The mock agent uses simple thresholds as an offline stand-in for the LLM;
  it's there so the project can be demoed/tested without an API key, not
  as a substitute for the real reasoning the Claude agent provides.
- No real-time streaming infrastructure (Kafka/etc.) — transactions are
  scored in batch/on-demand for this prototype.
- Class imbalance handling (`class_weight="balanced"`) is a reasonable
  baseline, but a production system would likely also use techniques like
  SMOTE or cost-sensitive learning tuned against real business costs of
  false positives vs. false negatives.

## Future work

- Streaming ingestion (Kafka/Kinesis) for true real-time scoring
- Feedback loop: analyst overrides retrain the model
- Multi-agent setup: a separate "policy agent" that adapts thresholds
  based on merchant risk tier
- Graph-based features (shared devices/cards across user accounts)
