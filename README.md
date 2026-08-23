# 🛡️ FraudShield — Financial Fraud Detection System

An end-to-end machine learning project that detects fraudulent financial transactions in real time. Built on the PaySim synthetic dataset, this project covers the full data science pipeline — from EDA and SQL analysis to a deployed XGBoost model with SHAP explainability and an AI-powered chat assistant.

🔗 **Live Demo:** [fraudshield.netlify.app](https://farhan-fraudshield.netlify.app)  
⚙️ **API:** [fraudshield-api-pdey.onrender.com](https://fraudshield-api-pdey.onrender.com/docs)

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | PaySim Synthetic Financial Transactions |
| **Rows** | 6,362,620 (6.3 Millions) transactions |
| **Features** | 24 columns (11 original + 13 engineered) |
| **Target** | `is_fraud` (binary: 0 = Legit, 1 = Fraud) |
| **Class Imbalance** | Fraud = 0.13% (8,213 rows), Legit = 99.87% |
| **Final Model** | XGBoost with Optuna hyperparameter tuning |

---

## 🗂️ Project Structure

```
financial-fraud-detection/
├── api/
│   ├── main.py              ← FastAPI backend
│   └── requirements.txt
├── app/
│   └── index.html           ← FraudShield frontend UI
├── models/
│   └── xgb_fraud_model.pkl  ← Trained XGBoost model
├── notebooks/
│   ├── 1-data_enrichment.ipynb
│   ├── 2-EDA.ipynb
│   ├── 3-SQL_Analysis.ipynb
│   └── 4-ML_Models.ipynb
└── render.yaml
```

---

## 🧬 Dataset Columns — Original vs Enriched

The base PaySim dataset only captures transaction mechanics (amount, balances, type). To simulate a realistic banking risk profile, 13 additional columns were synthetically engineered on top of it, each tied to a specific fraud pattern seen in real-world financial fraud research.

### Original PaySim Columns (11)

| Column | Description |
|--------|-------------|
| `step` | Time unit (1 step = 1 hour, simulation runs 744 steps = 30 days) |
| `type` | Transaction type — CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `nameOrig` | Sender account ID |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account ID |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Ground truth label |
| `isFlaggedFraud` | Original rule-based flag (very weak — see SQL Analysis below) |

### Enriched Columns (13) — and why they were added

| Column | Basis for Adding |
|--------|-------------------|
| `transaction_hour` | Derived from `step % 24`. Fraud shows time-of-day clustering (late night/early morning) in real fraud datasets. |
| `transaction_day_of_week` | Derived from `(step // 24) % 7`. Weekday vs weekend fraud behaviour differs in real transaction monitoring. |
| `time_of_day` | Bucketed version of the hour (late_night/morning/afternoon/evening/midnight) — easier for EDA and model to pick up cyclical risk windows. |
| `customer_age` | Added because fraud tends to disproportionately target older, less tech-savvy customers — modeled with fraud mean age ≈52 vs legit ≈38. |
| `customer_gender` | Standard demographic field used in banking risk profiling, included for completeness of the customer profile. |
| `account_age_days` | One of the strongest real-world fraud signals — freshly opened accounts are commonly used as mule/drop accounts. Fraud mean ≈46 days vs legit ≈1,223 days. |
| `is_new_account` | Binary flag (`account_age_days < 90`) built from the above, so the model/EDA doesn't have to re-derive the threshold every time. |
| `device_type` | Fraud rings often rely on specific device types (e.g. ATM-based cash-outs); added to simulate channel-level risk. |
| `channel` | Correlated with `device_type` (App/Web/ATM/POS) — mirrors how banks track the access channel of a transaction. |
| `is_international` | Cross-border transactions carry materially higher fraud risk in real banking data — modeled at 40% for fraud vs 5% for legit. |
| `account_txn_count_30d` | Captures account velocity. Made bimodal for fraud accounts — either dormant (sudden activity) or high-frequency (rapid cash-out bursts), both classic fraud velocity patterns. |
| `merchant_category` | Certain merchant categories (Gambling, Electronics) are known high-risk categories in real fraud/chargeback data — added to let the model learn category-level risk. |
| `customer_state` | Added for geographic clustering analysis. **Caveat:** PaySim's transactions are modeled on African mobile money data, so mapping to 20 Indian states is a synthetic regional overlay, not a real geographic signal — documented here so it isn't misread as ground truth. |

All enriched columns were generated using `numpy.random.seed(42)` for full reproducibility, and are documented as synthetic enrichment — not real PaySim fields.

---

## 📊 Exploratory Data Analysis (Notebook 2)

Key findings from EDA on 6.3M transactions:

- **Fraud only occurs in 2 transaction types** — `CASH_OUT` (50.12%) and `TRANSFER` (49.88%). No fraud in PAYMENT, DEBIT, or CASH_IN.
- **Fraud transactions have higher amounts** — median fraud amount is ₹441K vs ₹74K for legit transactions.
- **Fraud peaks at 2 AM and 10 AM** — lowest activity near 4 AM, spikes again after 5 AM.
- **New accounts are high risk** — ~7,600 fraud cases from new accounts vs ~500 from old accounts. Fraud account median age is ~40–50 days vs ~1,200 days for legit.
- **Older customers are most targeted** — 60+ age group has the highest fraud rate (~1.3%), while 18–30 group has the lowest (~0.03%).
- **High-risk merchant categories** — Gambling has the highest fraud rate (~1.2%), Electronics has the most fraud cases (~3,100).
- **ATM is the riskiest channel** — highest fraud rate at ~0.32% across all channels and devices.

---

## 🗃️ SQL Analysis (Notebook 3)

10 business questions answered using SQLite on the full dataset:

- **Rule-based system was failing badly** — the existing `is_flagged_fraud` system caught only **16 out of 8,213** actual frauds (99.81% missed). This justified building an ML model.
- **New accounts are fraud instruments** — fraud rate of **1.44%** for new accounts vs **0.01%** for established accounts. Fraudulent new accounts average just **10 days old**.
- **End-of-month fraud spike** — fraud rate jumps to **0.88%** in the last 10 days vs **0.09–0.10%** in earlier periods.
- **Gambling = highest fraud rate** — **1.18%** fraud rate, ~4x higher than other categories.
- **Large transactions are riskier** — top 1% transactions are **30x more likely to be fraud** (3.09% vs 0.10%), but 96% of large transactions are still legit.
- **60+ age group most vulnerable** — fraud rate of **1.28%**, nearly 32x higher than the 18–30 group.

---

## 🤖 ML Modeling (Notebook 4)

### Data Split
- Train: 80% | Test: 20% (stratified)
- Test set: 1,272,524 transactions (1,643 fraud cases)

### Why PR-AUC over ROC-AUC?
With 0.13% fraud rate (extreme class imbalance), ROC-AUC can look artificially high even for a bad model. **PR-AUC (Precision-Recall AUC) is the right metric** — it focuses only on the fraud class performance.

---

### Model 1 — Logistic Regression (Baseline)

Handles class imbalance using **SMOTE** (Synthetic Minority Oversampling).

| Metric | Score |
|--------|-------|
| Precision (Fraud) | 0.07 |
| Recall (Fraud) | 0.98 |
| F1-Score (Fraud) | 0.12 |
| ROC-AUC | 0.9983 |

**Problem:** Very low precision — model flags too many legit transactions as fraud (high false positives). Not production-ready.

---

### Model 2 — Random Forest (Optuna Tuned)

Handles imbalance using `class_weight='balanced'`. Hyperparameters tuned with **Optuna** (20 trials).

**Best Params:** `n_estimators=200, max_depth=20, min_samples_split=7, min_samples_leaf=5, max_features=sqrt`

| Metric | Score |
|--------|-------|
| Precision (Fraud) | 0.78 |
| Recall (Fraud) | 0.95 |
| F1-Score (Fraud) | 0.85 |
| ROC-AUC | 0.9999 |
| **PR-AUC** | **0.9632** |

Big improvement over Logistic Regression, but XGBoost does better.

---

### Model 3 — XGBoost ✅ Final Model

Handles imbalance using `scale_pos_weight` (ratio of legit to fraud = ~773). Hyperparameters tuned with **Optuna** (20 trials, 3-fold StratifiedKFold, optimized on PR-AUC).

**Best Params:** `n_estimators=216, max_depth=6, learning_rate=0.262, subsample=0.685, colsample_bytree=0.857, min_child_weight=1`

| Metric | Score |
|--------|-------|
| Precision (Fraud) | **0.90** |
| Recall (Fraud) | **0.97** |
| F1-Score (Fraud) | **0.94** |
| ROC-AUC | **1.0000** |
| **PR-AUC** | **0.9909** |

**Why XGBoost won:**
- Highest precision (0.90) — very few false alarms
- High recall (0.97) — catches almost all fraud
- Best PR-AUC (0.9909) — best performance on the fraud class specifically
- `scale_pos_weight` handles imbalance better than SMOTE for tree-based models

---

### Model Comparison Summary

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Logistic Regression | 0.07 | 0.98 | 0.12 | — |
| Random Forest | 0.78 | 0.95 | 0.85 | 0.9632 |
| **XGBoost ✅** | **0.90** | **0.97** | **0.94** | **0.9909** |

---

### SHAP Explainability

Every prediction is explained using **SHAP (SHapley Additive exPlanations)**:
- Shows which features pushed the model toward fraud or legit
- Negative SHAP = pushed toward legit | Positive SHAP = pushed toward fraud
- Top factors: Account Age, Transaction Type, Transaction Hour, Sender/Receiver Balance

---

## 🚀 API (FastAPI)

**Endpoint:** `POST /predict`

**Input:**
```json
{
  "amount": 85000,
  "sender_old_balance": 100000,
  "receiver_old_balance": 5000,
  "transaction_hour": 2,
  "customer_age": 35,
  "account_age_days": 365,
  "account_txn_count_30d": 10,
  "type": "TRANSFER",
  "device_type": "Mobile",
  "channel": "App",
  "merchant_category": "Electronics"
}
```

**Output:**
```json
{
  "is_fraud": 0,
  "probability": 0.003,
  "risk_level": "LOW",
  "feature_importance": [...]
}
```

---

## 💬 AI Chatbot — FraudShield Assistant

A conversational assistant is built directly into the dashboard to help users understand *why* a transaction was flagged, without digging through raw SHAP numbers themselves.

- **LLM:** Powered by the **Groq API**, running `openai/gpt-oss-120b` for fast, low-latency responses.
- **Context-aware, not generic:** On every prediction, the assistant is fed the transaction's SHAP values, key risk signals, and the model's confidence score as live context — so its answers are grounded in that specific prediction rather than generic fraud trivia.
- **Project knowledge base:** A `knowledge_base.py` file was built by extracting insights directly from the EDA, SQL analysis, and modeling notebooks, so the assistant can also answer higher-level questions about the dataset, model choices, and metrics (e.g. "why PR-AUC over accuracy?").
- **Conversation memory:** Chat history is maintained as an in-memory JS array on the frontend for the duration of the session, allowing natural follow-up questions.
- **UX:** Dedicated 420px-wide chat panel on the dashboard with 4 suggested starter questions, markdown-rendered replies (bold text, simple bullets — no heavy tables), styled to match the dark dashboard theme.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data & EDA | Python, Pandas, NumPy, Matplotlib, Seaborn |
| SQL Analysis | SQLite, SQL |
| ML | Scikit-learn, XGBoost, Optuna, SHAP, Imbalanced-learn |
| API | FastAPI, Uvicorn |
| AI Chatbot | Groq API (`openai/gpt-oss-120b`) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render (API), Netlify (Frontend), UptimeRobot (uptime monitoring) |

---

## ⚠️ Note on Dataset

The 13 enriched columns (`transaction_hour`, `customer_age`, `account_age_days`, `device_type`, `channel`, `merchant_category`, etc.) are **synthetically generated** with fraud-correlated distributions using `numpy.random.seed(42)` for reproducibility. The base 11 columns are original PaySim data. Data files are not included in this repo due to size (3GB+).
