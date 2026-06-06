# 🛡️ FraudShield — Financial Fraud Detection System

An end-to-end machine learning project that detects fraudulent financial transactions in real time. Built on the PaySim synthetic dataset, this project covers the full data science pipeline — from EDA and SQL analysis to a deployed XGBoost model with SHAP explainability.

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

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data & EDA | Python, Pandas, NumPy, Matplotlib, Seaborn |
| SQL Analysis | SQLite, SQL |
| ML | Scikit-learn, XGBoost, Optuna, SHAP, Imbalanced-learn |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render (API), Netlify (Frontend) |

---

## ⚠️ Note on Dataset

The 13 enriched columns (`transaction_hour`, `customer_age`, `account_age_days`, `device_type`, `channel`, `merchant_category`, etc.) are **synthetically generated** with fraud-correlated distributions using `numpy.random.seed(42)` for reproducibility. The base 11 columns are original PaySim data. Data files are not included in this repo due to size (3GB+).
