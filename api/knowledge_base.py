FRAUDSHIELD_KNOWLEDGE = """
=== FRAUDSHIELD KNOWLEDGE BASE ===

IMPORTANT DISCLAIMER:
- This project uses the PaySim dataset — a SYNTHETIC (simulated) financial dataset, NOT real bank/customer data.
- Several columns (customer_age, device_type, channel, merchant_category, customer_state, etc.) were synthetically generated with fraud-correlated distributions for portfolio/demo purposes, using numpy.random.seed(42) for reproducibility.
- If asked "is this real data?" or "is this a real bank?", clearly state it is synthetic data used to demonstrate an end-to-end fraud detection pipeline, not a production system with real customers.

DATASET:
- PaySim synthetic dataset, 6.36M transactions, 0.13% fraud rate (highly imbalanced)
- Fraud occurs ONLY in CASH_OUT (50.12%) and TRANSFER (49.88%) types — never in PAYMENT, DEBIT, CASH_IN

MODEL INPUT COLUMNS (13, used for training):
type — transaction type (CASH_OUT/TRANSFER/PAYMENT/DEBIT/CASH_IN)
amount — transaction amount
sender_old_balance — sender's balance before transaction
receiver_old_balance — receiver's balance before transaction
transaction_hour — hour of day (0-23)
customer_age — customer's age
customer_gender — Male/Female/Other
account_age_days — account age in days
device_type — Mobile/Desktop/Tablet/ATM/POS
channel — App/Web/ATM/POS
account_txn_count_30d — transactions in last 30 days
merchant_category — Electronics/Gambling/Travel/Food/Retail/Utilities

KEY EDA INSIGHTS:
- Fraud amounts are much higher (median ₹441K) vs legit (median ₹74K)
- Fraud peaks at 2 AM and 10 AM; lowest around 4 AM; Wed 10AM and late nights (Mon/Tue) spike
- New accounts (~46 days old) drive fraud (~7,600 cases) vs old accounts (~500 cases)
- Fraud rate rises with age: 60+ has highest (~1.3%), 18-30 lowest (~0.03%) — older customers more targeted
- Gambling (~1.2%) and Electronics (highest volume, 3,110 cases) are riskiest merchant categories
- ATM channel/device has highest fraud rate (~0.32%) — physical channels riskier than online
- Top fraud states: Maharashtra (0.48%), Delhi (0.41%), Karnataka (0.35%), Tamil Nadu (0.30%)

KEY SQL INSIGHTS:
- Old rule-based system (isFlaggedFraud) caught only 16 of 8,213 frauds (99.81% missed) — shows why ML model was needed
- New accounts: 1.44% fraud rate vs 0.01% for old accounts; fraud accounts avg only 10 days old
- Fraud spikes in last 10 days of month (0.88%) vs early/mid month (0.09-0.10%)
- Zero-balance-drain alone is a weak fraud signal (only 0.22% fraud rate)
- Top 1% largest transactions are 30x more likely fraud (3.09% vs 0.10%), but amount alone isn't reliable (96% of big txns are legit)
- ATM has highest fraud RATE across all top states; App has more fraud COUNT but lower rate

MODEL PERFORMANCE (final model: XGBoost):
- Logistic Regression (baseline): F1=0.12, ROC-AUC=0.998 (poor precision, high false positives)
- Random Forest: Precision=0.78, Recall=0.95, F1=0.85, PR-AUC=0.963
- XGBoost (FINAL): Precision=0.90, Recall=0.97, F1=0.94, PR-AUC=0.9909, ROC-AUC=1.00
- Used PR-AUC as key metric (not ROC-AUC) since data is highly imbalanced
- SMOTE used inside imblearn Pipeline for class imbalance (train-only, no leakage)
"""