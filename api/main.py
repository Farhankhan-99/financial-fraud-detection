from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import shap
import traceback
import os

app = FastAPI(title="FraudShield API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model load ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_fraud_model.pkl")
model      = joblib.load(MODEL_PATH)

# ── SHAP explainer — use preprocessed data ──
# TreeExplainer works on the XGB model directly (not the full pipeline)
xgb_model  = model.named_steps["model"]
explainer  = shap.TreeExplainer(xgb_model)

# ── Friendly display names for features ──
DISPLAY_NAMES = {
    "num__amount"                : "Transaction Amount",
    "num__sender_old_balance"    : "Sender Balance",
    "num__receiver_old_balance"  : "Receiver Balance",
    "num__transaction_hour"      : "Transaction Hour",
    "num__customer_age"          : "Customer Age",
    "num__account_age_days"      : "Account Age (Days)",
    "num__account_txn_count_30d" : "Txns Last 30 Days",
    "cat__type_CASH_OUT"         : "Type: CASH_OUT",
    "cat__type_DEBIT"            : "Type: DEBIT",
    "cat__type_PAYMENT"          : "Type: PAYMENT",
    "cat__type_TRANSFER"         : "Type: TRANSFER",
    "cat__device_type_Desktop"   : "Device: Desktop",
    "cat__device_type_Mobile"    : "Device: Mobile",
    "cat__device_type_POS"       : "Device: POS",
    "cat__device_type_Tablet"    : "Device: Tablet",
    "cat__channel_POS"           : "Channel: POS",
    "cat__channel_Web"           : "Channel: Web",
    "cat__channel_ATM"           : "Channel: ATM",
    "cat__merchant_category_Electronics" : "Merchant: Electronics",
    "cat__merchant_category_Food"        : "Merchant: Food",
    "cat__merchant_category_Gambling"    : "Merchant: Gambling",
    "cat__merchant_category_Retail"      : "Merchant: Retail",
    "cat__merchant_category_Travel"      : "Merchant: Travel",
    "cat__customer_gender_Male"          : "Gender: Male",
    "cat__customer_gender_Other"         : "Gender: Other",
}


class Transaction(BaseModel):
    amount                : float = Field(..., example=85000.0)
    sender_old_balance    : float = Field(..., example=100000.0)
    receiver_old_balance  : float = Field(..., example=5000.0)
    transaction_hour      : int   = Field(..., ge=0, le=23, example=2)
    customer_age          : int   = Field(..., ge=18, le=100, example=52)
    account_age_days      : int   = Field(..., ge=0, example=30)
    account_txn_count_30d : int   = Field(..., ge=0, example=3)
    type                  : Literal['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT']
    device_type           : Literal['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS']
    channel               : Literal['App', 'Web', 'ATM', 'POS']
    merchant_category     : Literal['Electronics', 'Gambling', 'Retail', 'Food', 'Travel', 'Utilities']
    customer_gender       : Literal['Male', 'Female', 'Other'] = Field(default='Male')


@app.get("/")
def home():
    return {"status": "running", "message": "FraudShield API is live", "version": "3.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Transaction):
    try:
        df = pd.DataFrame([data.dict()])

        # ── Prediction ──
        probability = model.predict_proba(df)[0][1]
        prediction  = int(probability >= 0.5)

        # ── SHAP: preprocess first, then explain ──
        preprocessor   = model.named_steps["pre"]
        X_processed    = preprocessor.transform(df)
        feature_names  = preprocessor.get_feature_names_out()
        shap_values    = explainer.shap_values(X_processed)[0]   # shape: (n_features,)

        # ── Build top-10 feature importance list ──
        shap_series = pd.Series(shap_values, index=feature_names)
        top10       = shap_series.reindex(shap_series.abs().nlargest(10).index)

        shap_output = []
        for feat, val in top10.items():
            display = DISPLAY_NAMES.get(feat, feat.replace("num__", "").replace("cat__", ""))
            shap_output.append({
                "feature"      : display,
                "shap_value"   : round(float(val), 4),
                "direction"    : "fraud" if val > 0 else "legit",
                "abs_value"    : round(abs(float(val)), 4),
            })

        return {
            "is_fraud"         : prediction,
            "probability"      : round(float(probability), 4),
            "risk_level"       : "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW",
            "feature_importance": shap_output,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})