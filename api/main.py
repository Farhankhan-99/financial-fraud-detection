from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import joblib
import pandas as pd
import numpy as np
import shap
import traceback
import os

## For AI-ChatBot
from groq import Groq
from dotenv import load_dotenv
from api.knowledge_base import FRAUDSHIELD_KNOWLEDGE

load_dotenv()

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

# ── Groq client ──
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


# ── Chatbot request models ──
class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str

class ChatRequest(BaseModel):
    message: str
    prediction_context: Optional[dict] = None
    history: list[ChatMessage] = []


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


@app.post("/chat")
def chat(data: ChatRequest):
    try:
        system_prompt = f"""You are FraudShield's AI assistant. You help users understand fraud predictions and answer questions about this fraud detection project (built for a Data Analyst portfolio).

{FRAUDSHIELD_KNOWLEDGE}

CURRENT PREDICTION CONTEXT (the transaction just analyzed):
{data.prediction_context if data.prediction_context else "No prediction has been made yet. Ask the user to run a prediction first if they ask about a specific transaction."}

INSTRUCTIONS:
- For transaction-specific questions (why fraud, what factors mattered, transaction type, what's unusual), use the CURRENT PREDICTION CONTEXT and feature importance data.
- If the user asks a transaction-specific question but NO prediction has been made yet, politely tell them to run a prediction first — do not guess or make up numbers.
- For project/technical questions (model used, accuracy, tech stack, dataset), use the knowledge base above.
- Reply in English by default. If the user writes in Hindi or Hinglish, reply in that same style.
- If asked something unrelated to fraud/this project, politely redirect back to FraudShield topics.

## RESPONSE STYLE
Talk like a normal helpful chat assistant, not like a document or report.
- Answer directly first, in plain conversational sentences.
- Keep it short — 2 to 5 sentences for most questions.
- Use **bold** occasionally for a key word or number, not entire phrases.
- Only use a short bullet list ("- point") if you're listing 3+ distinct things, and keep each bullet to one line.
- Do NOT use headings (no #, ##, ###).
- Do NOT use tables (no "|" pipe characters), even if comparing things — just describe it in a sentence or short bullets instead.
- Only use a code block if the user explicitly asks for code, SQL, or a formula.
- No "Key Points" / "Bottom Line" / "Direct Answer" style labels — just talk naturally, like ChatGPT's default replies.
- Never over-format a simple answer — a one-line question deserves a one or two line answer."""

        messages = [{"role": "system", "content": system_prompt}]
        for h in data.history:
            messages.append({"role": h.role, "content": h.content})
        messages.append({"role": "user", "content": data.message})

        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.3,
                max_tokens=400,
            )
            reply = response.choices[0].message.content
        except Exception as groq_error:
            traceback.print_exc()
            return {"reply": "Assistant is busy right now, please try again in a moment."}

        return {"reply": reply}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})