from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from financial_anomaly.features import build_features
from financial_anomaly.model import score_transactions

MODEL_PATH = Path("artifacts/isolation_forest.joblib")
app = FastAPI(title="Financial Anomaly Detection API", version="1.0.0")


class Transaction(BaseModel):
    transaction_id: str
    customer_id: int
    timestamp: str
    amount: float = Field(gt=0)
    channel: str
    merchant_category: str
    country_risk: float = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "ready" if MODEL_PATH.exists() else "missing"}


@app.post("/score")
def score(transaction: Transaction) -> dict[str, float | int | str]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Run `make demo` to create the model")
    frame = pd.DataFrame([transaction.model_dump()])
    scored = score_transactions(joblib.load(MODEL_PATH), build_features(frame)).iloc[0]
    return {
        "transaction_id": transaction.transaction_id,
        "anomaly_score": float(scored["anomaly_score"]),
        "is_anomaly": int(scored["is_anomaly"]),
    }

