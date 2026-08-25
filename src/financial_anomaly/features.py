from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "amount",
    "country_risk",
    "account_age_days",
    "hour",
    "is_night",
    "customer_transaction_count",
    "amount_to_customer_mean",
]
CATEGORICAL_FEATURES = ["channel", "merchant_category"]


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {"customer_id", "timestamp", "amount", "channel", "merchant_category"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

    featured = transactions.copy()
    featured["timestamp"] = pd.to_datetime(featured["timestamp"], utc=True)
    featured["hour"] = featured["timestamp"].dt.hour
    featured["is_night"] = featured["hour"].between(0, 5).astype(int)
    customer_groups = featured.groupby("customer_id")["amount"]
    featured["customer_transaction_count"] = customer_groups.transform("count")
    customer_mean = customer_groups.transform("mean").clip(lower=0.01)
    featured["amount_to_customer_mean"] = featured["amount"] / customer_mean
    return featured

