from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = np.array(["card", "ach", "wire", "mobile"])
MERCHANT_CATEGORIES = np.array(["retail", "travel", "grocery", "utilities", "services"])


def generate_transactions(
    rows: int = 10_000,
    anomaly_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100")
    if not 0 < anomaly_rate < 0.5:
        raise ValueError("anomaly_rate must be between 0 and 0.5")

    rng = np.random.default_rng(seed)
    customer_count = max(100, rows // 20)
    timestamps = pd.date_range("2026-04-01", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "transaction_id": [f"txn_{index:07d}" for index in range(rows)],
            "customer_id": rng.integers(1, customer_count + 1, rows),
            "timestamp": timestamps,
            "amount": rng.lognormal(mean=3.6, sigma=0.85, size=rows).round(2),
            "channel": rng.choice(CHANNELS, rows, p=[0.50, 0.20, 0.10, 0.20]),
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, rows),
            "country_risk": rng.beta(2, 8, rows).round(4),
            "account_age_days": rng.integers(7, 3650, rows),
        }
    )
    frame["is_known_anomaly"] = 0
    _inject_anomalies(frame, anomaly_rate, rng)
    return frame


def _inject_anomalies(frame: pd.DataFrame, anomaly_rate: float, rng: np.random.Generator) -> None:
    anomaly_count = max(1, int(len(frame) * anomaly_rate))
    positions = rng.choice(frame.index, anomaly_count, replace=False)
    frame.loc[positions, "amount"] *= rng.uniform(8, 20, anomaly_count)
    frame.loc[positions, "country_risk"] = rng.uniform(0.75, 1.0, anomaly_count)
    frame.loc[positions, "account_age_days"] = rng.integers(1, 15, anomaly_count)
    frame.loc[positions, "is_known_anomaly"] = 1
