from __future__ import annotations

import argparse

import joblib
from sklearn.metrics import precision_score, recall_score

from financial_anomaly.config import ProjectPaths
from financial_anomaly.data import generate_transactions
from financial_anomaly.features import build_features
from financial_anomaly.model import create_model, score_transactions


def run_pipeline(
    rows: int = 10_000,
    anomaly_rate: float = 0.02,
    seed: int = 42,
) -> dict[str, float]:
    paths = ProjectPaths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions(rows, anomaly_rate, seed)
    features = build_features(transactions)
    model = create_model(anomaly_rate, seed)
    model.fit(features)
    scored = score_transactions(model, features)

    transactions.to_csv(paths.transactions, index=False)
    scored.to_csv(paths.scores, index=False)
    joblib.dump(model, paths.model)

    return {
        "rows": float(rows),
        "flagged": float(scored["is_anomaly"].sum()),
        "precision": precision_score(scored["is_known_anomaly"], scored["is_anomaly"]),
        "recall": recall_score(scored["is_known_anomaly"], scored["is_anomaly"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local anomaly detection demo")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    metrics = run_pipeline(**vars(parser.parse_args()))
    print(" | ".join(f"{name}: {value:.3f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
