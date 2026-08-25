from financial_anomaly.data import generate_transactions
from financial_anomaly.features import build_features
from financial_anomaly.model import create_model, score_transactions


def test_model_scores_requested_fraction():
    features = build_features(generate_transactions(rows=500, anomaly_rate=0.02))
    model = create_model(contamination=0.02)
    model.fit(features)
    scored = score_transactions(model, features)
    assert scored["is_anomaly"].sum() == 10
    assert scored["anomaly_score"].is_monotonic_decreasing

