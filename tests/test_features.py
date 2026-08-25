import pytest

from financial_anomaly.data import generate_transactions
from financial_anomaly.features import build_features


def test_build_features_adds_behavioral_signals():
    features = build_features(generate_transactions(rows=200))
    expected = {"hour", "is_night", "customer_transaction_count", "amount_to_customer_mean"}
    assert expected.issubset(features)
    assert features["amount_to_customer_mean"].notna().all()


def test_build_features_reports_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_features(generate_transactions(rows=200).drop(columns="channel"))
