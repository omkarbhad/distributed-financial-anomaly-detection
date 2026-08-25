import pytest

from financial_anomaly.data import generate_transactions


def test_generate_transactions_is_reproducible():
    first = generate_transactions(rows=200, seed=7)
    second = generate_transactions(rows=200, seed=7)
    assert first.equals(second)
    assert first["is_known_anomaly"].sum() == 4


def test_generate_transactions_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        generate_transactions(rows=10)
    with pytest.raises(ValueError):
        generate_transactions(rows=200, anomaly_rate=0.8)

