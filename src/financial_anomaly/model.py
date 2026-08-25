from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financial_anomaly.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_model(contamination: float = 0.02, seed: int = 42) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("detector", detector)])


def score_transactions(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()
    raw_score = model.decision_function(features)
    scored["anomaly_score"] = (-raw_score).round(6)
    scored["is_anomaly"] = (model.predict(features) == -1).astype(int)
    return scored.sort_values("anomaly_score", ascending=False)

