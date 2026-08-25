from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SCORES_PATH = Path("artifacts/scored_transactions.csv")

st.set_page_config(page_title="Financial Anomaly Monitor", layout="wide")
st.title("Distributed Financial Anomaly Monitor")
st.caption("Portfolio project · April–May 2026")

if not SCORES_PATH.exists():
    st.info("Run `make demo` first to generate scored transactions.")
    st.stop()

scores = pd.read_csv(SCORES_PATH, parse_dates=["timestamp"])
flagged = scores[scores["is_anomaly"] == 1]
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{len(scores):,}")
col2.metric("Flagged", f"{len(flagged):,}")
col3.metric("Flag rate", f"{len(flagged) / len(scores):.2%}")

st.plotly_chart(
    px.scatter(
        scores,
        x="timestamp",
        y="amount",
        color="is_anomaly",
        hover_data=["transaction_id", "channel", "anomaly_score"],
        title="Transaction Amounts and Detected Anomalies",
    ),
    use_container_width=True,
)
st.subheader("Highest-risk transactions")
st.dataframe(flagged.head(100), use_container_width=True, hide_index=True)

