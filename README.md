# Distributed Financial Anomaly Detection

[![CI](https://github.com/omkarbhad/distributed-financial-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/omkarbhad/distributed-financial-anomaly-detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end system for detecting unusual financial transactions at local and distributed scale.

**Project period:** April 2026 – May 2026  
**Author:** [Omkar Bhad](https://www.linkedin.com/in/omkar-bhad-data-scientist/)

## Highlights

- Distributed feature engineering and anomaly scoring with **PySpark** and **Spark MLlib**
- Local **Isolation Forest** baseline with reproducible synthetic transactions
- Real-time scoring API with **FastAPI**
- Risk-monitoring dashboard with **Streamlit** and Plotly
- Docker packaging, unit tests, linting, and GitHub Actions CI

## Architecture

```text
Transactions ──┬── pandas → behavioral features → Isolation Forest
               └── PySpark → window features → MLlib K-Means distance
                                      ↓
                         FastAPI + Streamlit dashboard
```

## Quick Start

```bash
git clone https://github.com/omkarbhad/distributed-financial-anomaly-detection.git
cd distributed-financial-anomaly-detection
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,dashboard]'
make demo
make test
make dashboard
```

Open the dashboard at `http://localhost:8501` or run `make api` for API documentation at `http://localhost:8000/docs`.

## Distributed Run

```bash
pip install -e '.[spark]'
make demo
make spark
```

The Spark job writes scored results to `artifacts/spark_scores` as Parquet.

## Project Structure

```text
src/financial_anomaly/
├── data.py          # synthetic transactions
├── features.py      # behavioral features
├── model.py         # Isolation Forest baseline
├── pipeline.py      # local workflow
├── spark_job.py     # distributed Spark workflow
├── api.py           # FastAPI service
└── dashboard.py     # Streamlit dashboard
```

## Results

The seeded 1,000-transaction demo achieved **0.80 precision** and **0.80 recall** against injected anomalies. Results are reproducible and intended for demonstration—not financial decision-making.

## Resume Alignment

This project demonstrates experience with Python, PySpark, Apache Spark, machine learning, feature engineering, model evaluation, Docker, CI/CD, API development, and data visualization.

Released under the [MIT License](LICENSE).
