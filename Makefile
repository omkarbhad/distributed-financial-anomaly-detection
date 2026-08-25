.PHONY: install demo test lint api dashboard spark clean

install:
	python -m pip install -e '.[dev,dashboard]'

demo:
	python -m financial_anomaly.pipeline --rows 10000

test:
	pytest -q

lint:
	ruff check .

api:
	uvicorn financial_anomaly.api:app --reload

dashboard:
	streamlit run src/financial_anomaly/dashboard.py

spark:
	spark-submit src/financial_anomaly/spark_job.py --input data/generated/transactions.csv --output artifacts/spark_scores

clean:
	rm -rf artifacts data/generated .pytest_cache .ruff_cache

