from __future__ import annotations

import argparse
import math

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

FEATURE_COLUMNS = [
    "amount",
    "country_risk",
    "account_age_days",
    "hour",
    "customer_transaction_count",
    "amount_to_customer_mean",
]


def build_distributed_features(transactions):
    customer_window = Window.partitionBy("customer_id")
    return (
        transactions.withColumn("timestamp", F.to_timestamp("timestamp"))
        .withColumn("hour", F.hour("timestamp").cast("double"))
        .withColumn("customer_transaction_count", F.count("*").over(customer_window).cast("double"))
        .withColumn("customer_mean_amount", F.avg("amount").over(customer_window))
        .withColumn(
            "amount_to_customer_mean",
            F.col("amount") / F.greatest(F.col("customer_mean_amount"), F.lit(0.01)),
        )
    )


def add_cluster_distance(dataset, centers):
    @F.udf(T.DoubleType())
    def distance(cluster: int, vector) -> float:
        center = centers[cluster]
        squared_differences = (
            (left - right) ** 2 for left, right in zip(vector, center, strict=True)
        )
        return float(math.sqrt(sum(squared_differences)))

    return dataset.withColumn("anomaly_score", distance("prediction", "scaled_features"))


def run(
    input_path: str,
    output_path: str,
    clusters: int = 8,
    anomaly_fraction: float = 0.02,
) -> None:
    spark = SparkSession.builder.appName("financial-anomaly-detection").getOrCreate()
    transactions = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
    featured = build_distributed_features(transactions)
    assembled = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features").transform(featured)
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    scaled = scaler.fit(assembled).transform(assembled)
    kmeans = KMeans(k=clusters, seed=42, featuresCol="scaled_features")
    model = kmeans.fit(scaled)
    scored = add_cluster_distance(model.transform(scaled), model.clusterCenters())
    threshold = scored.approxQuantile("anomaly_score", [1 - anomaly_fraction], 0.001)[0]
    result = scored.withColumn("is_anomaly", (F.col("anomaly_score") >= threshold).cast("int"))
    result.drop("features", "scaled_features").write.mode("overwrite").parquet(output_path)
    spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Spark anomaly detection job")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--anomaly-fraction", type=float, default=0.02)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
