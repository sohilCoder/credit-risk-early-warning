"""
PySpark ingestion + feature engineering (production/scale version).

Reads the raw CSV, converts it to Parquet (partitioned by origination_year),
and engineers the credit-risk features. Runs on any Spark cluster or a local
SparkSession. Requires: pyspark >= 3.5.

Usage:
    spark-submit src/feature_engineering_pyspark.py \\
        --input data/raw/accounts.csv \\
        --output data/processed/features_parquet
"""
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


def build_features(df):
    """Engineer credit-risk features on a Spark DataFrame."""
    df = df.withColumn(
        "payment_to_balance",
        F.when(F.col("current_balance") > 0,
               F.col("payment_ratio")).otherwise(F.lit(1.0)),
    ).withColumn(
        "balance_to_limit",
        F.col("current_balance") / F.col("credit_limit"),
    ).withColumn(
        "utilization_bucket",
        F.when(F.col("utilization") < 0.30, "low")
         .when(F.col("utilization") < 0.60, "medium")
         .when(F.col("utilization") < 0.90, "high")
         .otherwise("maxed"),
    ).withColumn(
        "fico_bucket",
        F.when(F.col("fico") < 620, "subprime")
         .when(F.col("fico") < 680, "near_prime")
         .when(F.col("fico") < 740, "prime")
         .otherwise("super_prime"),
    ).withColumn(
        "months_on_book_bucket",
        F.when(F.col("months_on_book") < 12, "0-12m")
         .when(F.col("months_on_book") < 24, "12-24m")
         .when(F.col("months_on_book") < 48, "24-48m")
         .otherwise("48m+"),
    ).withColumn(
        "dpd_severity",
        F.col("dpd_30_last_12m") + 2 * F.col("dpd_60_last_12m") + 3 * F.col("dpd_90_last_12m"),
    ).withColumn(
        "income_to_limit",
        F.col("income") / F.col("credit_limit"),
    ).withColumn(
        "spend_to_limit",
        F.col("purchase_volume_3m") / F.col("credit_limit"),
    )

    # Vintage-level roll rates (window over origination cohort)
    vintage_win = Window.partitionBy("origination_year")
    df = df.withColumn(
        "vintage_avg_dpd30",
        F.avg("dpd_30_last_12m").over(vintage_win),
    ).withColumn(
        "vintage_avg_utilization",
        F.avg("utilization").over(vintage_win),
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--app-name", default="CreditRiskFeatureEngineering")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName(args.app_name)
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    raw = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    print(f"Loaded {raw.count():,} rows / {len(raw.columns)} columns")

    # Clean
    cleaned = (
        raw.dropDuplicates(["account_id"])
           .na.fill({"income": 0, "employment_length": 0})
           .filter(F.col("credit_limit") > 0)
    )

    features = build_features(cleaned)

    # Write partitioned Parquet
    (
        features.repartition("origination_year")
        .write.mode("overwrite")
        .partitionBy("origination_year")
        .parquet(args.output)
    )
    print(f"Wrote partitioned Parquet to {args.output}")

    # Also emit a compact summary
    features.groupBy("origination_year").agg(
        F.count("*").alias("n_accounts"),
        F.avg("charge_off").alias("charge_off_rate"),
        F.avg("utilization").alias("avg_utilization"),
        F.avg("fico").alias("avg_fico"),
    ).orderBy("origination_year").show()

    spark.stop()


if __name__ == "__main__":
    main()
