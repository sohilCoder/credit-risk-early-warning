"""
Pandas equivalent of feature_engineering_pyspark.py.

Same logic, single-machine implementation, used for local development and
for the environments where a Spark cluster isn't available. The two files
share the same feature schema so downstream code doesn't care which ran.
"""
import argparse
import os
import numpy as np
import pandas as pd


def fico_bucket(x):
    if x < 620: return "subprime"
    if x < 680: return "near_prime"
    if x < 740: return "prime"
    return "super_prime"


def util_bucket(x):
    if x < 0.30: return "low"
    if x < 0.60: return "medium"
    if x < 0.90: return "high"
    return "maxed"


def mob_bucket(x):
    if x < 12:  return "0-12m"
    if x < 24:  return "12-24m"
    if x < 48:  return "24-48m"
    return "48m+"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["payment_to_balance"]  = df["payment_ratio"]
    df["balance_to_limit"]    = df["current_balance"] / df["credit_limit"]
    df["utilization_bucket"]  = df["utilization"].apply(util_bucket)
    df["fico_bucket"]         = df["fico"].apply(fico_bucket)
    df["months_on_book_bucket"] = df["months_on_book"].apply(mob_bucket)
    df["dpd_severity"]        = df["dpd_30_last_12m"] + 2 * df["dpd_60_last_12m"] + 3 * df["dpd_90_last_12m"]
    df["income_to_limit"]     = df["income"] / df["credit_limit"]
    df["spend_to_limit"]      = df["purchase_volume_3m"] / df["credit_limit"]

    # Vintage-level roll rates
    vintage = df.groupby("origination_year").agg(
        vintage_avg_dpd30=("dpd_30_last_12m", "mean"),
        vintage_avg_utilization=("utilization", "mean"),
    ).reset_index()
    df = df.merge(vintage, on="origination_year", how="left")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/accounts.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows / {len(df.columns)} columns")

    df = df.drop_duplicates("account_id")
    df = df.fillna({"income": 0, "employment_length": 0})
    df = df[df["credit_limit"] > 0]

    feats = build_features(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    feats.to_csv(args.output, index=False)
    print(f"Wrote features to {args.output}")

    summary = feats.groupby("origination_year").agg(
        n_accounts=("account_id", "count"),
        charge_off_rate=("charge_off", "mean"),
        avg_utilization=("utilization", "mean"),
        avg_fico=("fico", "mean"),
    )
    print(summary.round(4))


if __name__ == "__main__":
    main()
