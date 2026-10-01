"""
Model health-check monitor.

Computes:
  - PSI  (Population Stability Index) per feature between training and current data
  - KS-statistic on the model's score distribution
  - Basic missing/summary stats

PSI thresholds (industry standard):
  <0.10  no shift
  0.10 - 0.25  moderate shift, investigate
  >0.25  significant shift, retrain
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    ref = pd.Series(reference).dropna().values
    cur = pd.Series(current).dropna().values
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    # Use reference quantiles as bin edges
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = np.maximum(ref_hist / max(ref_hist.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_hist / max(cur_hist.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_all_psi(ref_df, cur_df, features):
    rows = []
    for f in features:
        p = psi(ref_df[f].values, cur_df[f].values)
        band = "no_shift" if p < 0.10 else ("moderate" if p < 0.25 else "significant")
        rows.append({"feature": f, "psi": p, "band": band})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="data/processed/features.csv")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    ref = df[df["origination_year"] <= 2021].copy()
    cur = df[df["origination_year"] >= 2022].copy()

    # Simulate a realistic economic shift in the "current" cohort to
    # demonstrate the monitor actually catches drift. In production this
    # would be real month-over-month data.
    rng = np.random.default_rng(7)
    cur["fico"]              = np.clip(cur["fico"]        - rng.integers(15, 45, len(cur)), 500, 850)
    cur["utilization"]       = np.clip(cur["utilization"] + rng.uniform(0.03, 0.12, len(cur)), 0, 1.5)
    cur["dpd_30_last_12m"]   = np.clip(cur["dpd_30_last_12m"] + rng.integers(0, 3, len(cur)), 0, 12)
    cur["payment_ratio"]     = np.clip(cur["payment_ratio"]   - rng.uniform(0, 0.15, len(cur)), 0.01, 1.0)
    cur["balance_to_limit"]  = cur["current_balance"] / cur["credit_limit"]
    cur["dpd_severity"]      = cur["dpd_30_last_12m"] + 2 * cur["dpd_60_last_12m"] + 3 * cur["dpd_90_last_12m"]

    numeric_feats = [
        "fico", "utilization", "age", "income", "credit_limit", "current_balance",
        "months_on_book", "apr", "payment_ratio",
        "dpd_30_last_12m", "dpd_60_last_12m", "dpd_90_last_12m",
        "purchase_volume_3m", "balance_to_limit", "dpd_severity",
        "spend_to_limit", "income_to_limit",
    ]

    psi_df = compute_all_psi(ref, cur, numeric_feats)
    print("=== Feature drift (PSI) ===")
    print(psi_df.round(4).to_string(index=False))

    # Score drift
    scored = pd.read_csv(os.path.join(args.outputs_dir, "scored_test.csv"))
    with open("models/gbt.pkl", "rb") as f:
        gbt = pickle.load(f)

    NUMERIC_FEATURES = [c for c in df.columns if df[c].dtype != object and c not in ("account_id","charge_off","origination_year")]
    ref_scores = gbt.predict_proba(ref)[:, 1]
    cur_scores = gbt.predict_proba(cur)[:, 1]
    score_psi  = psi(ref_scores, cur_scores)
    score_ks   = stats.ks_2samp(ref_scores, cur_scores).statistic

    score_health = pd.DataFrame([{
        "metric": "score_psi",  "value": score_psi,
        "band":   "no_shift" if score_psi < 0.10 else ("moderate" if score_psi < 0.25 else "significant"),
    }, {
        "metric": "score_ks",   "value": score_ks,   "band": ""},
    ])

    os.makedirs(args.outputs_dir, exist_ok=True)
    psi_df.to_csv(os.path.join(args.outputs_dir, "psi_by_feature.csv"), index=False)
    score_health.to_csv(os.path.join(args.outputs_dir, "score_health.csv"), index=False)
    print("\n=== Score health ===")
    print(score_health.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
