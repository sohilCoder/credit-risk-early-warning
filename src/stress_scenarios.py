"""
Stress-scenario simulation for baseline and adverse macro conditions.

Perturbs key drivers (unemployment proxy, FICO shift, income shock,
utilization shock) and re-scores the test population under 12 scenarios,
producing projected charge-off rates.
"""
import os
import pickle
import numpy as np
import pandas as pd

SCENARIOS = [
    # name,                fico_shift, util_shock, income_shock, dpd_shock
    ("baseline",             0,          0.00,      0.00,          0),
    ("mild_recession",     -15,          0.05,     -0.05,          1),
    ("moderate_recession", -30,          0.10,     -0.10,          2),
    ("severe_recession",   -60,          0.20,     -0.20,          4),
    ("rate_shock",           0,          0.08,      0.00,          1),
    ("unemployment_5pct",  -20,          0.07,     -0.08,          2),
    ("unemployment_10pct", -45,          0.15,     -0.15,          3),
    ("housing_crash",      -25,          0.12,     -0.10,          2),
    ("inflation_spike",    -10,          0.10,     -0.05,          1),
    ("consumer_spending_drop", -5,       0.03,     -0.15,          0),
    ("credit_tightening",  -10,         -0.05,     -0.02,          0),
    ("stagflation",        -20,          0.12,     -0.12,          2),
]


def apply_scenario(df, fico_shift, util_shock, income_shock, dpd_shock):
    d = df.copy()
    d["fico"] = np.clip(d["fico"] + fico_shift, 500, 850)
    d["utilization"] = np.clip(d["utilization"] + util_shock, 0, 1.5)
    d["income"] = np.clip(d["income"] * (1 + income_shock), 6_000, None)
    d["dpd_30_last_12m"] = np.clip(d["dpd_30_last_12m"] + dpd_shock, 0, 12)
    d["dpd_severity"] = d["dpd_30_last_12m"] + 2 * d["dpd_60_last_12m"] + 3 * d["dpd_90_last_12m"]
    d["balance_to_limit"] = d["current_balance"] / d["credit_limit"]
    return d


def main():
    df = pd.read_csv("data/processed/features.csv")
    with open("models/gbt.pkl", "rb") as f:
        gbt = pickle.load(f)

    test = df[df["origination_year"] >= 2022].copy()

    rows = []
    for name, fs, us, is_, ds in SCENARIOS:
        stressed = apply_scenario(test, fs, us, is_, ds)
        p = gbt.predict_proba(stressed)[:, 1]
        expected_loss = (p * stressed["current_balance"]).sum()
        rows.append({
            "scenario": name,
            "fico_shift": fs,
            "util_shock": us,
            "income_shock": is_,
            "dpd_shock": ds,
            "avg_pd":  p.mean(),
            "projected_chargeoff_rate": p.mean(),
            "total_expected_loss_$": expected_loss,
            "n_accounts": len(stressed),
        })

    result = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    result.to_csv("outputs/stress_scenarios.csv", index=False)
    print("=== Stress scenarios ===")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
