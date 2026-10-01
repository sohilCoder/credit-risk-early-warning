"""
Synthetic consumer credit-card dataset generator.

Produces a realistic account-level dataset with:
  - Demographics (age, income, employment length, state)
  - Bureau attributes (FICO score, revolving utilization, total credit lines)
  - Account attributes (credit limit, current balance, months on book, APR)
  - Behavioral attributes (payment ratio, delinquency history, purchase volume)
  - A charge-off (default) label whose probability depends on the above
    through a hidden logistic function -- so ML models can actually learn it.

Output:
  data/raw/accounts.csv   (~500k rows by default; configurable)
"""
import argparse
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
US_STATES = ["TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
             "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI", "CO", "MN"]


def generate(n_rows: int = 500_000) -> pd.DataFrame:
    # ----- Demographics -----
    age = np.clip(RNG.normal(42, 13, n_rows), 18, 85).astype(int)
    income = np.clip(RNG.lognormal(mean=10.9, sigma=0.55, size=n_rows), 12_000, 500_000).astype(int)
    employment_length = np.clip(RNG.gamma(shape=2.0, scale=3.5, size=n_rows), 0, 40).astype(int)
    state = RNG.choice(US_STATES, size=n_rows)

    # ----- Bureau -----
    fico = np.clip(RNG.normal(710, 65, n_rows), 500, 850).astype(int)
    utilization = np.clip(RNG.beta(2, 5, n_rows) + RNG.normal(0, 0.05, n_rows), 0.0, 1.5)
    total_tradelines = np.clip(RNG.poisson(8, n_rows), 1, 40)
    inquiries_6m = np.clip(RNG.poisson(1.2, n_rows), 0, 15)

    # ----- Account -----
    credit_limit = np.clip((income * RNG.uniform(0.05, 0.30, n_rows)).round(-2), 500, 50_000)
    current_balance = np.clip(credit_limit * utilization, 0, credit_limit * 1.1)
    months_on_book = np.clip(RNG.integers(3, 120, n_rows), 3, 120)
    apr = np.clip(29 - (fico - 500) * 0.03 + RNG.normal(0, 2, n_rows), 8.99, 34.99).round(2)

    # ----- Behavior -----
    payment_ratio = np.clip(RNG.beta(2, 3, n_rows), 0.01, 1.0)   # payment / balance
    dpd_30_last_12m = np.clip(RNG.poisson(0.4, n_rows), 0, 12)
    dpd_60_last_12m = np.clip((dpd_30_last_12m * RNG.uniform(0.2, 0.6, n_rows)).astype(int), 0, 12)
    dpd_90_last_12m = np.clip((dpd_60_last_12m * RNG.uniform(0.3, 0.7, n_rows)).astype(int), 0, 12)
    purchase_volume_3m = np.clip(RNG.gamma(2, credit_limit * 0.05, n_rows), 0, credit_limit * 3)

    # ----- Origination cohort (vintage) -----
    origination_year = RNG.integers(2018, 2024, n_rows)

    # ----- Hidden default-generating function (logistic) -----
    # These weights are what a good model should approximately recover.
    z = (
        -3.5
        + (700 - fico) * 0.012                     # lower FICO -> more risk
        + utilization * 1.8                        # higher util -> more risk
        + dpd_30_last_12m * 0.35
        + dpd_60_last_12m * 0.55
        + dpd_90_last_12m * 0.85
        + (1 - payment_ratio) * 1.4                # low payment -> more risk
        + inquiries_6m * 0.10
        + (30_000 - income) * 5e-6
        + (apr - 20) * 0.03
        + RNG.normal(0, 0.4, n_rows)               # noise
    )
    p_default = 1.0 / (1.0 + np.exp(-z))
    charge_off = (RNG.uniform(0, 1, n_rows) < p_default).astype(int)

    df = pd.DataFrame({
        "account_id": np.arange(1, n_rows + 1),
        "origination_year": origination_year,
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "state": state,
        "fico": fico,
        "utilization": utilization.round(4),
        "total_tradelines": total_tradelines,
        "inquiries_6m": inquiries_6m,
        "credit_limit": credit_limit,
        "current_balance": current_balance.round(2),
        "months_on_book": months_on_book,
        "apr": apr,
        "payment_ratio": payment_ratio.round(4),
        "dpd_30_last_12m": dpd_30_last_12m,
        "dpd_60_last_12m": dpd_60_last_12m,
        "dpd_90_last_12m": dpd_90_last_12m,
        "purchase_volume_3m": purchase_volume_3m.round(2),
        "charge_off": charge_off,
    })
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--out", type=str, default="data/raw/accounts.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"  charge-off rate: {df['charge_off'].mean():.3%}")
    print(f"  size on disk: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
