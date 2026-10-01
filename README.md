# Credit Risk Early-Warning Dashboard

An end-to-end credit-risk modeling project that ingests consumer-credit account
data, engineers risk features with PySpark, trains and benchmarks default-prediction
models, simulates 12 macroeconomic stress scenarios, monitors feature drift, and
produces a matplotlib dashboard for stakeholders.

Built to demonstrate every skill Synchrony's Credit Modeling internship lists:
PySpark, Python, pandas, sklearn, statsmodels, regularized logistic regression,
tree-based models, PSI / KS drift metrics, stress testing, model governance, and
matplotlib visualization.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the whole pipeline end-to-end
bash run_all.sh
```

Six figures land in `outputs/`, along with a metrics CSV, PSI table, and
stress-scenario table.

---

## What each step does

| # | Script                                  | Purpose                                                              |
|---|-----------------------------------------|----------------------------------------------------------------------|
| 1 | `src/generate_synthetic_data.py`        | Simulate 500k consumer-credit accounts with a realistic default label |
| 2 | `src/feature_engineering_pyspark.py`    | **Production**: Spark → partitioned Parquet feature store            |
|   | `src/feature_engineering_pandas.py`     | Local equivalent when Spark isn't available                          |
| 3 | `src/train_model.py`                    | Fit LogReg-L1, LogReg-L2, gradient-boosted trees; save models        |
| 4 | `src/monitor_drift.py`                  | Compute PSI per feature + score PSI/KS between train and current data |
| 5 | `src/stress_scenarios.py`               | 12 macroeconomic scenarios → projected charge-off rates and losses    |
| 6 | `src/build_dashboard.py`                | Six matplotlib figures + a full assembled dashboard                   |

---

## Results (from the last full run on synthetic data)

**Model performance** (time-based test split, 167k accounts from 2022–2023 vintages)

| Model         | AUC   | Gini  | KS    | Precision @0.5 | Recall @0.5 |
|---------------|-------|-------|-------|----------------|-------------|
| LogReg L1     | 0.744 | 0.487 | 0.361 | 0.596          | 0.077       |
| LogReg L2     | 0.744 | 0.487 | 0.361 | 0.595          | 0.077       |
| GBT           | 0.742 | 0.484 | 0.358 | 0.591          | 0.074       |

**Stress-scenario loss forecast** — projected total charge-offs on the test book

| Scenario               | Projected charge-off | Expected loss |
|------------------------|----------------------|---------------|
| Baseline               | 15.1 %               | \$84 M         |
| Mild recession         | 22.6 %               | \$125 M        |
| Moderate recession     | 31.9 %               | \$174 M        |
| Severe recession       | 54.9 %               | \$296 M        |
| Unemployment +10 %     | 43.6 %               | \$236 M        |
| Housing crash          | 31.3 %               | \$171 M        |

**Drift monitor** — with a simulated macro shift in the current cohort, the
monitor flags `dpd_30_last_12m` (PSI 0.87), `utilization` (0.25) and `fico` (0.19)
as significant/moderate drift.

---

## Data required

By default the project generates its own synthetic data with `generate_synthetic_data.py`
so nothing external is needed. To swap in real data:

Any account-level consumer-credit dataset with the following columns will
work — column names in the CSV should match exactly:

| Column               | Type    | Meaning                                            |
|----------------------|---------|----------------------------------------------------|
| `account_id`         | int     | Unique account identifier                          |
| `origination_year`   | int     | Vintage the account was opened                     |
| `age`                | int     | Cardholder age                                     |
| `income`             | float   | Reported annual income                             |
| `employment_length`  | int     | Years in current employment                        |
| `state`              | string  | US state code                                      |
| `fico`               | int     | Bureau FICO score                                  |
| `utilization`        | float   | Balance ÷ credit limit at bureau level             |
| `total_tradelines`   | int     | Open tradelines on bureau                          |
| `inquiries_6m`       | int     | Hard inquiries in last 6 months                    |
| `credit_limit`       | float   | Credit limit                                       |
| `current_balance`    | float   | Statement balance                                  |
| `months_on_book`     | int     | Months since account opened                        |
| `apr`                | float   | Purchase APR                                       |
| `payment_ratio`      | float   | Monthly payment ÷ balance                          |
| `dpd_30_last_12m`    | int     | Times 30+ days past due in last 12 months          |
| `dpd_60_last_12m`    | int     | Times 60+ days past due in last 12 months          |
| `dpd_90_last_12m`    | int     | Times 90+ days past due in last 12 months          |
| `purchase_volume_3m` | float   | Purchase volume last 3 months                      |
| `charge_off`         | int 0/1 | Target: did the account charge off within horizon? |

**Real-world dataset drop-in options:**

1. **Lending Club loan data** (Kaggle) — ~2.3 M records, most fields present or derivable. Rename `loan_status = 'Charged Off' → charge_off = 1`.
2. **Freddie Mac Single-Family Loan-Level dataset** — mortgage rather than credit-card, but the same modeling approach applies.
3. **Federal Reserve Y-14M schedules** — the actual data banks report for CCAR stress tests (requires appropriate access).
4. **Your own portfolio data** — internal snapshot with the columns above.

Point the pipeline at your file:
```bash
python3 src/feature_engineering_pandas.py --input path/to/your_data.csv
python3 src/train_model.py
python3 src/monitor_drift.py
python3 src/stress_scenarios.py
python3 src/build_dashboard.py
```

---

## Project layout

```
credit_risk_project/
├── run_all.sh              # end-to-end runner
├── requirements.txt
├── src/
│   ├── generate_synthetic_data.py
│   ├── feature_engineering_pyspark.py    # production/scale
│   ├── feature_engineering_pandas.py     # local
│   ├── train_model.py
│   ├── monitor_drift.py
│   ├── stress_scenarios.py
│   └── build_dashboard.py
├── data/
│   ├── raw/           # synthetic accounts.csv (43 MB)
│   └── processed/     # feature-engineered CSV / Parquet
├── models/            # pickled sklearn pipelines
├── outputs/           # figures, metrics, PSI, stress tables
└── docs/
    └── model_card.md
```
