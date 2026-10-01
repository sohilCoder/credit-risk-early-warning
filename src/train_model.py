"""
Train and evaluate two default-prediction models:
  1. Logistic regression with L1 and L2 regularization  (interpretable baseline)
  2. Gradient-boosted trees (HistGradientBoostingClassifier)  (strong model)

Both are compared with Gini, KS, ROC-AUC on a time-based train/test split.
Saves fitted models and a metrics table.
"""
import argparse
import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats

TARGET = "charge_off"
NUMERIC_FEATURES = [
    "age", "income", "employment_length", "fico", "utilization",
    "total_tradelines", "inquiries_6m", "credit_limit", "current_balance",
    "months_on_book", "apr", "payment_ratio", "dpd_30_last_12m",
    "dpd_60_last_12m", "dpd_90_last_12m", "purchase_volume_3m",
    "payment_to_balance", "balance_to_limit", "dpd_severity",
    "income_to_limit", "spend_to_limit", "vintage_avg_dpd30",
    "vintage_avg_utilization",
]
CATEGORICAL_FEATURES = ["state", "utilization_bucket", "fico_bucket", "months_on_book_bucket"]


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic between default and non-default score distributions."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    return stats.ks_2samp(pos, neg).statistic


def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_test, y_test):
    scores = model.predict_proba(X_test)[:, 1]
    preds  = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "model": name,
        "auc":   roc_auc_score(y_test, scores),
        "gini":  gini(y_test, scores),
        "ks":    ks_statistic(y_test, scores),
        "precision_@0.5": tp / max(tp + fp, 1),
        "recall_@0.5":    tp / max(tp + fn, 1),
        "n_test": len(y_test),
        "positive_rate": float(y_test.mean()),
    }, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df):,} rows")

    # Time-based split: train on 2018-2021 vintages, test on 2022-2023
    train = df[df["origination_year"] <= 2021].copy()
    test  = df[df["origination_year"] >= 2022].copy()
    print(f"  train: {len(train):,} rows (vintages {sorted(train['origination_year'].unique())})")
    print(f"  test : {len(test):,}  rows (vintages {sorted(test['origination_year'].unique())})")

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test,  y_test  = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES],  test[TARGET]

    results = []
    scored_test = test[["account_id", "origination_year", TARGET]].copy()

    # ---- Logistic regression (L1) ----
    print("\nTraining Logistic Regression (L1)...")
    lr_l1 = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=0.5,
                                   max_iter=1000, n_jobs=-1)),
    ])
    lr_l1.fit(X_train, y_train)
    metrics, scores = evaluate("logreg_l1", lr_l1, X_test, y_test)
    results.append(metrics)
    scored_test["score_logreg_l1"] = scores
    print(f"  {metrics}")

    # ---- Logistic regression (L2) ----
    print("\nTraining Logistic Regression (L2)...")
    lr_l2 = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                   max_iter=1000, n_jobs=-1)),
    ])
    lr_l2.fit(X_train, y_train)
    metrics, scores = evaluate("logreg_l2", lr_l2, X_test, y_test)
    results.append(metrics)
    scored_test["score_logreg_l2"] = scores
    print(f"  {metrics}")

    # ---- Gradient-boosted trees ----
    print("\nTraining HistGradientBoostingClassifier...")
    gbt = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=42)),
    ])
    gbt.fit(X_train, y_train)
    metrics, scores = evaluate("gbt", gbt, X_test, y_test)
    results.append(metrics)
    scored_test["score_gbt"] = scores
    print(f"  {metrics}")

    # Save
    with open(os.path.join(args.models_dir, "logreg_l1.pkl"), "wb") as f: pickle.dump(lr_l1, f)
    with open(os.path.join(args.models_dir, "logreg_l2.pkl"), "wb") as f: pickle.dump(lr_l2, f)
    with open(os.path.join(args.models_dir, "gbt.pkl"),      "wb") as f: pickle.dump(gbt,   f)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(args.outputs_dir, "model_metrics.csv"), index=False)
    scored_test.to_csv(os.path.join(args.outputs_dir, "scored_test.csv"), index=False)

    print("\n=== Model comparison ===")
    print(metrics_df.round(4).to_string(index=False))

    # Save training feature importances (for GBT)
    try:
        pre = gbt.named_steps["pre"]
        num_names = NUMERIC_FEATURES
        cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        feat_names = num_names + cat_names
        # HistGradientBoosting doesn't expose feature_importances_ pre-1.4; use permutation as fallback
        clf = gbt.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = pd.DataFrame({"feature": feat_names, "importance": clf.feature_importances_})
            importances.sort_values("importance", ascending=False).to_csv(
                os.path.join(args.outputs_dir, "feature_importances.csv"), index=False)
    except Exception as e:
        print(f"(feature-importance export skipped: {e})")


if __name__ == "__main__":
    main()
