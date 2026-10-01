"""
Build the matplotlib dashboard: a single, clean set of figures the person
can drop into a presentation.

Outputs (in outputs/):
  01_portfolio_overview.png
  02_loss_forecast_by_scenario.png
  03_model_performance.png
  04_feature_drift_psi.png
  05_stress_test_heatmap.png
  06_full_dashboard.png
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

# ---------- Palette (validated for light/dark neutrality) ----------
COLORS = {
    "primary":   "#1F4E79",   # deep steel blue
    "accent":    "#C0504D",   # muted crimson
    "positive":  "#4E8E4A",   # sage green
    "warning":   "#E1A03A",   # amber
    "muted":     "#7F7F7F",   # neutral gray
    "grid":      "#E5E5E5",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       COLORS["grid"],
    "grid.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
})
os.makedirs("outputs", exist_ok=True)


def money(x, _pos):
    if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"${x/1e6:.0f}M"
    if abs(x) >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def pct(x, _pos):
    return f"{x*100:.1f}%"


# =========================================================
# 1. Portfolio overview
# =========================================================
def fig_portfolio_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Portfolio Overview", fontsize=15, fontweight="bold", y=0.995)

    # Balance by FICO bucket
    order = ["subprime", "near_prime", "prime", "super_prime"]
    bal = df.groupby("fico_bucket")["current_balance"].sum().reindex(order)
    axes[0, 0].bar(bal.index, bal.values, color=COLORS["primary"])
    axes[0, 0].set_title("Total balance by FICO band")
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(money))

    # Charge-off rate by vintage
    co = df.groupby("origination_year")["charge_off"].mean()
    axes[0, 1].plot(co.index, co.values, marker="o", color=COLORS["accent"], linewidth=2)
    axes[0, 1].set_title("Charge-off rate by origination vintage")
    axes[0, 1].yaxis.set_major_formatter(FuncFormatter(pct))
    axes[0, 1].set_xlabel("Origination year")

    # Utilization distribution
    axes[1, 0].hist(df["utilization"].clip(0, 1.2), bins=40, color=COLORS["primary"], alpha=0.85)
    axes[1, 0].set_title("Utilization distribution")
    axes[1, 0].axvline(0.30, color=COLORS["warning"], linestyle="--", label="Low/med (30%)")
    axes[1, 0].axvline(0.90, color=COLORS["accent"],  linestyle="--", label="High/maxed (90%)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("Utilization")

    # Delinquency mix
    dpd = df.groupby("fico_bucket")[["dpd_30_last_12m", "dpd_60_last_12m", "dpd_90_last_12m"]].mean().reindex(order)
    x = np.arange(len(order))
    w = 0.25
    axes[1, 1].bar(x - w, dpd["dpd_30_last_12m"], w, label="30+ DPD", color=COLORS["warning"])
    axes[1, 1].bar(x,     dpd["dpd_60_last_12m"], w, label="60+ DPD", color=COLORS["accent"])
    axes[1, 1].bar(x + w, dpd["dpd_90_last_12m"], w, label="90+ DPD", color="#7B241C")
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(order)
    axes[1, 1].set_title("Avg delinquency events (last 12m) by FICO band")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("outputs/01_portfolio_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 2. Loss forecast by scenario
# =========================================================
def fig_loss_forecast(scenarios):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stress-scenario loss forecast", fontsize=15, fontweight="bold", y=1.00)

    s = scenarios.sort_values("projected_chargeoff_rate")
    colors = [
        COLORS["positive"] if r < 0.05 else
        COLORS["warning"]  if r < 0.10 else
        COLORS["accent"]
        for r in s["projected_chargeoff_rate"]
    ]
    ax1.barh(s["scenario"], s["projected_chargeoff_rate"], color=colors)
    ax1.set_title("Projected charge-off rate")
    ax1.xaxis.set_major_formatter(FuncFormatter(pct))

    ax2.barh(s["scenario"], s["total_expected_loss_$"], color=colors)
    ax2.set_title("Total expected loss ($)")
    ax2.xaxis.set_major_formatter(FuncFormatter(money))

    plt.tight_layout()
    fig.savefig("outputs/02_loss_forecast_by_scenario.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 3. Model performance
# =========================================================
def fig_model_performance(scored, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Model performance on held-out test set", fontsize=15, fontweight="bold", y=1.02)

    # ROC
    ax = axes[0]
    for col, name, color in [
        ("score_logreg_l1", "LogReg (L1)", COLORS["muted"]),
        ("score_logreg_l2", "LogReg (L2)", COLORS["warning"]),
        ("score_gbt",       "Gradient boosting", COLORS["primary"]),
    ]:
        fpr, tpr, _ = roc_curve(scored["charge_off"], scored[col])
        auc = roc_auc_score(scored["charge_off"], scored[col])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], alpha=0.5)
    ax.set_title("ROC curves")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.legend(fontsize=8)

    # Gini / KS bar chart
    ax = axes[1]
    m = metrics.set_index("model")
    x = np.arange(len(m))
    w = 0.35
    ax.bar(x - w/2, m["gini"], w, label="Gini", color=COLORS["primary"])
    ax.bar(x + w/2, m["ks"],   w, label="KS",   color=COLORS["accent"])
    ax.set_xticks(x); ax.set_xticklabels(m.index, rotation=15)
    ax.set_title("Gini vs KS by model")
    ax.legend(fontsize=8)

    # Score distribution by outcome (GBT)
    ax = axes[2]
    ax.hist(scored.loc[scored["charge_off"] == 0, "score_gbt"], bins=40, alpha=0.6,
            color=COLORS["positive"], label="Non-default")
    ax.hist(scored.loc[scored["charge_off"] == 1, "score_gbt"], bins=40, alpha=0.6,
            color=COLORS["accent"],   label="Default")
    ax.set_title("Score distribution (gradient boosting)")
    ax.set_xlabel("Predicted probability of default"); ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("outputs/03_model_performance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 4. Drift PSI
# =========================================================
def fig_drift(psi_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    s = psi_df.sort_values("psi")
    colors = [
        COLORS["positive"] if p < 0.10 else
        COLORS["warning"]  if p < 0.25 else
        COLORS["accent"]
        for p in s["psi"]
    ]
    ax.barh(s["feature"], s["psi"], color=colors)
    ax.axvline(0.10, color=COLORS["warning"], linestyle="--", alpha=0.7, label="Moderate (0.10)")
    ax.axvline(0.25, color=COLORS["accent"],  linestyle="--", alpha=0.7, label="Significant (0.25)")
    ax.set_title("Feature drift — PSI (train vs current)", fontsize=13, fontweight="bold")
    ax.set_xlabel("PSI")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig("outputs/04_feature_drift_psi.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5. Stress heatmap
# =========================================================
def fig_stress_heatmap(scenarios):
    inputs = scenarios[["fico_shift", "util_shock", "income_shock", "dpd_shock"]].copy()
    outputs = scenarios[["avg_pd"]].copy()
    labels = scenarios["scenario"]

    fig, ax = plt.subplots(figsize=(10, 6))
    normed = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-9)
    combined = pd.concat([normed, outputs / outputs.max()], axis=1)
    im = ax.imshow(combined.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(range(combined.shape[1])); ax.set_xticklabels(combined.columns, rotation=30, ha="right")
    ax.set_title("Scenario input shocks vs projected default rate (normalized)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative magnitude")
    plt.tight_layout()
    fig.savefig("outputs/05_stress_test_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 6. Full dashboard (assembled)
# =========================================================
def fig_full_dashboard(df, scored, metrics, scenarios, psi_df):
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, hspace=0.55, wspace=0.30)

    fig.suptitle("Credit Risk Early-Warning Dashboard",
                 fontsize=18, fontweight="bold", y=0.995)

    # Row 1: charge-off by vintage + utilization
    ax = fig.add_subplot(gs[0, 0])
    co = df.groupby("origination_year")["charge_off"].mean()
    ax.plot(co.index, co.values, marker="o", color=COLORS["accent"], linewidth=2)
    ax.set_title("Charge-off rate by origination vintage")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df["utilization"].clip(0, 1.2), bins=40, color=COLORS["primary"], alpha=0.85)
    ax.set_title("Utilization distribution")
    ax.axvline(0.90, color=COLORS["accent"], linestyle="--", alpha=0.7)

    # Row 2: ROC + Gini/KS
    ax = fig.add_subplot(gs[1, 0])
    for col, name, color in [
        ("score_logreg_l1", "LogReg (L1)", COLORS["muted"]),
        ("score_logreg_l2", "LogReg (L2)", COLORS["warning"]),
        ("score_gbt",       "Gradient boosting", COLORS["primary"]),
    ]:
        fpr, tpr, _ = roc_curve(scored["charge_off"], scored[col])
        auc = roc_auc_score(scored["charge_off"], scored[col])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], alpha=0.5)
    ax.set_title("ROC curves"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    m = metrics.set_index("model")
    x = np.arange(len(m)); w = 0.35
    ax.bar(x - w/2, m["gini"], w, label="Gini", color=COLORS["primary"])
    ax.bar(x + w/2, m["ks"],   w, label="KS",   color=COLORS["accent"])
    ax.set_xticks(x); ax.set_xticklabels(m.index, rotation=15)
    ax.set_title("Gini vs KS by model"); ax.legend(fontsize=8)

    # Row 3: score distribution + confusion insight
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(scored.loc[scored["charge_off"] == 0, "score_gbt"], bins=40, alpha=0.6,
            color=COLORS["positive"], label="Non-default")
    ax.hist(scored.loc[scored["charge_off"] == 1, "score_gbt"], bins=40, alpha=0.6,
            color=COLORS["accent"],   label="Default")
    ax.set_title("Score distribution (GBT)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    thresholds = np.linspace(0.05, 0.95, 30)
    y = scored["charge_off"].values; s = scored["score_gbt"].values
    prec = [(s >= t).astype(int).dot(y) / max((s >= t).sum(), 1) for t in thresholds]
    rec  = [(s >= t).astype(int).dot(y) / max(y.sum(), 1) for t in thresholds]
    ax.plot(thresholds, prec, label="Precision", color=COLORS["primary"], linewidth=2)
    ax.plot(thresholds, rec,  label="Recall",    color=COLORS["accent"],  linewidth=2)
    ax.set_title("Precision / recall vs decision threshold"); ax.legend(fontsize=8)
    ax.set_xlabel("Threshold")

    # Row 4: stress scenarios
    ax = fig.add_subplot(gs[3, 0])
    s = scenarios.sort_values("projected_chargeoff_rate")
    colors = [
        COLORS["positive"] if r < 0.05 else
        COLORS["warning"]  if r < 0.10 else
        COLORS["accent"]
        for r in s["projected_chargeoff_rate"]
    ]
    ax.barh(s["scenario"], s["projected_chargeoff_rate"], color=colors)
    ax.set_title("Projected charge-off rate by scenario")
    ax.xaxis.set_major_formatter(FuncFormatter(pct))

    ax = fig.add_subplot(gs[3, 1])
    ax.barh(s["scenario"], s["total_expected_loss_$"], color=colors)
    ax.set_title("Total expected loss by scenario ($)")
    ax.xaxis.set_major_formatter(FuncFormatter(money))

    # Row 5: PSI drift (full width)
    ax = fig.add_subplot(gs[4, :])
    sp = psi_df.sort_values("psi")
    colors = [
        COLORS["positive"] if p < 0.10 else
        COLORS["warning"]  if p < 0.25 else
        COLORS["accent"]
        for p in sp["psi"]
    ]
    ax.barh(sp["feature"], sp["psi"], color=colors)
    ax.axvline(0.10, color=COLORS["warning"], linestyle="--", alpha=0.7, label="Moderate (0.10)")
    ax.axvline(0.25, color=COLORS["accent"],  linestyle="--", alpha=0.7, label="Significant (0.25)")
    ax.set_title("Feature drift — PSI"); ax.legend(fontsize=8)

    fig.savefig("outputs/06_full_dashboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    df        = pd.read_csv("data/processed/features.csv")
    scored    = pd.read_csv("outputs/scored_test.csv")
    metrics   = pd.read_csv("outputs/model_metrics.csv")
    scenarios = pd.read_csv("outputs/stress_scenarios.csv")
    psi_df    = pd.read_csv("outputs/psi_by_feature.csv")

    fig_portfolio_overview(df)
    fig_loss_forecast(scenarios)
    fig_model_performance(scored, metrics)
    fig_drift(psi_df)
    fig_stress_heatmap(scenarios)
    fig_full_dashboard(df, scored, metrics, scenarios, psi_df)

    print("Wrote 6 figures to outputs/")
    for f in sorted(os.listdir("outputs")):
        if f.endswith(".png"):
            print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
