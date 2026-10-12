"""
Phase 13: Monitoring & Maintenance
---------------------------------------
Drift monitoring via Population Stability Index (PSI), consistent with
the pattern established across this series (e.g. the Telco churn
project's monitor.py). PSI compares the distribution of a feature in a
recent production window against the distribution seen at training time:

    PSI < 0.1   -> no significant shift
    0.1 - 0.25  -> moderate shift, worth watching
    > 0.25      -> significant shift, investigate / consider retraining

MONITORED FEATURES: duration, credit_amount, age, and the engineered
monthly_payment_burden -- the four continuous features most likely to
shift if the applicant pool changes (e.g. the lender starts marketing to
a different income segment, or loan amounts/terms drift with the
market). The categorical no_checking_account flag is also tracked
directly, since EDA showed it carries unusually strong (and
counter-intuitive) signal for this specific use case.

RETRAIN TRIGGER: in addition to feature drift, this project tracks
recall on the bad-risk class against the test-set baseline recorded in
business_validation.json -- a drop of more than 0.05 (i.e. missing
meaningfully more bad-risk applicants than at validation time) is the
concrete, business-relevant signal that the model's catch-rate on the
costly error type has degraded, which is what actually matters given
the 5:1 cost asymmetry (missing more bad-risk applicants is far more
expensive than being slightly more conservative on good ones).
"""

import json

import numpy as np
import pandas as pd

MONITORED_NUMERIC_FEATURES = ["duration", "credit_amount", "age", "monthly_payment_burden"]
RECALL_DROP_TRIGGER = 0.05


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    # WHY quantile-based bins on the reference distribution: fixed-width
    # bins would be thrown off by this dataset's skewed credit_amount
    # distribution (a handful of very large loans). Quantile bins from the
    # training-time distribution keep each bin roughly equally populated
    # at training time, which is what makes a later shift in bin
    # population actually meaningful rather than an artifact of binning.
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = reference.quantile(quantiles).values.copy()
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_verdict(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate shift -- watch"
    else:
        return "significant shift -- investigate"


def run_drift_check(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    print("=== Feature drift (PSI) ===")
    for feature in MONITORED_NUMERIC_FEATURES:
        psi = compute_psi(reference_df[feature], current_df[feature])
        print(f"{feature:24s}  PSI={psi:.4f}  ({psi_verdict(psi)})")

    ref_flag_rate = reference_df["no_checking_account"].mean()
    cur_flag_rate = current_df["no_checking_account"].mean()
    print(f"\nno_checking_account rate: reference={ref_flag_rate:.1%}  current={cur_flag_rate:.1%}")


def check_recall_drop(current_recall: float, baseline_path: str = "outputs/business_validation.json") -> bool:
    with open(baseline_path) as f:
        baseline = json.load(f)
    baseline_recall = baseline["test_recall"]
    drop = baseline_recall - current_recall
    triggered = drop > RECALL_DROP_TRIGGER
    print(f"\n=== Retrain trigger check ===")
    print(f"Baseline test recall (bad-risk class): {baseline_recall:.3f}")
    print(f"Current recall: {current_recall:.3f}")
    print(f"Drop: {drop:.3f}  (trigger threshold: {RECALL_DROP_TRIGGER})")
    print(f"Retrain triggered: {triggered}")
    return triggered


if __name__ == "__main__":
    # Demonstration run: compares the saved engineered training data
    # against itself as a stand-in for a "current production window",
    # since this project has no live production feed. In a real
    # deployment, `current_df` would be the last N days of scored
    # applications logged by app.py.
    reference_df = pd.read_csv("outputs/engineered_data.csv")
    current_df = reference_df.sample(frac=0.3, random_state=7)  # simulated recent window

    run_drift_check(reference_df, current_df)

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)
    # Demonstration: assume current recall matches baseline (no live
    # scored-outcome feed exists yet) -- a real deployment would compute
    # this from recently matured loan outcomes.
    check_recall_drop(current_recall=baseline["test_recall"])
