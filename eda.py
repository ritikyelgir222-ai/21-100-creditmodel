"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: a lending-ops stakeholder's first question is
whether the "alternative" signals (checking-account status, employment
duration, telephone registration) actually separate good from bad risk
as well as traditional credit-history data does -- worth checking
directly, since the whole premise of alternative credit scoring is that
these signals CAN substitute when bureau data is thin.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data

CHECKING_MAP = {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no checking account"}
SAVINGS_MAP = {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">=1000 DM", "A65": "unknown/none"}
EMPLOYMENT_MAP = {"A71": "unemployed", "A72": "<1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">=7 years"}
HISTORY_MAP = {
    "A30": "no credits/all paid", "A31": "all paid (this bank)", "A32": "existing paid duly",
    "A33": "delay in past", "A34": "critical/other credits",
}

def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()
    df["is_bad_risk"] = (df["target"] == 2).astype(int)

    print(f"Overall bad-risk rate: {df['is_bad_risk'].mean():.1%}")

    print("\n=== Bad-risk rate by checking-account status (THE key underbanked signal) ===")
    checking_rates = df.assign(label=df["checking_status"].map(CHECKING_MAP)).groupby("label")["is_bad_risk"].mean().sort_values(ascending=False)
    print(checking_rates.round(3))
    print("-> Having NO checking account at all associates with the LOWEST bad-risk rate here,")
    print("   not the highest -- a genuinely counter-intuitive finding worth stating plainly")
    print("   rather than assuming 'no formal banking = higher risk.' (A plausible explanation:")
    print("   applicants with substantial savings elsewhere may simply not need a checking")
    print("   account -- correlation, not necessarily causation, is what EDA can show here.)")

    print("\n=== Bad-risk rate by savings-account status ===")
    print(df.assign(label=df["savings_status"].map(SAVINGS_MAP)).groupby("label")["is_bad_risk"].mean().sort_values(ascending=False).round(3))

    print("\n=== Bad-risk rate by employment duration ===")
    print(df.assign(label=df["employment_duration"].map(EMPLOYMENT_MAP)).groupby("label")["is_bad_risk"].mean().sort_values(ascending=False).round(3))

    print("\n=== Bad-risk rate by credit history ===")
    print(df.assign(label=df["credit_history"].map(HISTORY_MAP)).groupby("label")["is_bad_risk"].mean().sort_values(ascending=False).round(3))

    print("\n=== Bad-risk rate by telephone registration (an alternative-data signal) ===")
    print(df.groupby("telephone")["is_bad_risk"].mean().round(3))

    print("\n=== Duration and credit amount by risk ===")
    print(df.groupby("is_bad_risk")[["duration", "credit_amount", "age"]].mean().round(1))

    print(f"\n=== Data quality ===")
    print(f"Missing values: {df.isnull().sum().sum()}  |  Duplicates: {df.duplicated().sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    checking_rates.plot(kind="barh", ax=axes[0], color="#4C72B0", title="Bad-risk rate by checking-account status")

    df.assign(label=df["savings_status"].map(SAVINGS_MAP)).groupby("label")["is_bad_risk"].mean().sort_values(ascending=False).plot(
        kind="barh", ax=axes[1], color="#55A868", title="Bad-risk rate by savings status"
    )

    df.boxplot(column="duration", by="is_bad_risk", ax=axes[2])
    axes[2].set_title("Loan duration by risk outcome")
    axes[2].set_xlabel("is_bad_risk")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")

if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
