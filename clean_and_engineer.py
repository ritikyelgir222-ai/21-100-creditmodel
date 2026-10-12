"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it. FINAL FEATURE LIST is built at the bottom as
FEATURE_COLUMNS.
"""

import pandas as pd

# -----------------------------------------------------------------------
# Columns EXCLUDED from the model entirely for fair-lending compliance,
# not just flagged for review (a stronger step than Day 4's attrition
# project, which kept Gender but flagged it for a fairness review before
# deployment). `personal_status_sex` bundles sex and marital status --
# both prohibited bases for a credit decision under regulations like the
# US Equal Credit Opportunity Act. `foreign_worker` is national-origin-
# adjacent and carries the same risk. Structurally removing these before
# they ever reach the model is the correct move for a lending use case,
# where using a prohibited basis isn't just an ethical concern but a
# legal one.
# -----------------------------------------------------------------------
EXCLUDED_FOR_FAIR_LENDING = ["personal_status_sex", "foreign_worker"]

ONE_HOT_COLUMNS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment_duration", "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone",
]

def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # This dataset has zero missing values and zero duplicates (verified
    # in data_loader.py) -- Phase 5's real work here is the fair-lending
    # exclusion above, not imputation.
    # -----------------------------------------------------------------
    df = df.drop(columns=EXCLUDED_FOR_FAIR_LENDING)

    # target: 1=good, 2=bad -> is_bad_risk: 0=good, 1=bad (standard
    # "1 = the event we're trying to catch" convention used throughout
    # this series)
    df["is_bad_risk"] = (df["target"] == 2).astype(int)
    df = df.drop(columns=["target"])

    return df

def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Engineered feature: no_checking_account
    # BUSINESS LOGIC: EDA found applicants with NO checking account have
    # the LOWEST bad-risk rate (11.7%) of any checking-status group --
    # the opposite of what "underbanked = higher risk" would predict.
    # An explicit flag lets the model use this directly rather than
    # relying on the one-hot checking_status category alone to carry it,
    # and makes the (counter-intuitive) pattern easy to point to when
    # explaining a score to a loan officer.
    # -----------------------------------------------------------------
    df["no_checking_account"] = (df["checking_status"] == "A14").astype(int)

    # -----------------------------------------------------------------
    # Engineered feature: monthly_payment_burden
    # BUSINESS LOGIC: credit_amount and duration interact -- a $3,000
    # loan over 12 months is a very different monthly obligation than
    # the same amount over 48 months. This reconstructs an approximate
    # monthly payment size, a more direct affordability signal than
    # either raw column alone (ignoring interest for simplicity, which
    # this dataset doesn't provide per-applicant).
    # -----------------------------------------------------------------
    df["monthly_payment_burden"] = df["credit_amount"] / df["duration"]

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    dummy_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df

NUMERIC_MODEL_FEATURES = [
    "duration", "credit_amount", "installment_rate", "present_residence_since",
    "age", "existing_credits", "people_liable", "no_checking_account", "monthly_payment_burden",
]

def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    return NUMERIC_MODEL_FEATURES + dummy_cols

if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print(f"(Excluded for fair-lending compliance: {EXCLUDED_FOR_FAIR_LENDING})")
    print(f"\nno_checking_account rate: {engineered['no_checking_account'].mean():.1%}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
