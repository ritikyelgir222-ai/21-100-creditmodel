"""
Phase 7: Train/Validation/Test Split
---------------------------------------
WHY STRATIFIED (not time-based): this dataset is a single historical
snapshot of 1,000 applicants with no application-date column -- there is
no time axis to split on, unlike a churn or demand-forecasting project
where leaking future information into training is the main risk here.
The risk in THIS dataset is class imbalance: only 30% of applicants are
bad-risk. A plain random split could easily over- or under-represent bad
cases in one split by chance with only 300 positive examples total to
work with. Stratifying on is_bad_risk keeps the 70/30 ratio consistent
across train, validation, and test, which matters a lot here because
the whole business metric (Phase 9) is cost-weighted toward correctly
catching the minority bad-risk class -- an unlucky split could make a
genuinely fine model look bad, or a weak model look fine, by chance alone.

SPLIT SIZES: 60% train / 20% validation / 20% test. With only 1,000 rows
total (300 bad-risk), a larger test carve-out would leave too few
positive examples in train to learn the pattern; a smaller one would make
the final test metric too noisy to trust. 60/20/20 is a reasonable
balance at this sample size.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns

RANDOM_STATE = 42

def make_splits(engineered_df: pd.DataFrame):
    feature_cols = get_feature_columns(engineered_df)
    X = engineered_df[feature_cols]
    y = engineered_df["is_bad_risk"]

    # First carve off test (20%), stratified on the target.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # Then split remaining 80% into train (60% overall) / val (20% overall).
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(engineered)

    print(f"Train: {len(X_train)} rows  (bad-risk rate: {y_train.mean():.1%})")
    print(f"Val:   {len(X_val)} rows  (bad-risk rate: {y_val.mean():.1%})")
    print(f"Test:  {len(X_test)} rows  (bad-risk rate: {y_test.mean():.1%})")

    X_train.to_csv("outputs/X_train.csv", index=False)
    X_val.to_csv("outputs/X_val.csv", index=False)
    X_test.to_csv("outputs/X_test.csv", index=False)
    y_train.to_csv("outputs/y_train.csv", index=False)
    y_val.to_csv("outputs/y_val.csv", index=False)
    y_test.to_csv("outputs/y_test.csv", index=False)
    print("\nSaved splits -> outputs/*.csv")
