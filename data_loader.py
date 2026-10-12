"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the Statlog (German Credit Data) dataset, donated by
Prof. Hans Hofmann (University of Hamburg) to the UCI Machine Learning
Repository in 1994 -- one of the most widely used real credit-risk
datasets in applied ML:
    https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

1,000 real historical loan applicants, 20 attributes each, labeled good
(700) or bad (300) credit risk based on actual repayment outcome.

If you're following along fresh: the file is `german.data` (the coded
attribute version), space-separated, no header -- already included at
`data/german.data`. Column names and category decodings are applied in
code (see clean_and_engineer.py), following the dataset's own published
attribute documentation.

WHY THIS DATASET FITS "ALTERNATIVE CREDIT SCORING FOR THE UNDERBANKED"
-------------------------------------------------------------------------
- It includes genuine non-traditional signals beyond a formal bureau
  score: whether the applicant even HAS a checking account (a real,
  direct proxy for "underbanked"), telephone registration, employment
  duration, housing status, and existing property -- exactly the kind
  of soft/alternative data an underbanked-focused lender would lean on
  when a traditional credit history is thin or absent.
- It ships with a REAL, dataset-provided cost matrix: misclassifying a
  bad-risk applicant as good costs 5x more than misclassifying a
  good-risk applicant as bad. This project uses that exact ratio as its
  business metric (see train_model.py), rather than inventing one --
  a genuinely stronger grounding than most of this series' business
  validations, which had to construct a capacity or tolerance threshold
  from scratch.

IMPORTANT FAIRNESS/COMPLIANCE NOTE: this dataset's `personal_status_sex`
column bundles sex and marital status into a single categorical field
-- a well-documented critique of this dataset. In most real jurisdictions
(e.g. the US Equal Credit Opportunity Act), sex and marital status are
PROHIBITED bases for a credit decision. This project excludes that
column, and `foreign_worker` (national-origin-adjacent), from the model
entirely -- not just flagged, but structurally removed -- see
clean_and_engineer.py for where and why.
"""

import pandas as pd

COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "target",
]
RAW_DATA_PATH = "data/german.data"

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"\nTarget distribution (1=good, 2=bad): {df['target'].value_counts().to_dict()}")
    print(f"Bad-risk rate: {(df['target'] == 2).mean():.1%}")
    print(f"\nMissing values: {df.isnull().sum().sum()}  |  Duplicate rows: {df.duplicated().sum()}")
