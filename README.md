# Day 21: Alternative Credit Scoring for the Underbanked (Fintech)

Real-data, SDLC-documented data science project — Day 21 of a 100-day series.

## What this is

A credit-risk model for a fintech lender serving underbanked applicants —
people with thin or no traditional credit history — built on the real
**UCI Statlog German Credit Data** dataset (1,000 historical loan
applicants, Hofmann 1994). The model predicts whether an applicant is a
good or bad credit risk using a mix of traditional and alternative
signals (checking/savings account status, employment duration, telephone
registration, housing, property), and is evaluated against the
dataset's own **documented 5:1 cost matrix** rather than a made-up
business assumption.

## Why every script is commented the way it is

- **`data_loader.py`** explains *why this specific dataset* fits an
  "alternative credit scoring" use case (it has a real underbanked proxy
  — checking-account status — and a real cost matrix), and flags the
  fair-lending issue up front so it's not a surprise later.
- **`eda.py`** is built around one real, counter-intuitive finding:
  applicants with **no checking account at all have the lowest bad-risk
  rate (11.7%)** of any checking-status group — the opposite of what
  "underbanked = riskier" would predict. The comments explain this is
  correlation observed in the data, not a causal claim.
- **`clean_and_engineer.py`** structurally **drops** `personal_status_sex`
  and `foreign_worker` — not just flags them — because they are
  prohibited bases for a US-style credit decision (ECOA). The comment
  explains why this is a *stronger* step than this series took on an
  earlier HR project, and why a lending use case demands it.
- **`train_model.py`** uses the dataset's own 5:1 cost ratio as the
  business metric (bad-risk-as-good costs 5x more than good-as-bad),
  tunes the decision threshold to minimize that real cost rather than
  using a default 0.5 cutoff, and selects the final model by comparing
  raw unrounded validation cost floats — never a hardcoded pick.
- **`app.py`** reuses `clean_and_engineer.py`'s exact transformation
  logic to avoid train/serve skew, and intentionally **does not accept**
  `personal_status_sex` or `foreign_worker` as API inputs at all.
- **`monitor.py`** uses PSI-based drift detection on the continuous
  features most likely to shift (duration, credit amount, age, monthly
  payment burden) and a recall-drop retrain trigger tied to the costly
  error type (missing bad-risk applicants).

## Setup & run

```bash
pip install -r requirements.txt

python3 data_loader.py          # verify data loads, check quality
python3 eda.py                  # EDA + outputs/eda_summary.png
python3 clean_and_engineer.py   # cleaning + feature engineering
python3 split.py                # stratified train/val/test split
python3 train_model.py          # train, select model, evaluate, save artifacts
python3 monitor.py              # drift check demonstration

uvicorn app:app --reload        # serve the scoring API
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "checking_status": "A11", "duration": 24, "credit_history": "A32", "purpose": "A43",
  "credit_amount": 3500, "savings_status": "A61", "employment_duration": "A73",
  "installment_rate": 3, "other_debtors": "A101", "present_residence_since": 2,
  "property": "A121", "age": 30, "other_installment_plans": "A143", "housing": "A152",
  "existing_credits": 1, "job": "A173", "people_liable": 1, "telephone": "A191"
}'
```

## Results (real, reproducible numbers)

- Logistic Regression selected over Random Forest and XGBoost on
  validation business cost (95 vs. 108 vs. 103 cost units)
- Test set: AUC 0.783, recall on bad-risk class 0.883, business cost 130
- **56.7% cost reduction** vs. an "approve everyone" baseline (300 → 130
  cost units) on the test set

## Known limitations

- 1,000-row dataset from a 1990s-era German lending market — not
  representative of any current market; illustrative only.
- Excluding `personal_status_sex` and `foreign_worker` removes the two
  most obviously prohibited signals, but doesn't guarantee the remaining
  features carry zero residual correlation with protected attributes.
  A real deployment needs a formal disparate-impact audit before go-live.
- `monitor.py`'s drift demonstration compares the training data against
  a resample of itself, since there's no live production feed — a real
  deployment would feed it actual recent scoring logs.

---
**Hashtags:** #DataScience #MachineLearning #Fintech #CreditScoring #FairLending #100DaysOfProjects
