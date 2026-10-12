# Project Documentation: Alternative Credit Scoring for the Underbanked

## 1. Project Summary

**Client context:** a fintech lender targeting underbanked applicants —
people whose thin or absent traditional credit history makes them
invisible to conventional bureau-score-based underwriting. The lender
needs to approve viable applicants from this population without taking
on disproportionate default risk.

**Problem statement:** build a credit-risk classifier that can use
alternative signals (account status, employment duration, telephone
registration) alongside traditional ones, scored against a real,
documented cost asymmetry rather than a generic accuracy target.

**Business objective:** reduce the dollar cost of misclassification —
specifically, minimize the dataset's own 5:1 cost ratio (approving a
bad-risk applicant costs 5x more than declining a good-risk one) —
rather than optimizing a metric disconnected from lending economics.

## 2. Architecture

```
german.data (raw, coded)
      |
      v
data_loader.py  --------->  raw DataFrame (21 cols, coded categories)
      |
      v
clean_and_engineer.py
   clean_data()       -> drops personal_status_sex, foreign_worker (fair lending);
                          recodes target -> is_bad_risk
   engineer_features() -> no_checking_account, monthly_payment_burden,
                          one-hot encodes 11 categorical columns
      |
      v
split.py  --------->  stratified 60/20/20 train/val/test (on is_bad_risk)
      |
      v
train_model.py
   LogisticRegression / RandomForest / XGBoost
   -> threshold tuned to minimize 5:1 business cost on validation
   -> model selected by lowest validation business cost
   -> credit_model.joblib, scaler.joblib, feature_columns.joblib,
      model_type.joblib, business_validation.json, model_card.json
      |
      v
app.py (FastAPI)  --------->  POST /score  -> bad_risk_probability, tier, top factors
      |
      v
monitor.py  --------->  PSI drift check + recall-drop retrain trigger
```

## 3. Data Dictionary

| Column | Type | Description |
|---|---|---|
| checking_status | categorical | Checking account status (A11 `<0 DM` … A14 `no checking account`) |
| duration | numeric | Loan duration, months |
| credit_history | categorical | Repayment history on prior credits |
| purpose | categorical | Loan purpose (car, furniture, education, etc.) |
| credit_amount | numeric | Requested loan amount (DM) |
| savings_status | categorical | Savings account balance tier |
| employment_duration | categorical | Years at current employment |
| installment_rate | numeric | Installment as % of disposable income |
| ~~personal_status_sex~~ | categorical | **Excluded** — bundles sex + marital status (ECOA prohibited basis) |
| other_debtors | categorical | Co-applicants / guarantors |
| present_residence_since | numeric | Years at current residence |
| property | categorical | Property ownership type |
| age | numeric | Applicant age |
| other_installment_plans | categorical | Other installment commitments |
| housing | categorical | Rent / own / free housing |
| existing_credits | numeric | Number of existing credits at this bank |
| job | categorical | Job/skill category |
| people_liable | numeric | Number of dependents |
| telephone | categorical | Telephone registration (alternative-data signal) |
| ~~foreign_worker~~ | categorical | **Excluded** — national-origin-adjacent |
| target → is_bad_risk | binary | 1 = bad credit risk (defaulted), 0 = good |

Data quality: 0 missing values, 0 duplicates (1,000 rows total).

## 4. Key EDA Findings

- Overall bad-risk rate: 30.0%
- **Checking-account status** (the core underbanked signal): bad-risk
  rate ranges from 49.3% (`<0 DM`) down to **11.7% for applicants with
  no checking account at all** — the lowest of any group. This is
  counter-intuitive and reported as an observed association, not a
  causal claim (a plausible explanation: applicants with other
  substantial assets may simply not need a checking account).
- **Savings status:** bad-risk rate falls from 36.0% (`<100 DM`) to
  12.5% (`>=1000 DM`) — a much more intuitive monotonic pattern.
- **Employment duration:** bad-risk rate falls from 40.7% (`<1 year`) to
  22.4-25.3% for applicants employed 4+ years.
- **Credit history:** counter-intuitively, "no credits/all paid" has the
  *highest* bad-risk rate (62.5%) — likely because very few prior
  credits means less track record to vouch for the applicant, while
  "critical/other credits" has the lowest (17.1%) — these applicants
  already proved they can manage larger credit lines.
- **Telephone registration:** a much weaker signal (31.4% vs. 28.0% bad-
  risk rate) than the account-status signals.
- Bad-risk applicants have longer loan durations (24.9 vs. 19.2 months)
  and larger credit amounts (3,938 vs. 2,986 DM) on average.

## 5. Modeling Results

| Model | Validation AUC | Validation Business Cost | Threshold |
|---|---|---|---|
| Logistic Regression | 0.716 | **95** (selected) | 0.20 |
| Random Forest | 0.737 | 108 | 0.28 |
| XGBoost | 0.723 | 103 | 0.09 |

Logistic Regression had the lowest AUC of the three but the lowest
**business cost** — the metric that actually matters here — so it was
selected. This is a deliberate illustration of why a business-aligned
metric can pick a different "winner" than a generic technical one.

**Test set (selected model, threshold carried over from validation, not re-tuned):**
AUC 0.783, precision 0.358, recall 0.883, business cost 130.

**Business validation:** an "approve everyone" baseline would cost 300
units on the test set (60 bad-risk applicants × 5 cost units each). The
model's cost of 130 units is a **56.7% reduction** — the concrete,
dataset-grounded business impact of this project.

## 6. API Reference

`POST /score` — accepts a `LoanApplication` (raw coded fields, excluding
`personal_status_sex` and `foreign_worker` by design) and returns:

```json
{
  "bad_risk_probability": 0.6376,
  "decision_threshold": 0.2,
  "tier": "Decline",
  "top_model_factors": [
    {"feature": "installment_rate", "weight": 0.4932},
    {"feature": "duration", "weight": 0.4311},
    ...
  ]
}
```

Tiers: `Approve` (well below threshold), `Manual Review` (near
threshold), `Decline` (well above threshold) — giving a loan officer a
judgment zone around the model's cost-optimal cutoff rather than a
single hard line.

`GET /` — service info, including the fairness note on excluded fields.

## 7. Monitoring Plan

- **Drift:** PSI on `duration`, `credit_amount`, `age`,
  `monthly_payment_burden` against the training distribution; PSI ≥ 0.25
  on any feature is a significant-shift flag.
- **Retrain trigger:** recall on the bad-risk class (the costly error
  type) dropping more than 0.05 below the recorded test baseline
  (0.883), measured against recently matured loan outcomes in a real
  deployment.

## 8. Known Limitations

- Small (1,000-row), decades-old dataset from a different market and
  currency — illustrative of method, not a production-ready model.
- Structural exclusion of `personal_status_sex` and `foreign_worker`
  addresses the two most obvious prohibited-basis columns in this
  dataset, but does not itself constitute a full fair-lending audit;
  other features may carry residual correlation with protected classes.
- No live production feed exists for this demo, so `monitor.py`
  illustrates its logic against a resample of the training data rather
  than real drift.

## 9. File Map

```
day21/
├── data/german.data              # raw UCI Statlog German Credit data
├── data_loader.py                # Phase 5: load + quality check
├── eda.py                        # Phase 6: EDA + eda_summary.png
├── clean_and_engineer.py         # Phase 5 fixes + Phase 7: features
├── split.py                      # Phase 7: stratified train/val/test split
├── train_model.py                # Phase 8-9: train, select, validate
├── app.py                        # Phase 10: FastAPI scoring service
├── monitor.py                    # Phase 13: PSI drift + retrain trigger
├── outputs/                      # eda_summary.png, engineered_data.csv,
│                                  #   splits, business_validation.json, model_card.json
├── credit_model.joblib / scaler.joblib / feature_columns.joblib / model_type.joblib
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md
```
