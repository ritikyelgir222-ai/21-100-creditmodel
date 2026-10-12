"""
Phase 10: MLOps & Deployment
---------------------------------
Reuses clean_and_engineer.py's exact transformation logic so the API
never drifts from the training-time feature pipeline (train/serve
skew is one of the most common real-world causes of a model quietly
degrading after deployment).

The selected model turned out to be Logistic Regression (see
train_model.py / model_card.json), so this service also loads the
fitted StandardScaler and applies it before scoring -- unlike the
Telco reference project, where the final model was a tree ensemble
that didn't need scaling. Loading model_type.joblib makes this
explicit rather than hardcoding an assumption about which model won.
"""

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Alternative Credit Scoring API", version="1.0")

MODEL = joblib.load("credit_model.joblib")
SCALER = joblib.load("scaler.joblib")
FEATURE_COLUMNS = joblib.load("feature_columns.joblib")
MODEL_TYPE = joblib.load("model_type.joblib")

with open("outputs/business_validation.json") as f:
    BUSINESS_VALIDATION = json.load(f)
DECISION_THRESHOLD = BUSINESS_VALIDATION["decision_threshold"]


class LoanApplication(BaseModel):
    # Mirrors the raw coded dataset fields a loan-origination system would
    # send -- personal_status_sex and foreign_worker are intentionally NOT
    # accepted here: structurally excluding them at the API boundary means
    # this prohibited-basis data never even reaches the scoring pipeline,
    # not just the model (see clean_and_engineer.py for why).
    checking_status: str       # e.g. "A11", "A12", "A13", "A14"
    duration: int               # loan duration, months
    credit_history: str         # e.g. "A30".."A34"
    purpose: str                 # e.g. "A40".."A410"
    credit_amount: float
    savings_status: str          # e.g. "A61".."A65"
    employment_duration: str     # e.g. "A71".."A75"
    installment_rate: int
    other_debtors: str            # e.g. "A101".."A103"
    present_residence_since: int
    property: str                  # e.g. "A121".."A124"
    age: int
    other_installment_plans: str   # e.g. "A141".."A143"
    housing: str                    # e.g. "A151".."A153"
    existing_credits: int
    job: str                         # e.g. "A171".."A174"
    people_liable: int
    telephone: str                   # "A191" or "A192"


def tier_from_score(bad_risk_prob: float) -> str:
    # WHY these cutoffs: the model's own decision_threshold (tuned to
    # minimize the dataset's 5:1 cost matrix) is the line between "approve"
    # and "decline". A middle "Manual Review" band around that threshold
    # gives a loan officer a zone to apply judgment on borderline cases,
    # rather than a single hard cutoff -- consistent with this series'
    # established tiering pattern (e.g. the Telco churn project's
    # High/Medium/Low tiers).
    if bad_risk_prob >= DECISION_THRESHOLD + 0.15:
        return "Decline"
    elif bad_risk_prob >= DECISION_THRESHOLD - 0.10:
        return "Manual Review"
    else:
        return "Approve"


@app.get("/")
def root():
    return {
        "service": "Alternative Credit Scoring API",
        "model": MODEL_TYPE,
        "decision_threshold": DECISION_THRESHOLD,
        "fairness_note": BUSINESS_VALIDATION["fairness_note"],
    }


@app.post("/score")
def score(application: LoanApplication):
    raw_row = application.model_dump()
    # The fitted pipeline expects the two excluded fair-lending columns to
    # exist (they're dropped inside clean_data), and the raw target column.
    # Since this is a live scoring request (no known outcome yet), we
    # supply placeholder values for the columns clean_data() drops anyway,
    # so the exact training-time code path runs unmodified.
    raw_row["personal_status_sex"] = "A91"   # placeholder; dropped immediately, never used
    raw_row["foreign_worker"] = "A201"        # placeholder; dropped immediately, never used
    raw_row["target"] = 1                      # placeholder; dropped immediately, never used

    df = pd.DataFrame([raw_row])
    cleaned = clean_data(df)
    engineered = engineer_features(cleaned)

    # Align to the exact training-time feature set/order -- any one-hot
    # category not seen in this single request (e.g. a purpose code this
    # applicant didn't have) must appear as a 0 column, and no extra
    # columns should leak through.
    engineered = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    if MODEL_TYPE == "logistic_regression":
        X = SCALER.transform(engineered)
    else:
        X = engineered

    bad_risk_prob = float(MODEL.predict_proba(X)[0, 1])
    tier = tier_from_score(bad_risk_prob)

    # Lightweight explainability stand-in: for the logistic model, the
    # coefficients are directly the feature importances. (If a different
    # model type had won, this would fall back to .feature_importances_,
    # matching the Telco reference project's approach.)
    if hasattr(MODEL, "coef_"):
        importances = dict(zip(FEATURE_COLUMNS, MODEL.coef_[0]))
    else:
        importances = dict(zip(FEATURE_COLUMNS, MODEL.feature_importances_))
    top_factors = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

    return {
        "bad_risk_probability": round(bad_risk_prob, 4),
        "decision_threshold": DECISION_THRESHOLD,
        "tier": tier,
        "top_model_factors": [{"feature": f, "weight": round(w, 4)} for f, w in top_factors],
    }
