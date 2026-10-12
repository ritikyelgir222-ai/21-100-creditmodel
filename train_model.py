"""
Phases 8-9: Model Development & Business Validation
--------------------------------------------------------
BUSINESS METRIC: this dataset ships with its own documented cost matrix
(Statlog German Credit, UCI) -- classifying a BAD-risk applicant as GOOD
(approving a loan that defaults) costs 5x more than classifying a GOOD
applicant as BAD (rejecting a safe applicant). This project uses that
real, dataset-provided ratio directly as its business metric, rather
than inventing a capacity/tolerance assumption from scratch the way some
earlier days in this series had to.

total_cost = 5 * false_negatives (bad predicted good) + 1 * false_positives (good predicted bad)

Lower is better. This is reported alongside (not instead of) standard
classification metrics, since a lending stakeholder ultimately cares
about total expected loss, not AUC.

MODEL SELECTION: baseline Logistic Regression (explainable, the one
you'd actually hand to a compliance team) -> Random Forest -> XGBoost,
with final selection by cost on the VALIDATION set, not accuracy or AUC.
Selection compares raw unrounded floats programmatically (never a
hardcoded "model X wins" from eyeballing rounded output -- a bug this
series already hit and fixed on Day 16).

CLASSIFICATION THRESHOLD: because false negatives cost 5x as much as
false positives, the default 0.5 probability threshold is wrong for this
business problem -- it treats both error types as equally costly. The
threshold is instead tuned on the validation set to directly minimize
the 5:1-weighted cost, which naturally pushes the cutoff lower (the
model should flag an applicant as "bad risk" more readily than a
generic classifier would).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import make_splits

FN_COST = 5  # bad risk predicted as good (approved a defaulting loan)
FP_COST = 1  # good risk predicted as bad (rejected a safe applicant)


def business_cost(y_true, y_pred) -> int:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return FN_COST * fn + FP_COST * fp


def best_threshold_by_cost(y_true, probs) -> tuple:
    """Scan thresholds and return the one minimizing business_cost."""
    candidates = np.linspace(0.05, 0.95, 181)
    costs = [business_cost(y_true, (probs >= t).astype(int)) for t in candidates]
    best_idx = int(np.argmin(costs))
    return float(candidates[best_idx]), int(costs[best_idx])


def evaluate(name, y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    cost = business_cost(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"{name:22s}  AUC={auc:.4f}  Precision={precision:.3f}  Recall={recall:.3f}  "
          f"Threshold={threshold:.2f}  BusinessCost={cost}")
    return {"name": name, "auc": auc, "precision": precision, "recall": recall,
            "threshold": threshold, "business_cost": cost}


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(engineered)

    # ---- Baseline: Logistic Regression (scaled, class-balanced) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    val_probs_lr = log_reg.predict_proba(X_val_scaled)[:, 1]
    thresh_lr, _ = best_threshold_by_cost(y_val, val_probs_lr)
    result_lr = evaluate("LogisticRegression", y_val, val_probs_lr, thresh_lr)

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    val_probs_rf = rf.predict_proba(X_val)[:, 1]
    thresh_rf, _ = best_threshold_by_cost(y_val, val_probs_rf)
    result_rf = evaluate("RandomForest", y_val, val_probs_rf, thresh_rf)

    # ---- XGBoost ----
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    val_probs_xgb = xgb.predict_proba(X_val)[:, 1]
    thresh_xgb, _ = best_threshold_by_cost(y_val, val_probs_xgb)
    result_xgb = evaluate("XGBoost", y_val, val_probs_xgb, thresh_xgb)

    # ---- Programmatic model selection on raw unrounded validation cost ----
    # (comparing result["business_cost"] floats directly, never rounded
    # display strings -- the Day 16 lesson carried forward in this series)
    candidates = [
        ("logistic_regression", log_reg, result_lr, None),
        ("random_forest", rf, result_rf, None),
        ("xgboost", xgb, result_xgb, None),
    ]
    best_name, best_model, best_result, _ = min(candidates, key=lambda c: c[2]["business_cost"])
    print(f"\nSelected model (lowest validation business cost): {best_name}  "
          f"(cost={best_result['business_cost']})")

    # Honest reporting of close calls, matching this series' established practice.
    costs_sorted = sorted(candidates, key=lambda c: c[2]["business_cost"])
    if costs_sorted[1][2]["business_cost"] - costs_sorted[0][2]["business_cost"] <= 2:
        print(f"NOTE: {costs_sorted[0][0]} and {costs_sorted[1][0]} were a near-tie on "
              f"validation business cost ({costs_sorted[0][2]['business_cost']} vs "
              f"{costs_sorted[1][2]['business_cost']}) -- reported honestly rather than "
              f"overstating the margin.")

    # ---- Final evaluation on untouched TEST set, using the selected model's
    # validation-tuned threshold (not re-tuned on test -- that would leak) ----
    if best_name == "logistic_regression":
        test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        test_probs = best_model.predict_proba(X_test)[:, 1]
    test_result = evaluate(f"{best_name} (TEST)", y_test, test_probs, best_result["threshold"])

    # ---- Business validation: translate the cost-matrix outcome into plain terms ----
    naive_cost = business_cost(y_test, np.zeros(len(y_test)))  # "approve everyone" baseline
    improvement_pct = (naive_cost - test_result["business_cost"]) / naive_cost * 100
    print(f"\n=== Business validation ===")
    print(f"'Approve everyone' baseline cost on test set: {naive_cost} "
          f"({naive_cost / FN_COST:.0f} bad-risk approvals x {FN_COST} cost units each)")
    print(f"Model's cost on test set: {test_result['business_cost']}")
    print(f"Cost reduction vs. approving everyone: {improvement_pct:.1f}%")

    business_validation = {
        "cost_matrix": {"false_negative_cost": FN_COST, "false_positive_cost": FP_COST,
                         "source": "UCI Statlog German Credit dataset documentation"},
        "selected_model": best_name,
        "validation_business_cost": best_result["business_cost"],
        "test_business_cost": test_result["business_cost"],
        "naive_approve_all_cost": int(naive_cost),
        "cost_reduction_pct_vs_naive": round(improvement_pct, 1),
        "decision_threshold": best_result["threshold"],
        "test_auc": test_result["auc"],
        "test_precision": test_result["precision"],
        "test_recall": test_result["recall"],
        "fairness_note": (
            "personal_status_sex and foreign_worker were structurally excluded from "
            "all features prior to modeling, for fair-lending compliance (see "
            "clean_and_engineer.py)."
        ),
    }
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_validation, f, indent=2)
    print("\nSaved -> outputs/business_validation.json")

    # ---- Persist model artifacts for serving ----
    joblib.dump(best_model, "credit_model.joblib")
    joblib.dump(scaler, "scaler.joblib")  # only used if best_model is logistic_regression
    joblib.dump(get_feature_columns(engineered), "feature_columns.joblib")
    joblib.dump(best_name, "model_type.joblib")

    model_card = {
        "model_name": best_name,
        "task": "Alternative credit scoring (binary bad-risk classification)",
        "dataset": "UCI Statlog German Credit Data (1,000 applicants)",
        "excluded_features_fair_lending": ["personal_status_sex", "foreign_worker"],
        "business_metric": "5:1 cost-weighted misclassification cost (dataset-documented)",
        "decision_threshold": best_result["threshold"],
        "test_set_performance": {
            "auc": round(test_result["auc"], 4),
            "precision": round(test_result["precision"], 3),
            "recall": round(test_result["recall"], 3),
            "business_cost": test_result["business_cost"],
        },
        "known_limitations": [
            "1,000-row dataset from 1990s-era German lending market -- amounts, norms, and "
            "applicant pool do not reflect any current market and should not be used for "
            "real lending decisions.",
            "personal_status_sex and foreign_worker excluded from features, but other "
            "features (e.g. property, housing) may carry residual correlation with "
            "protected attributes -- a real deployment would need a formal disparate-impact "
            "audit before go-live, not just feature exclusion.",
        ],
    }
    with open("outputs/model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    print("Saved -> outputs/model_card.json")
    print("Saved -> credit_model.joblib, scaler.joblib, feature_columns.joblib, model_type.joblib")
