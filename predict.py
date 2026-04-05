import pandas as pd
import numpy as np
import pickle
import joblib          # FIX 1: use joblib, not pickle, for scaler
import json

# ── Load threshold ─────────────────────────────────────────────
with open("models/threshold.json") as f:
    guard_rules = json.load(f)

DECISION_THRESHOLD = guard_rules["decision_threshold"]

# ── Load fair model ────────────────────────────────────────────
with open("models/xgboost_fair.pkl", "rb") as f:
    fair_model = pickle.load(f)

# ── Load scaler (must use joblib — saved with joblib) ──────────
# FIX 1: was pickle.load() — crashes with _pickle.UnpicklingError
_default_scaler = joblib.load("models/scaler.pkl")

# ── Model-ready columns ────────────────────────────────────────
MODEL_COLUMNS = pd.read_csv("data/processed/X_train.csv").columns.tolist()


def predict_with_guard(raw_input_df, scaler=None, decision_threshold=DECISION_THRESHOLD):
    """
    Wraps the fair model with hard business-logic rules.

    Parameters:
        raw_input_df       : pd.DataFrame — applicant row(s), RAW unscaled values.
                             Must contain guard columns: Age, MonthlyIncome,
                             EMIBurdenRatio, SavingsToLoanRatio,
                             CreditCardUtilizationRate, PreviousLoanDefaults
        scaler             : fitted StandardScaler (optional).
                             If None, uses the default scaler loaded from models/scaler.pkl.
        decision_threshold : float — loaded from threshold.json

    Returns:
        dict — keys: prediction, approved, flag, reason, risk_score
    """
    # FIX 2: always fall back to default scaler so caller never has to pass it
    if scaler is None:
        scaler = _default_scaler

    results = []

    for i, row in raw_input_df.iterrows():
        flag = False
        reason = None

        # ── Rule 1: Age out of valid range ────────────────────
        if "Age" in row.index and (row["Age"] < 18 or row["Age"] > 85):
            flag = True
            reason = (f"Age {int(row['Age'])} is outside the valid applicant "
                      f"range (18–85). Cannot make a reliable prediction.")

        # ── Rule 2: Zero or negative income ───────────────────
        elif "MonthlyIncome" in row.index and row["MonthlyIncome"] <= 0:
            flag = True
            reason = "Monthly income must be greater than 0. Cannot compute EMI burden ratio."

        # ── Rule 3: EMI burden exceeds 100% ───────────────────
        elif "EMIBurdenRatio" in row.index and row["EMIBurdenRatio"] > 1.0:
            flag = True
            reason = (f"EMI burden ratio of {row['EMIBurdenRatio']:.2f} means "
                      f"monthly debt exceeds income. Auto-rejected on business rules.")

        # ── Rule 4: SavingsToLoanRatio is undefined ───────────
        elif "SavingsToLoanRatio" in row.index and (
            pd.isna(row["SavingsToLoanRatio"]) or np.isinf(row["SavingsToLoanRatio"])
        ):
            flag = True
            reason = ("Savings-to-loan ratio is undefined (division by zero). "
                      "Check LoanAmount and SavingsAccountBalance inputs.")

        # ── Rule 5: Credit utilization above 100% ─────────────
        elif "CreditCardUtilizationRate" in row.index and row["CreditCardUtilizationRate"] > 1.0:
            flag = True
            reason = (f"Credit utilization of {row['CreditCardUtilizationRate']:.2f} "
                      f"exceeds 100%, which is not physically possible.")

        # ── Rule 6: Excessive prior defaults ──────────────────
        elif "PreviousLoanDefaults" in row.index and row["PreviousLoanDefaults"] >= 10:
            flag = True
            reason = (f"{int(row['PreviousLoanDefaults'])} previous loan defaults. "
                      f"Automatic rejection on business rules (threshold: 10+).")

        if flag:
            results.append({
                "prediction": 0,
                "approved":   False,
                "flag":       True,
                "reason":     reason,
                "risk_score": None,
            })
        else:
            row_df = pd.DataFrame([row])

            cols_present = [c for c in MODEL_COLUMNS if c in row_df.columns]
            cols_missing = [c for c in MODEL_COLUMNS if c not in row_df.columns]

            if cols_missing:
                results.append({
                    "prediction": 0,
                    "approved":   False,
                    "flag":       True,
                    "reason":     f"Missing model columns: {cols_missing}",
                    "risk_score": None,
                })
                continue

            # Keep only model columns, drop extras like EMIBurdenRatio
            row_df = row_df[MODEL_COLUMNS].astype(float)

            # FIX 3: scale the raw input (scaler is always available now)
            row_scaled = pd.DataFrame(
                scaler.transform(row_df),
                columns=MODEL_COLUMNS
            )

            # FIX 4: convert to numpy before _pmf_predict to avoid DataFrame issues
            # across different fairlearn versions
            proba = fair_model._pmf_predict(row_scaled.values)[:, 1][0]
            pred  = int(proba >= decision_threshold)

            results.append({
                "prediction": pred,
                "approved":   bool(pred == 1),
                "flag":       False,
                "reason":     None,
                "risk_score": round(float(proba), 4),
            })

    # Return single dict if single row (what app.py expects)
    return results[0] if len(results) == 1 else results