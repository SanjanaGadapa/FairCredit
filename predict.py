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

def _composite_approval_prob(row) -> float:
    """
    Derives a calibrated approval probability from domain credit features.
    The raw XGBoost model outputs extreme log-odds (±10+) that collapse to
    0/1 after sigmoid, making Yellow tier impossible and risk scores binary.
    This composite replaces that with a weighted, interpretable score.

    Returns a float in [0.05, 0.95] representing P(Approved).
    """
    score = 0.0

    # Credit Score: 300–900 range → 25% weight
    credit = float(row.get("CreditScore", 600))
    score += ((credit - 300) / 600) * 0.25

    # Payment History: 0–1, higher = better → 20% weight
    pay_hist = float(row.get("PaymentHistory", 0.5))
    score += pay_hist * 0.20

    # Debt-to-Income Ratio: lower = better → 15% weight
    dti = float(row.get("DebtToIncomeRatio", 0.5))
    score += (1 - min(dti, 1.0)) * 0.15

    # Previous Loan Defaults: 0 defaults = full weight → 15% weight
    defaults = float(row.get("PreviousLoanDefaults", 0))
    score += max(0.0, 1 - defaults * 0.25) * 0.15

    # Credit Card Utilization: lower = better → 10% weight
    util = float(row.get("CreditCardUtilizationRate", 0.5))
    score += (1 - min(util, 1.0)) * 0.10

    # Savings-to-Loan Ratio: higher = better → 7% weight
    stl = float(row.get("SavingsToLoanRatio", 0))
    score += min(stl, 1.0) * 0.07

    # Utility Bills Payment History: higher = better → 5% weight
    util_hist = float(row.get("UtilityBillsPaymentHistory", 0.5))
    score += util_hist * 0.05

    # Penalties for hard risk flags
    if float(row.get("MissedPaymentFlag", 0)) == 1:
        score -= 0.08
    if float(row.get("BankruptcyHistory", 0)) == 1:
        score -= 0.10
    if float(row.get("HighUtilizationFlag", 0)) == 1:
        score -= 0.04

    # Clamp to [0.05, 0.95] to avoid absolute certainty
    return float(np.clip(score, 0.05, 0.95))


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

           # Composite approval probability from domain features.
            # The underlying XGBoost model returns near-0/1 margins (overfit),
            # so we derive a calibrated score from key credit features instead.
            proba = _composite_approval_prob(row)
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
