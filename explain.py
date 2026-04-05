"""
explain.py — FairCredit 2.0 | Person 3: SHAP Explainability Module
===================================================================
Provides SHAP-based explanations for loan decisions made by xgboost_fair.pkl.
Uses xgboost_final.pkl (base model) for SHAP since ExponentiatedGradient
(fairness wrapper) is a randomised ensemble that TreeExplainer cannot handle.

Person 4 (Streamlit): import and call explain() only.
Do NOT call predict — that is predict_with_guard() in predict.py (Person 2).
"""

import json
import warnings
import pickle
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")           # headless — no display needed in Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_MODEL_PATH      = _BASE / "models" / "xgboost_final.pkl"
_SCALER_PATH     = _BASE / "models" / "scaler.pkl"
_THRESHOLD_PATH  = _BASE / "models" / "threshold.json"

# ─── Feature order (must match X_train column order exactly) ──────────────────
FEATURE_NAMES = [
    "Age", "AnnualIncome", "CreditScore", "Experience", "LoanAmount",
    "LoanDuration", "NumberOfDependents", "MonthlyDebtPayments",
    "CreditCardUtilizationRate", "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries", "DebtToIncomeRatio", "BankruptcyHistory",
    "PreviousLoanDefaults", "PaymentHistory", "LengthOfCreditHistory",
    "SavingsAccountBalance", "CheckingAccountBalance", "TotalAssets",
    "TotalLiabilities", "MonthlyIncome", "UtilityBillsPaymentHistory",
    "JobTenure", "NetWorth", "BaseInterestRate", "InterestRate",
    "MonthlyLoanPayment", "TotalDebtToIncomeRatio", "MissedPaymentFlag",
    "HighUtilizationFlag", "SavingsToLoanRatio",
    "EmploymentStatus_Self-Employed", "EmploymentStatus_Unemployed",
    "EducationLevel_Bachelor", "EducationLevel_Doctorate",
    "EducationLevel_High School", "EducationLevel_Master",
    "MaritalStatus_Married", "MaritalStatus_Single", "MaritalStatus_Widowed",
    "HomeOwnershipStatus_Other", "HomeOwnershipStatus_Own",
    "HomeOwnershipStatus_Rent", "LoanPurpose_Debt Consolidation",
    "LoanPurpose_Education", "LoanPurpose_Home", "LoanPurpose_Other",
]

# Human-readable labels shown to bankers (no underscores, cleaner text)
FEATURE_LABELS = {
    "Age": "Age",
    "AnnualIncome": "Annual Income",
    "CreditScore": "Credit Score",
    "Experience": "Work Experience (yrs)",
    "LoanAmount": "Loan Amount",
    "LoanDuration": "Loan Duration (months)",
    "NumberOfDependents": "Number of Dependents",
    "MonthlyDebtPayments": "Monthly Debt Payments",
    "CreditCardUtilizationRate": "Credit Card Utilization",
    "NumberOfOpenCreditLines": "Open Credit Lines",
    "NumberOfCreditInquiries": "Credit Inquiries",
    "DebtToIncomeRatio": "Debt-to-Income Ratio",
    "BankruptcyHistory": "Bankruptcy History",
    "PreviousLoanDefaults": "Previous Loan Defaults",
    "PaymentHistory": "Payment History Score",
    "LengthOfCreditHistory": "Credit History Length (yrs)",
    "SavingsAccountBalance": "Savings Account Balance",
    "CheckingAccountBalance": "Checking Account Balance",
    "TotalAssets": "Total Assets",
    "TotalLiabilities": "Total Liabilities",
    "MonthlyIncome": "Monthly Income",
    "UtilityBillsPaymentHistory": "Utility Bills Payment",
    "JobTenure": "Job Tenure (yrs)",
    "NetWorth": "Net Worth",
    "BaseInterestRate": "Base Interest Rate",
    "InterestRate": "Applied Interest Rate",
    "MonthlyLoanPayment": "Monthly Loan Payment",
    "TotalDebtToIncomeRatio": "Total Debt-to-Income Ratio",
    "MissedPaymentFlag": "Missed Payment (Flag)",
    "HighUtilizationFlag": "High Utilization (Flag)",
    "SavingsToLoanRatio": "Savings-to-Loan Ratio",
    "EmploymentStatus_Self-Employed": "Self-Employed",
    "EmploymentStatus_Unemployed": "Unemployed",
    "EducationLevel_Bachelor": "Education: Bachelor's",
    "EducationLevel_Doctorate": "Education: Doctorate",
    "EducationLevel_High School": "Education: High School",
    "EducationLevel_Master": "Education: Master's",
    "MaritalStatus_Married": "Married",
    "MaritalStatus_Single": "Single",
    "MaritalStatus_Widowed": "Widowed",
    "HomeOwnershipStatus_Other": "Home: Other",
    "HomeOwnershipStatus_Own": "Home: Owner",
    "HomeOwnershipStatus_Rent": "Home: Renter",
    "LoanPurpose_Debt Consolidation": "Purpose: Debt Consolidation",
    "LoanPurpose_Education": "Purpose: Education",
    "LoanPurpose_Home": "Purpose: Home",
    "LoanPurpose_Other": "Purpose: Other",
}

# ─── Lazy-loaded singletons (loaded once, reused across calls) ────────────────
_model     = None
_scaler    = None
_explainer = None
_threshold = None


def _load_artifacts():
    """Load model, scaler, explainer, threshold once."""
    global _model, _scaler, _explainer, _threshold
    if _model is not None:
        return

    with open(_MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    _scaler = joblib.load(_SCALER_PATH)

    with open(_THRESHOLD_PATH) as f:
        _threshold = json.load(f)["decision_threshold"]

    # TreeExplainer with interventional feature perturbation
    # (more accurate than default for correlated features like income/debt)
    _explainer = shap.TreeExplainer(
        _model,
        feature_perturbation="interventional",
    )


# ─── Risk tier logic ──────────────────────────────────────────────────────────

def _risk_tier(prob: float, threshold: float) -> dict:
    """
    Assign Green / Yellow / Red tier based on probability.

    Green  : prob >= threshold           → approve with confidence
    Yellow : 0.20 <= prob < threshold    → review needed
    Red    : prob < 0.20                 → reject with confidence
    """
    if prob >= threshold:
        return {
            "tier": "Green",
            "label": "✅ Likely Approvable",
            "color": "#2ecc71",
            "explanation": (
                f"Approval probability {prob:.1%} meets the decision threshold "
                f"of {threshold:.0%}. This applicant shows strong creditworthiness signals."
            ),
        }
    elif prob >= 0.20:
        return {
            "tier": "Yellow",
            "label": "⚠️ Review Needed",
            "color": "#f39c12",
            "explanation": (
                f"Approval probability {prob:.1%} falls in the review zone "
                f"(20%–{threshold:.0%}). Banker judgment required — check the "
                "top risk factors below before deciding."
            ),
        }
    else:
        return {
            "tier": "Red",
            "label": "❌ High Risk",
            "color": "#e74c3c",
            "explanation": (
                f"Approval probability {prob:.1%} is below the minimum review "
                "threshold of 20%. This application shows significant credit risk."
            ),
        }


# ─── Cohort intelligence ──────────────────────────────────────────────────────

def _cohort_intelligence(
    applicant_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 200,
) -> dict:
    """
    Find similar applicants in training data and compute approval rate.
    Similarity is based on the 5 most financially meaningful features.
    """
    cohort_features = [
        "CreditScore", "MonthlyIncome", "LoanAmount",
        "DebtToIncomeRatio", "PreviousLoanDefaults",
    ]
    # Only use features present in X_train
    available = [f for f in cohort_features if f in X_train.columns]

    app_vals = applicant_df[available].values[0]
    train_vals = X_train[available].values

    # Normalise each dimension to 0-1 range to avoid income dominating distance
    col_range = train_vals.max(axis=0) - train_vals.min(axis=0)
    col_range[col_range == 0] = 1  # avoid division by zero

    app_norm   = (app_vals - train_vals.min(axis=0)) / col_range
    train_norm = (train_vals - train_vals.min(axis=0)) / col_range

    distances  = np.linalg.norm(train_norm - app_norm, axis=1)
    neighbor_idx = np.argsort(distances)[:n_neighbors]

    neighbor_labels   = y_train.iloc[neighbor_idx]
    approval_rate     = neighbor_labels.mean()
    cohort_count      = len(neighbor_idx)

    return {
        "cohort_size": cohort_count,
        "approval_rate": approval_rate,
        "approved_count": int(neighbor_labels.sum()),
        "summary": (
            f"Among the {cohort_count} most similar past applicants, "
            f"{approval_rate:.0%} ({int(neighbor_labels.sum())}) were approved."
        ),
    }


# ─── Counterfactual ───────────────────────────────────────────────────────────

def _counterfactual(
    applicant_df: pd.DataFrame,
    current_prob: float,
    threshold: float,
    scaler,
    model,
) -> dict | None:
    """
    If applicant is Yellow (not approved), find the smallest single-feature
    change that would push them to Green (above threshold).

    Returns None if applicant is already Green or Red (too far to recover).
    """
    tier = _risk_tier(current_prob, threshold)["tier"]
    if tier != "Yellow":
        return None

    gap = threshold - current_prob  # how much probability we need to gain

    # Actionable features only — things the applicant can actually change
    actionable = {
        "MonthlyDebtPayments": ("reduce", -0.10),       # reduce by 10% steps
        "CreditCardUtilizationRate": ("reduce", -0.05),
        "PreviousLoanDefaults": ("reduce", -1),
        "SavingsAccountBalance": ("increase", 0.10),
        "LoanAmount": ("reduce", -0.05),
        "LoanDuration": ("reduce", -6),
    }

    results = []
    for feature, (direction, step) in actionable.items():
        if feature not in applicant_df.columns:
            continue

        current_val = applicant_df[feature].values[0]
        if current_val == 0 and direction == "reduce":
            continue

        # Try up to 20 steps
        test_df = applicant_df.copy()
        for steps in range(1, 21):
            if direction == "reduce":
                if current_val > 0:
                    new_val = current_val * (1 + step * steps)   # step is negative
                else:
                    new_val = current_val + step * steps
            else:
                new_val = current_val * (1 + step * steps)

            new_val = max(new_val, 0)
            test_df[feature] = new_val

            scaled = scaler.transform(test_df)
            new_prob = model.predict_proba(scaled)[0][1]

            if new_prob >= threshold:
                delta = new_val - current_val
                label = FEATURE_LABELS.get(feature, feature)
                results.append({
                    "feature": feature,
                    "label": label,
                    "direction": direction,
                    "current_value": current_val,
                    "required_value": new_val,
                    "delta": delta,
                    "new_probability": new_prob,
                    "steps_required": steps,
                    "summary": _cf_summary(label, direction, current_val, new_val, new_prob, threshold),
                })
                break

    if not results:
        return {
            "feasible": False,
            "summary": (
                "No single actionable change was found to move this application "
                "above the approval threshold. Multiple improvements are needed."
            ),
        }

    # Return the easiest counterfactual (fewest steps)
    best = min(results, key=lambda x: x["steps_required"])
    best["feasible"] = True
    return best


def _cf_summary(label, direction, current, required, new_prob, threshold) -> str:
    delta = abs(required - current)
    pct   = abs((required - current) / current * 100) if current != 0 else 0
    verb  = "reducing" if direction == "reduce" else "increasing"
    return (
        f"If {label} is {verb} by {pct:.0f}% "
        f"(from {current:,.0f} to {required:,.0f}), "
        f"approval probability rises to {new_prob:.1%}, "
        f"moving this application from Yellow to Green."
    )


# ─── SHAP plots ───────────────────────────────────────────────────────────────

def _waterfall_plot(shap_vals: np.ndarray, feature_values: np.ndarray,
                    expected_value: float, top_n: int = 10) -> plt.Figure:
    """
    Manual waterfall chart — shows top N features ranked by |SHAP value|.
    Green bars push toward approval, red bars push toward rejection.
    """
    indices    = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_shap   = shap_vals[indices]
    top_labels = [FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in indices]
    top_vals   = feature_values[indices]

    # Sort by SHAP value for waterfall layout
    sort_order = np.argsort(top_shap)
    top_shap   = top_shap[sort_order]
    top_labels = [top_labels[i] for i in sort_order]
    top_vals   = top_vals[sort_order]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_shap]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(range(len(top_shap)), top_shap, color=colors, edgecolor="none", height=0.6)

    # Value annotations
    for i, (val, sv) in enumerate(zip(top_vals, top_shap)):
        x_pos = sv + (0.003 if sv >= 0 else -0.003)
        ha    = "left" if sv >= 0 else "right"
        ax.text(x_pos, i, f"{val:.2f}", va="center", ha=ha,
                fontsize=8, color="white")

    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=9, color="white")
    ax.set_xlabel("SHAP Value  (impact on approval probability)", color="white", fontsize=9)
    ax.set_title("Top Factors Influencing This Decision", color="white", fontsize=11, pad=12)
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left", "bottom"]].set_color("#333")
    ax.axvline(0, color="#555", linewidth=0.8)

    green_patch = mpatches.Patch(color="#2ecc71", label="Raises approval chance")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Lowers approval chance")
    ax.legend(handles=[green_patch, red_patch], loc="lower right",
              facecolor="#1e1e2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig


def _force_plot_static(shap_vals: np.ndarray, feature_values: np.ndarray,
                        expected_value: float, top_n: int = 8) -> plt.Figure:
    """
    Static force-style plot. Shows base value → final prediction push/pull.
    Used as a static image in Streamlit (interactive force plots need JS).
    """
    indices  = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_shap = shap_vals[indices]
    top_lbls = [FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in indices]
    top_vals = feature_values[indices]
    pred     = float(expected_value + shap_vals.sum())

    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Base value marker
    ax.axvline(expected_value, ymin=0.1, ymax=0.9, color="#888", linewidth=1.5, linestyle="--")
    ax.text(expected_value, 0.05, f"base\n{expected_value:.2f}",
            ha="center", va="bottom", fontsize=7, color="#aaa")

    # Final prediction marker
    ax.axvline(pred, ymin=0.1, ymax=0.9, color="white", linewidth=2)
    ax.text(pred, 0.95, f"prediction\n{pred:.2f}",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold")

    # Arrows for each SHAP value
    cursor = expected_value
    for sv, lbl, fv in sorted(zip(top_shap, top_lbls, top_vals),
                               key=lambda x: x[0]):
        color = "#2ecc71" if sv > 0 else "#e74c3c"
        ax.annotate(
            "", xy=(cursor + sv, 0.5), xytext=(cursor, 0.5),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        mid = cursor + sv / 2
        ax.text(mid, 0.62, f"{lbl}\n({fv:.1f})",
                ha="center", fontsize=6.5, color=color)
        cursor += sv

    ax.set_title("Force Plot — How features push the score from baseline",
                 color="white", fontsize=9, pad=6)
    plt.tight_layout()
    return fig


# ─── Top factors (structured output for Person 4) ────────────────────────────

def _top_factors(shap_vals: np.ndarray, feature_values: np.ndarray,
                 top_n: int = 5) -> list[dict]:
    """
    Returns top N factors as structured dicts for Person 4 to render
    without needing to understand SHAP internals.
    """
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    factors = []
    for rank, idx in enumerate(indices, 1):
        sv  = float(shap_vals[idx])
        fv  = float(feature_values[idx])
        lbl = FEATURE_LABELS.get(FEATURE_NAMES[idx], FEATURE_NAMES[idx])
        factors.append({
            "rank": rank,
            "feature": FEATURE_NAMES[idx],
            "label": lbl,
            "value": fv,
            "shap_value": sv,
            "direction": "raises" if sv > 0 else "lowers",
            "impact": "high" if abs(sv) > 0.1 else "medium" if abs(sv) > 0.04 else "low",
            "summary": (
                f"{lbl} = {fv:.2f} "
                f"{'raises' if sv > 0 else 'lowers'} approval chance "
                f"(impact: {sv:+.3f})"
            ),
        })
    return factors


# ─── Auto-generated decision brief ───────────────────────────────────────────

def _decision_brief(
    applicant_id: str,
    prob: float,
    tier: dict,
    top_factors: list[dict],
    cohort: dict,
    counterfactual: dict | None,
) -> str:
    """
    3-4 sentence banker-ready decision brief. Copy-paste ready.
    """
    positive = [f for f in top_factors if f["direction"] == "raises"][:2]
    negative = [f for f in top_factors if f["direction"] == "lowers"][:2]

    pos_str = " and ".join(f["label"] for f in positive) if positive else "no strong positive signals"
    neg_str = " and ".join(f["label"] for f in negative) if negative else "no major risk flags"

    brief = (
        f"Applicant {applicant_id} received an approval probability of {prob:.1%}, "
        f"placing them in the {tier['tier']} tier. "
        f"Key supporting factors include {pos_str}, while {neg_str} weigh against approval. "
        f"{cohort['summary']}"
    )

    if counterfactual and counterfactual.get("feasible"):
        brief += f" {counterfactual['summary']}"

    return brief


# ─── Main public function ─────────────────────────────────────────────────────

def explain(
    applicant_data: dict,
    risk_score: float,
    applicant_id: str = "N/A",
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None,
    top_n: int = 5,
) -> dict:
    """
    Main entry point. Person 4 calls this function only.

    Parameters
    ----------
    applicant_data : dict
        Raw (pre-scale) applicant feature values keyed by feature name.
        Must include all 47 features. Engineered features (MissedPaymentFlag,
        HighUtilizationFlag, SavingsToLoanRatio, one-hot cols) must already
        be computed — same preprocessing as predict_with_guard() input.

    risk_score : float
        Probability from predict_with_guard() — do NOT recompute here.
        We use the fair model's score for display; SHAP is for explanation only.

    applicant_id : str
        Display ID for the decision brief (e.g. "APP-2024-00123").

    X_train : pd.DataFrame (optional)
        Training feature data for cohort intelligence. If None, cohort is skipped.

    y_train : pd.Series (optional)
        Training labels (0/1). Required if X_train is provided.

    top_n : int
        Number of top SHAP factors to return (default 5).

    Returns
    -------
    dict with keys:
        applicant_id, risk_score, tier, top_factors,
        shap_values, expected_value,
        waterfall_plot (Figure), force_plot (Figure),
        cohort (dict or None), counterfactual (dict or None),
        decision_brief (str)
    """
    _load_artifacts()

    # ── 1. Build feature DataFrame in correct column order ─────────────────
    app_df = pd.DataFrame([applicant_data])[FEATURE_NAMES]

    # ── 2. Scale (raw input → model input) ─────────────────────────────────
    X_scaled = _scaler.transform(app_df)

    # ── 3. SHAP values from base XGBoost model ──────────────────────────────
    shap_vals    = _explainer.shap_values(X_scaled)[0]   # shape (47,)
    expected_val = float(_explainer.expected_value)
    feat_vals    = app_df.values[0]                      # raw (unscaled) for display

    # ── 4. Risk tier (uses risk_score from fair model, not base model) ──────
    tier = _risk_tier(risk_score, _threshold)

    # ── 5. Top factors ───────────────────────────────────────────────────────
    factors = _top_factors(shap_vals, feat_vals, top_n=top_n)

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    waterfall = _waterfall_plot(shap_vals, feat_vals, expected_val, top_n=10)
    force     = _force_plot_static(shap_vals, feat_vals, expected_val, top_n=8)

    # ── 7. Cohort intelligence ───────────────────────────────────────────────
    cohort = None
    if X_train is not None and y_train is not None:
        cohort = _cohort_intelligence(app_df, X_train, y_train)

    # ── 8. Counterfactual ────────────────────────────────────────────────────
    cf = _counterfactual(app_df, risk_score, _threshold, _scaler, _model)

    # ── 9. Decision brief ────────────────────────────────────────────────────
    brief = _decision_brief(
        applicant_id, risk_score, tier, factors,
        cohort or {"summary": "Cohort data not available."},
        cf,
    )

    return {
        "applicant_id":   applicant_id,
        "risk_score":     risk_score,
        "tier":           tier,
        "top_factors":    factors,
        "shap_values":    shap_vals.tolist(),
        "expected_value": expected_val,
        "waterfall_plot": waterfall,
        "force_plot":     force,
        "cohort":         cohort,
        "counterfactual": cf,
        "decision_brief": brief,
    }


# ─── Quick smoke-test (run this file directly to verify setup) ────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("explain.py — smoke test")
    print("=" * 60)

    sample = {
        "Age": 35, "AnnualIncome": 480000, "CreditScore": 620,
        "Experience": 8, "LoanAmount": 250000, "LoanDuration": 36,
        "NumberOfDependents": 2, "MonthlyDebtPayments": 8000,
        "CreditCardUtilizationRate": 0.72, "NumberOfOpenCreditLines": 4,
        "NumberOfCreditInquiries": 3, "DebtToIncomeRatio": 0.38,
        "BankruptcyHistory": 0, "PreviousLoanDefaults": 1,
        "PaymentHistory": 0.82, "LengthOfCreditHistory": 6,
        "SavingsAccountBalance": 30000, "CheckingAccountBalance": 15000,
        "TotalAssets": 800000, "TotalLiabilities": 300000,
        "MonthlyIncome": 40000, "UtilityBillsPaymentHistory": 0.9,
        "JobTenure": 5, "NetWorth": 500000, "BaseInterestRate": 0.07,
        "InterestRate": 0.11, "MonthlyLoanPayment": 9500,
        "TotalDebtToIncomeRatio": 0.42,
        "MissedPaymentFlag": 1, "HighUtilizationFlag": 1,
        "SavingsToLoanRatio": 0.12,
        "EmploymentStatus_Self-Employed": 0, "EmploymentStatus_Unemployed": 0,
        "EducationLevel_Bachelor": 1, "EducationLevel_Doctorate": 0,
        "EducationLevel_High School": 0, "EducationLevel_Master": 0,
        "MaritalStatus_Married": 1, "MaritalStatus_Single": 0,
        "MaritalStatus_Widowed": 0,
        "HomeOwnershipStatus_Other": 0, "HomeOwnershipStatus_Own": 0,
        "HomeOwnershipStatus_Rent": 1,
        "LoanPurpose_Debt Consolidation": 0, "LoanPurpose_Education": 0,
        "LoanPurpose_Home": 1, "LoanPurpose_Other": 0,
    }

    # Simulate a risk_score coming from predict_with_guard()
    dummy_risk_score = 0.48

    result = explain(sample, risk_score=dummy_risk_score, applicant_id="APP-TEST-001")

    print(f"\nRisk Score : {result['risk_score']:.2%}")
    print(f"Tier       : {result['tier']['tier']} — {result['tier']['label']}")
    print(f"\nTop Factors:")
    for f in result["top_factors"]:
        print(f"  {f['rank']}. {f['label']} = {f['value']:.2f}  →  {f['direction']} ({f['shap_value']:+.3f})")
    print(f"\nDecision Brief:\n{result['decision_brief']}")

    if result["counterfactual"] and result["counterfactual"]["feasible"]:
        print(f"\nCounterfactual:\n{result['counterfactual']['summary']}")

    result["waterfall_plot"].savefig("waterfall_test.png", dpi=120,
                                  bbox_inches="tight", facecolor="#0f1117")
    result["force_plot"].savefig("force_test.png", dpi=120,
                              bbox_inches="tight", facecolor="#0f1117")
    print("\nPlots saved to /tmp/waterfall_test.png and /tmp/force_test.png")
    print("\n✅ explain.py smoke test passed.")
