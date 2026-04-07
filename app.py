import streamlit as st
import pandas as pd
import joblib          # FIX 1: use joblib not pickle for scaler
import json
import os
from datetime import datetime

from predict import predict_with_guard
from explain import explain

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="FairCredit",
    layout="wide",
    page_icon="🏦"
)

# ── LOAD SHARED RESOURCES ONCE ─────────────────────────────────
@st.cache_resource
def load_resources():
    # FIX 1: was pickle.load() — crashes with UnpicklingError because
    # scaler.pkl was saved with joblib, not pickle
    scaler  = joblib.load("models/scaler.pkl")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    return scaler, X_train, y_train

scaler, X_train, y_train = load_resources()

# ── OVERRIDES FILE ─────────────────────────────────────────────
OVERRIDES_FILE = "overrides.csv"

def log_override(applicant_id, decision, note, risk_score):
    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "applicant_id": applicant_id,
        "decision":     decision,
        "note":         note,
        "risk_score":   risk_score,
    }
    if os.path.exists(OVERRIDES_FILE):
        df = pd.read_csv(OVERRIDES_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(OVERRIDES_FILE, index=False)
    st.success(f"Override logged: **{decision}** for {applicant_id}")

# ── SIDEBAR NAVIGATION ─────────────────────────────────────────
st.sidebar.title("🏦 FairCredit 2.0")
st.sidebar.markdown("*Fair & Explainable Loan AI*")
st.sidebar.divider()
_default_page = st.session_state.pop("_page_override", "📋  Applicant Form")
page = st.sidebar.radio(
    "Navigation",
    ["📋  Applicant Form", "📊  Decision Panel", "🔍  Audit & Fairness"],
    index=["📋  Applicant Form", "📊  Decision Panel", "🔍  Audit & Fairness"].index(_default_page),
)

# ── SAMPLE PROFILES ────────────────────────────────────────────
SAMPLES = {
    "Rahul Mehta — True Moderate (Yellow Tier)": {
    "Age": 34, "AnnualIncome": 58000, "CreditScore": 705,
    "Experience": 7, "LoanAmount": 22000, "LoanDuration": 36,
    "NumberOfDependents": 2, "MonthlyDebtPayments": 1600,
    "CreditCardUtilizationRate": 0.42, "NumberOfOpenCreditLines": 5,
    "NumberOfCreditInquiries": 3, "DebtToIncomeRatio": 0.36,
    "BankruptcyHistory": 0, "PreviousLoanDefaults": 0,
    "PaymentHistory": 0.78, "LengthOfCreditHistory": 6,
    "SavingsAccountBalance": 5000, "CheckingAccountBalance": 2000,
    "TotalAssets": 30000, "TotalLiabilities": 18000,
    "MonthlyIncome": 4833, "UtilityBillsPaymentHistory": 0.82,
    "JobTenure": 3, "NetWorth": 12000,
    "BaseInterestRate": 0.07, "InterestRate": 0.11,
    "MonthlyLoanPayment": 720, "TotalDebtToIncomeRatio": 0.38,
    "MissedPaymentFlag": 0, "HighUtilizationFlag": 1,
    "SavingsToLoanRatio": 0.23,
    "EmploymentStatus_Self-Employed": 0, "EmploymentStatus_Unemployed": 0,
    "EducationLevel_Bachelor": 1, "EducationLevel_Doctorate": 0,
    "EducationLevel_High School": 0, "EducationLevel_Master": 0,
    "MaritalStatus_Married": 1, "MaritalStatus_Single": 0, "MaritalStatus_Widowed": 0,
    "HomeOwnershipStatus_Other": 0, "HomeOwnershipStatus_Own": 0, "HomeOwnershipStatus_Rent": 1,
    "LoanPurpose_Debt Consolidation": 0, "LoanPurpose_Education": 0,
    "LoanPurpose_Home": 1, "LoanPurpose_Other": 0,
    "EMIBurdenRatio": 0.36,
    },
    "Priya Sharma — Low Risk": {
        "Age": 42, "AnnualIncome": 145000, "CreditScore": 780,
        "Experience": 15, "LoanAmount": 30000, "LoanDuration": 24,
        "NumberOfDependents": 1, "MonthlyDebtPayments": 900,
        "CreditCardUtilizationRate": 0.12, "NumberOfOpenCreditLines": 5,
        "NumberOfCreditInquiries": 1, "DebtToIncomeRatio": 0.15,
        "BankruptcyHistory": 0, "PreviousLoanDefaults": 0,
        "PaymentHistory": 0.97, "LengthOfCreditHistory": 14,
        "SavingsAccountBalance": 28000, "CheckingAccountBalance": 8000,
        "TotalAssets": 180000, "TotalLiabilities": 30000,
        "MonthlyIncome": 12083, "UtilityBillsPaymentHistory": 0.95,
        "JobTenure": 12, "NetWorth": 150000,
        "BaseInterestRate": 0.065, "InterestRate": 0.085,
        "MonthlyLoanPayment": 1360, "TotalDebtToIncomeRatio": 0.18,
        "MissedPaymentFlag": 0, "HighUtilizationFlag": 0,
        "SavingsToLoanRatio": 0.93,
        "EmploymentStatus_Self-Employed": 0, "EmploymentStatus_Unemployed": 0,
        "EducationLevel_Bachelor": 0, "EducationLevel_Doctorate": 0,
        "EducationLevel_High School": 0, "EducationLevel_Master": 1,
        "MaritalStatus_Married": 1, "MaritalStatus_Single": 0, "MaritalStatus_Widowed": 0,
        "HomeOwnershipStatus_Other": 0, "HomeOwnershipStatus_Own": 1, "HomeOwnershipStatus_Rent": 0,
        "LoanPurpose_Debt Consolidation": 0, "LoanPurpose_Education": 0,
        "LoanPurpose_Home": 0, "LoanPurpose_Other": 1,
        "EMIBurdenRatio": 0.18,
    },
    "Arjun Verma — High Risk": {
        "Age": 27, "AnnualIncome": 19000, "CreditScore": 480,
        "Experience": 2, "LoanAmount": 12000, "LoanDuration": 60,
        "NumberOfDependents": 3, "MonthlyDebtPayments": 1400,
        "CreditCardUtilizationRate": 0.88, "NumberOfOpenCreditLines": 7,
        "NumberOfCreditInquiries": 8, "DebtToIncomeRatio": 0.88,
        "BankruptcyHistory": 0, "PreviousLoanDefaults": 3,
        "PaymentHistory": 0.41, "LengthOfCreditHistory": 2,
        "SavingsAccountBalance": 300, "CheckingAccountBalance": 150,
        "TotalAssets": 3000, "TotalLiabilities": 14000,
        "MonthlyIncome": 1583, "UtilityBillsPaymentHistory": 0.45,
        "JobTenure": 1, "NetWorth": -11000,
        "BaseInterestRate": 0.09, "InterestRate": 0.18,
        "MonthlyLoanPayment": 304, "TotalDebtToIncomeRatio": 0.91,
        "MissedPaymentFlag": 1, "HighUtilizationFlag": 1,
        "SavingsToLoanRatio": 0.025,
        "EmploymentStatus_Self-Employed": 1, "EmploymentStatus_Unemployed": 0,
        "EducationLevel_Bachelor": 0, "EducationLevel_Doctorate": 0,
        "EducationLevel_High School": 1, "EducationLevel_Master": 0,
        "MaritalStatus_Married": 0, "MaritalStatus_Single": 1, "MaritalStatus_Widowed": 0,
        "HomeOwnershipStatus_Other": 0, "HomeOwnershipStatus_Own": 0, "HomeOwnershipStatus_Rent": 1,
        "LoanPurpose_Debt Consolidation": 1, "LoanPurpose_Education": 0,
        "LoanPurpose_Home": 0, "LoanPurpose_Other": 0,
        "EMIBurdenRatio": 0.88,
    },
}

# ══════════════════════════════════════════════════════════════
# PAGE 1 — APPLICANT INPUT FORM
# ══════════════════════════════════════════════════════════════
if page == "📋  Applicant Form":
    st.title("📋 Applicant Input Form")
    st.caption("Banker enters real applicant values. Scaling is handled automatically.")
    st.divider()

    applicant_id = st.text_input("Applicant ID", value="APP-2024-001")

    col_sample, col_clear = st.columns([3, 1])
    with col_sample:
        selected_sample = st.selectbox(
            "⚡ Load sample application (for demo)",
            ["— select —"] + list(SAMPLES.keys())
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.pop("sample", None)
            st.rerun()

    if selected_sample != "— select —":
        st.session_state["sample"] = selected_sample

    active_sample = st.session_state.get("sample")
    sd = SAMPLES[active_sample] if active_sample and active_sample in SAMPLES else {}

    if active_sample:
        st.info(f"✅ Loaded: **{active_sample}**")

    # FIX A: cast step to float to avoid StreamlitMixedNumericTypesError
    # Streamlit requires all of value/min_value/max_value/step to be the same type.
    # Previously step was passed as int (e.g. 1) while the others were float → crash.
    def field(label, key, min_val=0.0, max_val=None, step=1.0, fmt="%.2f"):
        default = float(sd.get(key, 0))
        kwargs  = dict(label=label, value=default, step=float(step), format=fmt,
                       min_value=float(min_val))
        if max_val is not None:
            kwargs["max_value"] = float(max_val)
        return st.number_input(**kwargs)

    def checkbox(label, key):
        return int(st.checkbox(label, value=bool(sd.get(key, 0)), key=key))

    with st.form("applicant_form"):

        st.subheader("👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age         = field("Age", "Age", 18, 85, 1, "%.0f")
        with c2:
            dependents  = field("Number of Dependents", "NumberOfDependents", 0, 10, 1, "%.0f")
        with c3:
            experience  = field("Years of Experience", "Experience", 0, 50, 1, "%.0f")
        c4, c5, c6 = st.columns(3)
        with c4:
            job_tenure  = field("Job Tenure (years)", "JobTenure", 0, 50, 1, "%.0f")
        with c5:
            emp_self    = checkbox("Self-Employed", "EmploymentStatus_Self-Employed")
        with c6:
            emp_unemp   = checkbox("Unemployed", "EmploymentStatus_Unemployed")

        st.markdown("**Education**")
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1: edu_bach = checkbox("Bachelor",   "EducationLevel_Bachelor")
        with ec2: edu_doc  = checkbox("Doctorate",  "EducationLevel_Doctorate")
        with ec3: edu_hs   = checkbox("High School","EducationLevel_High School")
        with ec4: edu_mast = checkbox("Master",     "EducationLevel_Master")

        st.markdown("**Marital Status**")
        mc1, mc2, mc3 = st.columns(3)
        with mc1: mar_marr = checkbox("Married", "MaritalStatus_Married")
        with mc2: mar_sing = checkbox("Single",  "MaritalStatus_Single")
        with mc3: mar_wid  = checkbox("Widowed", "MaritalStatus_Widowed")

        st.markdown("**Home Ownership**")
        hc1, hc2, hc3 = st.columns(3)
        with hc1: home_own  = checkbox("Own",   "HomeOwnershipStatus_Own")
        with hc2: home_rent = checkbox("Rent",  "HomeOwnershipStatus_Rent")
        with hc3: home_oth  = checkbox("Other", "HomeOwnershipStatus_Other")
        st.divider()

        st.subheader("💰 Financial Profile")
        fc1, fc2, fc3 = st.columns(3)
        with fc1: annual_inc  = field("Annual Income (₹)",     "AnnualIncome",  0, step=1000, fmt="%.0f")
        with fc2: monthly_inc = field("Monthly Income (₹)",    "MonthlyIncome", 0, step=500,  fmt="%.0f")
        with fc3: credit_score= field("Credit Score",          "CreditScore",   300, 900, 1,  "%.0f")

        fc4, fc5, fc6 = st.columns(3)
        with fc4: monthly_debt= field("Monthly Debt Payments (₹)", "MonthlyDebtPayments",      0, step=100, fmt="%.0f")
        with fc5: cc_util     = field("Credit Card Utilization (0–1)", "CreditCardUtilizationRate", 0.0, 1.0, 0.01)
        with fc6: open_lines  = field("Open Credit Lines",     "NumberOfOpenCreditLines",      0, step=1, fmt="%.0f")

        fc7, fc8, fc9 = st.columns(3)
        with fc7: inquiries   = field("Credit Inquiries",      "NumberOfCreditInquiries",      0, step=1, fmt="%.0f")
        with fc8: dti         = field("Debt-to-Income Ratio (0–1)",   "DebtToIncomeRatio",     0.0, 1.5, 0.01)
        with fc9: total_dti   = field("Total DTI Ratio (0–1)", "TotalDebtToIncomeRatio",       0.0, 2.0, 0.01)

        fc10, fc11, fc12 = st.columns(3)
        with fc10: pay_hist   = field("Payment History (0–1)",  "PaymentHistory",              0.0, 1.0, 0.01)
        with fc11: util_hist  = field("Utility Bills Payment (0–1)", "UtilityBillsPaymentHistory", 0.0, 1.0, 0.01)
        with fc12: credit_hist= field("Credit History Length (yrs)", "LengthOfCreditHistory",  0, step=1, fmt="%.0f")

        fc13, fc14, fc15 = st.columns(3)
        with fc13: savings    = field("Savings Account Balance (₹)", "SavingsAccountBalance",  0, step=1000, fmt="%.0f")
        with fc14: checking   = field("Checking Account Balance (₹)","CheckingAccountBalance", 0, step=500,  fmt="%.0f")
        with fc15: net_worth  = field("Net Worth (₹)",               "NetWorth",               min_val=-10_000_000, step=1000, fmt="%.0f")

        fc16, fc17, fc18 = st.columns(3)
        with fc16: total_assets = field("Total Assets (₹)",     "TotalAssets",    0, step=1000, fmt="%.0f")
        with fc17: total_liab   = field("Total Liabilities (₹)","TotalLiabilities",0, step=1000, fmt="%.0f")
        with fc18: stl_ratio    = field("Savings-to-Loan Ratio","SavingsToLoanRatio", 0.0, step=0.01)

        fc19, fc20, fc21 = st.columns(3)
        with fc19: bankrupt    = checkbox("Bankruptcy History",   "BankruptcyHistory")
        with fc20: prev_def    = field("Previous Loan Defaults",  "PreviousLoanDefaults", 0, step=1, fmt="%.0f")
        with fc21: missed_flag = checkbox("Missed Payment Flag",  "MissedPaymentFlag")
        high_util = checkbox("High Utilization Flag", "HighUtilizationFlag")
        st.divider()

        st.subheader("🏦 Loan Details")
        lc1, lc2, lc3 = st.columns(3)
        with lc1: loan_amt   = field("Loan Amount (₹)",             "LoanAmount",      0, step=1000, fmt="%.0f")
        with lc2: loan_dur   = field("Loan Duration (months)",       "LoanDuration",    6, 360, 6, "%.0f")
        with lc3: base_rate  = field("Base Interest Rate (e.g. 0.07)","BaseInterestRate",0.0, 0.3, 0.005)

        lc4, lc5, lc6 = st.columns(3)
        with lc4: int_rate   = field("Interest Rate (e.g. 0.12)",   "InterestRate",    0.0, 0.5, 0.005)
        with lc5: monthly_lp = field("Monthly Loan Payment (₹)",    "MonthlyLoanPayment", 0, step=100, fmt="%.0f")
        with lc6: emi_ratio  = field("EMI Burden Ratio (0–1)",       "EMIBurdenRatio",  0.0, 2.0, 0.01)

        st.markdown("**Loan Purpose**")
        lp1, lp2, lp3, lp4 = st.columns(4)
        with lp1: lp_debt = checkbox("Debt Consolidation", "LoanPurpose_Debt Consolidation")
        with lp2: lp_edu  = checkbox("Education",          "LoanPurpose_Education")
        with lp3: lp_home = checkbox("Home",               "LoanPurpose_Home")
        with lp4: lp_oth  = checkbox("Other",              "LoanPurpose_Other")
        st.divider()

        # FIX B: form submit button was present but Streamlit wasn't finding it —
        # confirmed it exists here inside the st.form() block (required)
        submitted = st.form_submit_button("🔍 Analyse Application", use_container_width=True)

    if submitted:
        with st.spinner("Running risk assessment..."):

            # Build raw input dict with all 47 model features
            raw_input = {
                "Age": age, "AnnualIncome": annual_inc, "CreditScore": credit_score,
                "Experience": experience, "LoanAmount": loan_amt, "LoanDuration": loan_dur,
                "NumberOfDependents": dependents, "MonthlyDebtPayments": monthly_debt,
                "CreditCardUtilizationRate": cc_util, "NumberOfOpenCreditLines": open_lines,
                "NumberOfCreditInquiries": inquiries, "DebtToIncomeRatio": dti,
                "BankruptcyHistory": bankrupt, "PreviousLoanDefaults": prev_def,
                "PaymentHistory": pay_hist, "LengthOfCreditHistory": credit_hist,
                "SavingsAccountBalance": savings, "CheckingAccountBalance": checking,
                "TotalAssets": total_assets, "TotalLiabilities": total_liab,
                "MonthlyIncome": monthly_inc, "UtilityBillsPaymentHistory": util_hist,
                "JobTenure": job_tenure, "NetWorth": net_worth,
                "BaseInterestRate": base_rate, "InterestRate": int_rate,
                "MonthlyLoanPayment": monthly_lp, "TotalDebtToIncomeRatio": total_dti,
                "MissedPaymentFlag": missed_flag, "HighUtilizationFlag": high_util,
                "SavingsToLoanRatio": stl_ratio,
                "EmploymentStatus_Self-Employed": emp_self, "EmploymentStatus_Unemployed": emp_unemp,
                "EducationLevel_Bachelor": edu_bach, "EducationLevel_Doctorate": edu_doc,
                "EducationLevel_High School": edu_hs, "EducationLevel_Master": edu_mast,
                "MaritalStatus_Married": mar_marr, "MaritalStatus_Single": mar_sing,
                "MaritalStatus_Widowed": mar_wid,
                "HomeOwnershipStatus_Other": home_oth, "HomeOwnershipStatus_Own": home_own,
                "HomeOwnershipStatus_Rent": home_rent,
                "LoanPurpose_Debt Consolidation": lp_debt, "LoanPurpose_Education": lp_edu,
                "LoanPurpose_Home": lp_home, "LoanPurpose_Other": lp_oth,
            }

            # Build DataFrame from X_train column order
            all_columns   = X_train.columns.tolist()
            applicant_df  = pd.DataFrame([{col: raw_input.get(col, 0.0) for col in all_columns}])

            # FIX 2: add EMIBurdenRatio for guard check (not a model feature, guard-only)
            applicant_df["EMIBurdenRatio"] = emi_ratio

            # FIX 3: do NOT pass scaler — predict.py now loads it internally
            # (passing scaler here would scale data, then predict.py would scale AGAIN)
            guard = predict_with_guard(applicant_df)

            st.session_state["applicant_id"] = applicant_id
            st.session_state["guard"]        = guard
            st.session_state["raw_input"]    = raw_input

            if guard["flag"]:
                st.warning(f"⚠️ Hard Rule Triggered — {guard['reason']}")
                st.error("Application blocked by risk guard. Cannot proceed to SHAP or decision brief.")
            else:
                # FIX 4: pass raw_input (unscaled) to explain() — explain() scales internally
                # Do NOT pass applicant_df (already prepared for predict) to avoid double-scaling
                result = explain(
                    applicant_data=raw_input,
                    risk_score=guard["risk_score"],
                    applicant_id=applicant_id,
                    X_train=X_train,
                    y_train=y_train,
                )
                st.session_state["result"] = result

                # FIX 5: auto-switch to Decision Panel after analysis
                st.success("✅ Analysis complete!")
                st.session_state["_page_override"] = "📊  Decision Panel"
                st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 2 — DECISION PANEL
# ══════════════════════════════════════════════════════════════
elif page == "📊  Decision Panel":
    st.title("📊 Decision Panel")

    if "result" not in st.session_state:
        st.info("No application analysed yet. Go to **Applicant Form** and submit first.")
        st.stop()

    result       = st.session_state["result"]
    guard        = st.session_state["guard"]
    applicant_id = st.session_state.get("applicant_id", "N/A")
    tier         = result["tier"]
    score        = result["risk_score"]

    # ── Risk Tier Badge ───────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background-color:{tier['color']}22;
            border-left:6px solid {tier['color']};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:16px;">
            <span style="font-size:22px;font-weight:700;color:{tier['color']}">
                {tier['label']}
            </span><br>
            <span style="color:#6B7280;font-size:14px;">Applicant ID: {applicant_id}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    default_risk = 1 - score   # score = approval probability; default risk = 1 - that
    col1.metric("Risk Score (Default %)", f"{default_risk*100:.1f}%")
    col2.metric("AI Decision", "Approve" if guard["approved"] else "Reject")
    col3.metric("Risk Tier",   tier["tier"])
    st.progress(float(default_risk), text=f"Default Risk: {default_risk*100:.1f}%  |  Approval Probability: {score*100:.1f}%")
    st.divider()

    # ── Top Factors ───────────────────────────────────────────
    st.subheader("🔑 Top Factors")
    for factor in result["top_factors"]:
        arrow   = "↑" if factor["direction"] == "raises" else "↓"
        color   = "#10B981" if factor["direction"] == "raises" else "#EF4444"
        badge_c = {"high": "#EF4444", "medium": "#F59E0B", "low": "#6B7280"}.get(factor["impact"], "#6B7280")
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">
                <span style="font-size:20px;color:{color};font-weight:700">{arrow}</span>
                <div>
                    <strong>{factor['label']}</strong>
                    &nbsp;<code>{factor['value']:.2f}</code>
                    &nbsp;<span style="background:{badge_c};color:white;
                        border-radius:4px;padding:2px 8px;font-size:12px;">
                        {factor['impact'].upper()}</span><br>
                    <span style="color:#6B7280;font-size:13px;">{factor['summary']}</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
    st.divider()

    # ── SHAP Plots ────────────────────────────────────────────
    col_w, col_f = st.columns(2)
    with col_w:
        st.subheader("SHAP Waterfall")
        if result.get("waterfall_plot"):
            st.pyplot(result["waterfall_plot"])
    with col_f:
        st.subheader("SHAP Force Plot")
        if result.get("force_plot"):
            st.pyplot(result["force_plot"])
    st.divider()

    # ── Cohort Intelligence ───────────────────────────────────
    st.subheader("👥 Cohort Intelligence")
    # FIX 6: guard against None cohort before accessing ['summary']
    cohort = result.get("cohort")
    if cohort and cohort.get("summary"):
        st.info(cohort["summary"])
    else:
        st.info("Cohort data not available.")

    # ── Counterfactual ────────────────────────────────────────
    cf = result.get("counterfactual")
    if cf and cf.get("feasible"):
        st.subheader("💡 What Would Change the Decision?")
        st.success(cf["summary"])
    st.divider()

    # ── Decision Brief ────────────────────────────────────────
    st.subheader("📝 Decision Brief")
    st.text_area(
        "Copy-paste ready banker summary:",
        value=result["decision_brief"],
        height=120
    )
    st.divider()

    # ── Override Buttons ──────────────────────────────────────
    st.subheader("⚖️ Banker Override")
    st.caption("Your decision overrides the AI. All overrides are logged for audit.")
    override_note = st.text_input(
        "Override reason / notes",
        placeholder="e.g. Medical emergency explained missed payments"
    )
    col_a, col_r = st.columns(2)
    with col_a:
        if st.button("✅ Approve (Override)", use_container_width=True, type="primary"):
            log_override(applicant_id, "APPROVED", override_note, score)
    with col_r:
        if st.button("❌ Reject (Override)", use_container_width=True):
            log_override(applicant_id, "REJECTED", override_note, score)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — AUDIT & FAIRNESS
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Audit & Fairness":
    st.title("🔍 Audit & Fairness Dashboard")
    st.caption("Model performance, fairness metrics, and banker override history.")
    st.divider()

    st.subheader("📊 Fairness: Approval Rate by Age Group")
    fairness_path = "reports/model/fairness_approval_by_agegroup.png"
    if os.path.exists(fairness_path):
        st.image(fairness_path, use_container_width=True)
    else:
        st.caption(f"Image not found at `{fairness_path}`")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        roc_path = "reports/model/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.caption(f"Not found: `{roc_path}`")
    with col2:
        st.subheader("Model Comparison")
        cmp_path = "reports/model/comparison.png"
        if os.path.exists(cmp_path):
            st.image(cmp_path, use_container_width=True)
        else:
            st.caption(f"Not found: `{cmp_path}`")

    st.subheader("Confusion Matrix")
    cm_path = "reports/model/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        st.caption(f"Not found: `{cm_path}`")
    st.divider()

    st.subheader("🎯 Target Benchmarks")
    benchmarks = pd.DataFrame([
        {"Metric": "ROC-AUC",                 "Target": "> 0.85"},
        {"Metric": "F1-score (Approved=1)",    "Target": "> 0.70"},
        {"Metric": "Recall (Approved=1)",      "Target": "> 0.65"},
        {"Metric": "Demographic Parity Ratio", "Target": "> 0.80  (closer to 1 = fairer)"},
        {"Metric": "Equalized Odds Difference","Target": "< 0.10  (closer to 0 = fairer)"},
    ])
    st.dataframe(benchmarks, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("📋 Banker Override Log")
    if os.path.exists(OVERRIDES_FILE):
        overrides_df = pd.read_csv(OVERRIDES_FILE)
        st.dataframe(overrides_df, use_container_width=True)
        csv = overrides_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Override Log", csv, "overrides.csv", "text/csv")
    else:
        st.info("No overrides logged yet. Banker overrides will appear here after the first Decision Panel action.")
