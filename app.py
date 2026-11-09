"""
Heart Failure 30-Day Readmission — Interactive Risk Scoring App
File: app.py

Run:
    streamlit run app.py

Requires:
    - outputs/heart_failure_readmission_best_model.joblib
    - dataset_12000_records.csv  (for feature schema and defaults)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
MODEL_PATH = os.path.join("outputs", "heart_failure_readmission_best_model.joblib")
DATA_PATH = "dataset_12000_records.csv"
TARGET = "Readmitted_30_Days"
ID_COL = "Patient_ID"

st.set_page_config(
    page_title="HF Readmission Risk",
    page_icon="❤️",
    layout="wide",
)

# ============================================================
# Load model and reference data
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference():
    df = pd.read_csv(DATA_PATH)
    return df

try:
    model = load_model()
    ref_df = load_reference()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: {e}.\n\n"
        "Run `heart_failure_readmission_complete.py` first to produce "
        "`outputs/heart_failure_readmission_best_model.joblib`."
    )
    st.stop()

# Derive schema from reference data (excluding target and ID)
feature_cols = [c for c in ref_df.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in feature_cols if ref_df[c].dtype == object]
num_cols = [c for c in feature_cols if c not in cat_cols]

# ============================================================
# Sidebar — model information
# ============================================================
with st.sidebar:
    st.title("❤️ HF Readmission Risk")
    st.markdown("**Model:** Hist Gradient Boosting (best of 3)")
    st.markdown("**Target:** 30-day readmission")
    st.markdown("**Trained on:** 12,000 HF admissions")
    st.markdown("---")
    st.caption(
        "⚠️ **Decision support only.** Not a substitute for clinical "
        "judgment. Thresholds should be validated locally."
    )
    threshold = st.slider(
        "Risk threshold for 'High Risk' flag",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lower = catch more readmissions (higher recall), more false alarms."
    )

# ============================================================
# Main form
# ============================================================
st.title("Heart Failure 30-Day Readmission Risk")
st.markdown(
    "Enter patient characteristics. The model returns a predicted "
    "probability of readmission within 30 days."
)

with st.form("patient_form"):
    st.header("Patient Profile")

    # Group inputs into logical tabs
    tab_demo, tab_clin, tab_labs, tab_meds = st.tabs(
        ["Demographics & Lifestyle", "Clinical", "Labs & Vitals", "Medications"]
    )

    inputs = {}

    # ---------- Demographics & lifestyle ----------
    with tab_demo:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["Age"] = st.number_input("Age (years)", 18, 110, 70)
            inputs["Gender"] = st.selectbox(
                "Gender", sorted(ref_df["Gender"].dropna().unique())
            )
            inputs["BMI"] = st.number_input("BMI", 12.0, 60.0, 26.0, step=0.1)
        with c2:
            inputs["Smoking_Status"] = st.selectbox(
                "Smoking Status", sorted(ref_df["Smoking_Status"].dropna().unique())
            )
            inputs["Alcohol_Consumption"] = st.selectbox(
                "Alcohol Consumption",
                sorted(ref_df["Alcohol_Consumption"].dropna().unique()),
            )
            inputs["Exercise_Frequency"] = st.number_input(
                "Exercise Frequency (sessions/week)", 0, 7, 1
            )
        with c3:
            inputs["Hypertension"] = st.selectbox(
                "Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["Diabetes"] = st.selectbox(
                "Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["Chronic_Kidney_Disease"] = st.selectbox(
                "Chronic Kidney Disease",
                [0, 1], format_func=lambda x: "Yes" if x else "No",
            )

    # ---------- Clinical ----------
    with tab_clin:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["Coronary_Artery_Disease"] = st.selectbox(
                "Coronary Artery Disease",
                [0, 1], format_func=lambda x: "Yes" if x else "No",
            )
            inputs["Previous_Stroke"] = st.selectbox(
                "Previous Stroke", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["Atrial_Fibrillation"] = st.selectbox(
                "Atrial Fibrillation",
                [0, 1], format_func=lambda x: "Yes" if x else "No",
            )
        with c2:
            inputs["Previous_HF_Admissions"] = st.number_input(
                "Previous HF Admissions", 0, 20, 0
            )
            inputs["Previous_Hospital_Admissions"] = st.number_input(
                "Previous Hospital Admissions", 0, 30, 1
            )
            inputs["Heart_Failure_Type"] = st.selectbox(
                "Heart Failure Type",
                sorted(ref_df["Heart_Failure_Type"].dropna().unique()),
            )
        with c3:
            inputs["NYHA_Class"] = st.selectbox(
                "NYHA Class", sorted(ref_df["NYHA_Class"].dropna().unique())
            )
            inputs["Ejection_Fraction"] = st.number_input(
                "Ejection Fraction (%)", 5.0, 80.0, 35.0, step=0.5
            )
            inputs["BNP"] = st.number_input(
                "BNP (pg/mL)", 0.0, 5000.0, 300.0, step=10.0
            )

    # ---------- Labs & vitals ----------
    with tab_labs:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["Systolic_BP"] = st.number_input(
                "Systolic BP (mmHg)", 60, 220, 120
            )
            inputs["Diastolic_BP"] = st.number_input(
                "Diastolic BP (mmHg)", 30, 140, 75
            )
            inputs["Heart_Rate"] = st.number_input("Heart Rate (bpm)", 30, 180, 80)
        with c2:
            inputs["Oxygen_Saturation"] = st.number_input(
                "Oxygen Saturation (%)", 70.0, 100.0, 95.0, step=0.5
            )
            inputs["Creatinine"] = st.number_input(
                "Creatinine (mg/dL)", 0.2, 15.0, 1.2, step=0.1
            )
            inputs["Sodium"] = st.number_input(
                "Sodium (mEq/L)", 110.0, 160.0, 138.0, step=0.5
            )
        with c3:
            inputs["Potassium"] = st.number_input(
                "Potassium (mEq/L)", 2.0, 7.0, 4.3, step=0.1
            )
            inputs["Hemoglobin"] = st.number_input(
                "Hemoglobin (g/dL)", 5.0, 20.0, 12.5, step=0.1
            )
            inputs["Blood_Glucose"] = st.number_input(
                "Blood Glucose (mg/dL)", 40.0, 500.0, 110.0, step=5.0
            )

    # ---------- Medications & utilization ----------
    with tab_meds:
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs["Length_of_Stay"] = st.number_input(
                "Length of Stay (days)", 1, 60, 5
            )
            inputs["ICU_Admission"] = st.selectbox(
                "ICU Admission", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["Emergency_Admission"] = st.selectbox(
                "Emergency Admission",
                [0, 1], format_func=lambda x: "Yes" if x else "No",
            )
        with c2:
            inputs["Beta_Blocker"] = st.selectbox(
                "Beta Blocker", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["ACE_ARB"] = st.selectbox(
                "ACE inhibitor / ARB", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            inputs["Diuretic"] = st.selectbox(
                "Diuretic", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
        with c3:
            inputs["SGLT2_Inhibitor"] = st.selectbox(
                "SGLT2 Inhibitor", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )

    submitted = st.form_submit_button("Predict Readmission Risk", type="primary")

# ============================================================
# Prediction & display
# ============================================================
if submitted:
    # Build a single-row DataFrame in the model's expected column order
    patient = pd.DataFrame([inputs])[feature_cols]

    proba = float(model.predict_proba(patient)[0, 1])

    st.markdown("---")
    st.header("Prediction")

    # KPI block
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("30-Day Readmission Risk", f"{proba * 100:.1f}%")
    with col2:
        risk_band = (
            "Low" if proba < 0.33 else "Moderate" if proba < threshold else "High"
        )
        st.metric("Risk Band", risk_band)
    with col3:
        decision = "🚨 High — flag for follow-up" if proba >= threshold else "✅ Below threshold"
        st.metric("Action", decision)

    # Risk gauge
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([0], [1], color="#e2e8f0", height=0.5)
    color = "#ef4444" if proba >= threshold else "#f59e0b" if proba >= 0.33 else "#22c55e"
    ax.barh([0], [proba], color=color, height=0.5)
    ax.axvline(threshold, color="#0f172a", linestyle="--", linewidth=1.5)
    ax.text(threshold, 0.35, f" threshold={threshold:.2f}",
            fontsize=9, va="bottom")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Predicted 30-day readmission probability")
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    # ---- Per-patient SHAP explanation ----
    with st.expander("Why this prediction? (feature contributions)"):
        try:
            import shap
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            X_t = preprocessor.transform(patient)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(classifier)
            sv = explainer.shap_values(X_t)
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)

            contrib = sv[0]
            top_idx = np.argsort(-np.abs(contrib))[:8]
            contrib_df = pd.DataFrame({
                "Feature": [str(feature_names[i]) for i in top_idx],
                "SHAP (log-odds)": contrib[top_idx].round(4),
                "Direction": ["↑ risk" if contrib[i] > 0 else "↓ risk" for i in top_idx],
            })
            st.dataframe(contrib_df, use_container_width=True)
            st.caption(
                "Positive SHAP = pushes risk up; negative = pulls risk down. "
                "Values are in log-odds units."
            )
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    # Interpretation
    st.subheader("Interpretation")
    if proba >= threshold:
        st.warning(
            f"Predicted risk of **{proba*100:.1f}%** is at or above the "
            f"selected threshold ({threshold:.2f}). Consider:"
        )
        st.markdown(
            "- Enroll in a transition-of-care / follow-up program\n"
            "- Schedule early post-discharge phone call (48–72 h)\n"
            "- Prioritize medication reconciliation\n"
            "- Screen for social risk (transport, caregiver, insurance)"
        )
    else:
        st.success(
            f"Predicted risk of **{proba*100:.1f}%** is below the "
            f"threshold ({threshold:.2f}). Standard discharge pathway."
        )

    # Feature transparency
    with st.expander("Input summary"):
        st.dataframe(
            pd.DataFrame([inputs]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Note:** This risk score is generated by a statistical model "
        "trained on retrospective data. It is intended to support — not "
        "replace — clinical judgment. Validate locally before clinical use."
    )
else:
    st.info(
        "Fill in the patient profile and click **Predict Readmission Risk**. "
        "Defaults are prefilled with typical values."
    )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Project 1 · Heart Failure 30-Day Readmission Prediction · "
    "Hist Gradient Boosting · Trained on 12,000 records"
)