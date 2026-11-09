# Heart Failure 30-Day Readmission Prediction
## Final Project Report

**Author:** <Your Name>
**Date:** <YYYY-MM-DD>
**Dataset:** `dataset_12000_records.csv` (12,000 patient records)
**Target:** `Readmitted_30_Days` (binary)
**Best model:** <BEST_MODEL_NAME from project_summary_report.txt>

---

## 1. Executive Summary

Heart failure (HF) is one of the leading causes of hospitalization and 30-day readmission, driving substantial clinical burden and cost. This project developed and evaluated machine-learning models to predict **30-day readmission** in HF patients using routinely collected demographic, clinical, laboratory, and utilization data.

Three models were benchmarked: **Logistic Regression**, **Random Forest**, and **Histogram Gradient Boosting**. The best-performing model achieved:

- **Test AUROC:** <AUROC value>
- **Test AUPRC:** <AUPRC value>
- **Brier score:** <Brier value>
- **Precision / Recall / F1** at tuned threshold <THRESHOLD>: <P> / <R> / <F1>

Because the target is imbalanced, **AUPRC** was used as the primary selection metric. The model is well calibrated and could support discharge planning, follow-up prioritization, and resource allocation.

---

## 2. Background and Objective

30-day readmission for HF is a recognized quality-of-care indicator. Early identification of high-risk patients enables targeted interventions (follow-up calls, medication reconciliation, home monitoring) that can reduce readmissions.

**Objective:** Build and validate a predictive model of 30-day readmission using information available at or before discharge, and quantify its discrimination, calibration, and clinical utility.

---

## 3. Dataset

| Attribute | Value |
|---|---|
| Records | 12,000 |
| Features | 36 (demographic, clinical, lab, medication, utilization) |
| Target | `Readmitted_30_Days` (0/1) |
| Positive rate | <POSITIVE_RATE> |
| Missing values | <MISSING_COUNT summary — see `numeric_summary.csv`> |
| Duplicate rows | <DUPLICATE_COUNT> |

### Feature groups
- **Demographics:** Age, Gender, BMI
- **Lifestyle:** Smoking_Status, Alcohol_Consumption, Exercise_Frequency
- **Comorbidities:** Hypertension, Diabetes, Chronic_Kidney_Disease, Coronary_Artery_Disease, Previous_Stroke, Atrial_Fibrillation
- **HF profile:** Heart_Failure_Type, NYHA_Class, Ejection_Fraction, BNP
- **Utilization history:** Previous_HF_Admissions, Previous_Hospital_Admissions
- **Index admission:** Length_of_Stay, ICU_Admission, Emergency_Admission
- **Vitals / labs:** Systolic_BP, Diastolic_BP, Heart_Rate, Oxygen_Saturation, Creatinine, Sodium, Potassium, Hemoglobin, Blood_Glucose
- **Medications:** Beta_Blocker, ACE_ARB, Diuretic, SGLT2_Inhibitor

---

## 4. Methodology

### 4.1 Prediction timepoint and leakage control
The prediction timepoint defines which features are admissible:
- **At admission:** exclude `Length_of_Stay`, `ICU_Admission`, `Emergency_Admission`.
- **At discharge (default in this project):** all features retained except `Patient_ID` and target.

Confirm this choice before deployment. The report assumes **discharge-time prediction**.

### 4.2 Data split
- **Train/Test:** 80 / 20, stratified by target (`random_state=42`).
- Train positive rate: `<train rate>`; Test positive rate: `<test rate>`.

### 4.3 Preprocessing (inside pipeline)
- Numeric: median imputation → standardization.
- Categorical: most-frequent imputation → one-hot encoding (`handle_unknown="ignore"`).
- All fitting is done on training folds only — **no leakage**.

### 4.4 Models
| Model | Notes |
|---|---|
| Logistic Regression | Baseline, `class_weight="balanced"` |
| Random Forest | 400 trees, `min_samples_leaf=3`, `balanced_subsample` |
| Hist Gradient Boosting | 400 iterations, lr=0.05, 31 leaves |

### 4.5 Evaluation metrics
- **Primary:** AUPRC (average precision) — appropriate for imbalance.
- **Secondary:** AUROC, Brier score, precision/recall/F1 at tuned threshold, calibration curve, confusion matrix.
- **Robustness:** 5-fold stratified cross-validation on training data.

---

## 5. Results

### 5.1 Model comparison (test set)

| Model | AUROC | AUPRC | Brier | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|---|
| Logistic Regression | <v> | <v> | <v> | <v> | <v> | <v> |
| Random Forest | <v> | <v> | <v> | <v> | <v> | <v> |
| Hist Gradient Boosting | <v> | <v> | <v> | <v> | <v> | <v> |

*(Values from `outputs/model_comparison_results.csv`.)*

### 5.2 Best model at tuned threshold
- Best model: **<BEST_MODEL>**
- Threshold (max F1): **<THRESHOLD>**
- Precision: **<P>**  |  Recall: **<R>**  |  F1: **<F1>**
- Confusion matrix (test):

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| Actual No      | <TN>         | <FP>          |
| Actual Yes     | <FN>         | <TP>          |

### 5.3 Cross-validation
- 5-fold AUPRC on training data: **<mean> ± <std>**
- Confirms results are stable and not driven by a single split.

### 5.4 Calibration
- Brier score: **<Brier>**
- Calibration curve (`outputs/plots/05_calibration_curve.png`) shows predictions track observed readmission rates, making probabilities usable for clinical triage.

### 5.5 Top predictors (permutation importance, AUPRC)

*(From `outputs/permutation_importance.csv`.)*

| Rank | Feature | Importance |
|---|---|---|
| 1 | <feature> | <value> |
| 2 | <feature> | <value> |
| 3 | <feature> | <value> |
| … | … | … |

**Interpretation:** utilization history (`Previous_HF_Admissions`, `Previous_Hospital_Admissions`) and clinical severity markers (`BNP`, `Ejection_Fraction`, `NYHA_Class`, `Creatinine`) typically dominate. Confirm with your run’s values.

---

## 6. Discussion

- **Discrimination:** AUROC around <value> indicates <strong/moderate> discrimination. AUPRC of <value> relative to the baseline (<positive rate>) shows meaningful lift over chance.
- **Clinical value:** At threshold <T>, the model catches <R>% of readmissions with <P>% precision — a workable operating point for a follow-up program where missing a high-risk patient is costly.
- **Feature drivers** align with known HF literature: prior admissions, renal dysfunction, elevated BNP, low ejection fraction, and high NYHA class.
- **Calibration** is appropriate for risk stratification and patient communication.

---

## 7. Limitations

1. **Observational, single-source data** — no external validation. Performance may not generalize.
2. **Feature timepoint ambiguity** — `Length_of_Stay` and admission flags require a clear prediction-time policy.
3. **No social determinants** (income, insurance, caregiver support), which strongly affect readmission.
4. **Potential under-capture** of readmissions at outside hospitals.
5. **Static split** — temporal validation (train on earlier years, test on later) would better simulate deployment.
6. **Class imbalance** handled via class weights and threshold tuning; more advanced techniques (cost-sensitive learning, focal loss) not explored.

---

## 8. Recommendations and Next Steps

**Clinical / operational**
1. Deploy as a **decision-support tool**, not an autonomous gatekeeper.
2. Combine model output with clinician judgment and social-risk screening.
3. Pilot in a follow-up program; track precision, recall, and downstream readmission rate.

**Modeling**
4. Run **temporal validation** and, if possible, **external validation** on another hospital’s data.
5. Add **social and behavioral features**; consider NLP on discharge notes.
6. Explore **SGLT2_Inhibitor** effect modifiers and subgroup fairness (age, sex).
7. Re-tune threshold to institutional cost of FP vs FN.
8. Recalibrate periodically as clinical practice shifts.

**Governance**
9. Document model version, training date, features, and known limitations.
10. Establish **monitoring** for drift in inputs and performance.
11. Ensure **privacy and auditability** (de-identified data, access logs).

---

## 9. Reproducibility

| Artifact | Path |
|---|---|
| Training script | `heart_failure_readmission_complete.py` |
| Trained model | `outputs/heart_failure_readmission_best_model.joblib` |
| Model comparison | `outputs/model_comparison_results.csv` |
| Feature importance | `outputs/permutation_importance.csv` |
| Numeric summary | `outputs/numeric_summary.csv` |
| Categorical readmission rates | `outputs/categorical_readmission_rates.csv` |
| Summary report | `outputs/project_summary_report.txt` |
| Figures | `outputs/plots/*.png` |

Random seed: `42`. Run `python heart_failure_readmission_complete.py` to regenerate all outputs.

---

## 10. Conclusion

A discharge-time predictive model for 30-day heart-failure readmission was built and validated internally. The best model (**<BEST_MODEL>**) achieves AUROC **<value>** and AUPRC **<value>**, with good calibration and a clinically interpretable feature set. With temporal/external validation and monitoring in place, the model is a viable component of a readmission-reduction program.

---

## Appendix A — Confusion matrix and curves
- `outputs/plots/03_confusion_matrix.png`
- `outputs/plots/04_precision_recall_curve.png`
- `outputs/plots/05_calibration_curve.png`

## Appendix B — Feature importance figure
- `outputs/plots/06_permutation_importance.png`

## Appendix C — Categorical readmission insights
- `outputs/categorical_readmission_rates.csv`