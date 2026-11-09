"""
Project 1 - Heart Failure 30-Day Readmission Prediction
File: heart_failure_readmission_complete.py

Input:  dataset_12000_records.csv
Outputs:
    - heart_failure_readmission_best_model.joblib
    - model_comparison_results.csv
    - permutation_importance.csv
    - project_summary_report.txt
    - plots/ (EDA, calibration, importance, confusion matrix)

Run:
    python heart_failure_readmission_complete.py
"""

# ============================================================
# 0. Imports and configuration
# ============================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, f1_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42
TARGET = "Readmitted_30_Days"
ID_COL = "Patient_ID"
DATA_FILE = "dataset_12000_records.csv"
OUTPUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_style("whitegrid")


# ============================================================
# 1. Load and audit
# ============================================================
print("=" * 60)
print("STEP 1: LOAD AND AUDIT DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"\nTarget counts:\n{df[TARGET].value_counts(dropna=False)}")
print(f"\nTarget positive rate: {df[TARGET].mean():.4f}")
print(f"\nMissing values (top 20):\n{df.isna().sum().sort_values(ascending=False).head(20)}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")


# ============================================================
# 2. EDA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 2a. Target balance
plt.figure(figsize=(5, 3))
sns.countplot(data=df, x=TARGET)
plt.title("30-Day Readmission Balance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_target_balance.png"), dpi=120)
plt.close()

# 2b. Numeric summary
num_summary = df.describe().T
num_summary.to_csv(os.path.join(OUTPUT_DIR, "numeric_summary.csv"))
print("\nNumeric summary saved.")

# 2c. Categorical readmission rates
all_cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
all_cat_cols = [c for c in all_cat_cols if c not in [ID_COL, TARGET]]

cat_rate_records = []
for c in all_cat_cols:
    if df[c].nunique() <= 12:
        rate = df.groupby(c)[TARGET].mean()
        for level, r in rate.items():
            cat_rate_records.append({"feature": c, "level": level, "readmission_rate": r})
cat_rate_df = pd.DataFrame(cat_rate_records)
cat_rate_df.to_csv(os.path.join(OUTPUT_DIR, "categorical_readmission_rates.csv"), index=False)
print("Categorical readmission rates saved.")

# 2d. Correlation heatmap for numeric features
plt.figure(figsize=(14, 12))
corr = df.select_dtypes(exclude=["object"]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar=True)
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_correlation_heatmap.png"), dpi=100)
plt.close()

# 2e. Target correlation ranking
target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nTop numeric correlations with target:")
print(target_corr.head(15))


# ============================================================
# 3. Feature preparation (leakage-aware)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE PREPARATION")
print("=" * 60)

# NOTE ON LEAKAGE:
# If predicting at ADMISSION, remove: Length_of_Stay, ICU_Admission, Emergency_Admission
# (they are only known after admission/discharge).
# Default below keeps them (assumes discharge-time prediction). Adjust as needed.

drop_cols = [TARGET, ID_COL]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"Numeric features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")


# ============================================================
# 4. Train / test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test  positive rate: {y_test.mean():.4f}")


# ============================================================
# 5. Preprocessing pipelines
# ============================================================
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])


# ============================================================
# 6. Candidate models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        random_state=RANDOM_STATE
    ),
}


# ============================================================
# 7. Train and evaluate each model
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING AND COMPARISON")
print("=" * 60)

results = []
fitted_models = {}

for name, clf in models.items():
    print(f"\nTraining: {name}")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    # Default 0.5 threshold metrics
    pred05 = (proba >= 0.5).astype(int)
    results.append({
        "Model": name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": brier,
        "Precision@0.5": precision_score(y_test, pred05, zero_division=0),
        "Recall@0.5": recall_score(y_test, pred05, zero_division=0),
        "F1@0.5": f1_score(y_test, pred05, zero_division=0),
    })
    fitted_models[name] = pipe
    print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")

results_df = pd.DataFrame(results).sort_values("AUPRC", ascending=False).reset_index(drop=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))


# ============================================================
# 8. Best model, threshold tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: THRESHOLD TUNING ON BEST MODEL")
print("=" * 60)

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
proba_test = best_model.predict_proba(X_test)[:, 1]

prec, rec, thresholds = precision_recall_curve(y_test, proba_test)
f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.argmax(f1_curve))
best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

pred_test = (proba_test >= best_threshold).astype(int)
print(f"Best model: {best_name}")
print(f"Best threshold (max F1): {best_threshold:.4f}")
print(f"Precision: {precision_score(y_test, pred_test):.4f}")
print(f"Recall:    {recall_score(y_test, pred_test):.4f}")
print(f"F1:        {f1_score(y_test, pred_test):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred_test, digits=3))

cm = confusion_matrix(y_test, pred_test)
print("Confusion matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Readmit", "Readmit"],
            yticklabels=["No Readmit", "Readmit"])
plt.title(f"Confusion Matrix — {best_name} @ thr={best_threshold:.3f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_confusion_matrix.png"), dpi=120)
plt.close()


# ============================================================
# 9. Precision-recall and calibration curves
# ============================================================
# PR curve
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"{best_name} (AUPRC={results_df.iloc[0]['AUPRC']:.3f})")
plt.axvline(recall_score(y_test, pred_test), color="gray", linestyle="--", label="Chosen threshold")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_precision_recall_curve.png"), dpi=120)
plt.close()

# Calibration
prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10)
plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o", label=best_name)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
plt.xlabel("Predicted probability")
plt.ylabel("Observed readmission rate")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_calibration_curve.png"), dpi=120)
plt.close()


# ============================================================
# 10. Permutation importance
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE (PERMUTATION)")
print("=" * 60)

sample_n = min(2000, len(X_test))
X_imp = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
y_imp = y_test.loc[X_imp.index]

perm = permutation_importance(
    best_model, X_imp, y_imp,
    scoring="average_precision",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

imp_df = pd.DataFrame({
    "Feature": X_imp.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

imp_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)
print("\nTop 15 features:")
print(imp_df.head(15).to_string(index=False))

plt.figure(figsize=(9, 7))
sns.barplot(data=imp_df.head(15), x="Importance", y="Feature")
plt.title("Top 15 Permutation Importances (AUPRC)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_permutation_importance.png"), dpi=120)
plt.close()


# ============================================================
# 11. Cross-validation sanity check
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: 5-FOLD CROSS-VALIDATION ON BEST MODEL")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    best_model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
)
print(f"CV AUPRC scores: {cv_scores.round(4)}")
print(f"CV AUPRC mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ============================================================
# 12. Save model and summary report
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SAVE ARTIFACTS")
print("=" * 60)

model_path = os.path.join(OUTPUT_DIR, "heart_failure_readmission_best_model.joblib")
joblib.dump(best_model, model_path)

summary_lines = [
    "=" * 60,
    "HEART FAILURE 30-DAY READMISSION PREDICTION — SUMMARY",
    "=" * 60,
    f"Dataset: {DATA_FILE}",
    f"Rows: {df.shape[0]}  Columns: {df.shape[1]}",
    f"Target: {TARGET}",
    f"Overall positive rate: {y.mean():.4f}",
    "",
    "Model comparison:",
    results_df.to_string(index=False),
    "",
    f"Best model: {best_name}",
    f"Test AUROC:  {results_df.iloc[0]['AUROC']:.4f}",
    f"Test AUPRC:  {results_df.iloc[0]['AUPRC']:.4f}",
    f"Test Brier:  {results_df.iloc[0]['Brier']:.4f}",
    f"Chosen threshold: {best_threshold:.4f}",
    f"Precision: {precision_score(y_test, pred_test):.4f}",
    f"Recall:    {recall_score(y_test, pred_test):.4f}",
    f"F1:        {f1_score(y_test, pred_test):.4f}",
    f"CV AUPRC (5-fold, mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
    "",
    "Top 15 permutation importances:",
    imp_df.head(15).to_string(index=False),
    "",
    "Notes:",
    "- Adjust the drop list for Length_of_Stay / ICU_Admission / Emergency_Admission",
    "  depending on prediction timepoint (admission vs discharge).",
    "- AUPRC is the primary metric due to class imbalance.",
    "- Model is calibrated but check calibration curve before deployment.",
    "=" * 60,
]
report_text = "\n".join(summary_lines)
report_path = os.path.join(OUTPUT_DIR, "project_summary_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(report_text)
print("\nSaved files:")
print(f"  - {model_path}")
print(f"  - {OUTPUT_DIR}/model_comparison_results.csv")
print(f"  - {OUTPUT_DIR}/permutation_importance.csv")
print(f"  - {OUTPUT_DIR}/numeric_summary.csv")
print(f"  - {OUTPUT_DIR}/categorical_readmission_rates.csv")
print(f"  - {report_path}")
print(f"  - {PLOT_DIR}/ (all figures)")
print("\nDone.")