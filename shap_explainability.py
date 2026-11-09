"""
Heart Failure 30-Day Readmission — SHAP Explainability
File: shap_explainability.py

Run:
    python shap_explainability.py

Requires:
    - dataset_12000_records.csv
    - outputs/heart_failure_readmission_best_model.joblib
      (produced by heart_failure_readmission_complete.py)

Requires packages:
    pip install shap

Outputs:
    - outputs/shap/global_summary_bar.png
    - outputs/shap/global_summary_beeswarm.png
    - outputs/shap/global_waterfall_examples.png
    - outputs/shap/patient_explanations.csv
    - outputs/shap/explanation_<patient_id>.png (per-patient, optional)
    - outputs/shap/shap_values_test.npy
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "Readmitted_30_Days"
ID_COL = "Patient_ID"
DATA_FILE = "dataset_12000_records.csv"
MODEL_FILE = os.path.join("outputs", "heart_failure_readmission_best_model.joblib")
OUT_DIR = os.path.join("outputs", "shap")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Load model and data
# ============================================================
print("=" * 60)
print("STEP 1: LOAD MODEL AND DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Same split as training pipeline (must match RANDOM_STATE and test_size)
X = df.drop(columns=[TARGET, ID_COL])
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Test set: {X_test.shape}")


# ============================================================
# 2. Choose SHAP strategy based on model type
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: SELECT SHAP EXPLAINER")
print("=" * 60)

# Get the fitted preprocessor and classifier from the pipeline
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform test set to the model's feature space
X_test_t = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# For sparse one-hot outputs, convert to dense
if hasattr(X_test_t, "toarray"):
    X_test_t = X_test_t.toarray()

print(f"Transformed feature space: {X_test_t.shape}")
print(f"First 10 transformed feature names: {list(feature_names[:10])}")

# Tree-based classifier → TreeExplainer (fast, exact)
# Linear classifier → LinearExplainer
# Otherwise → fall back to KernelExplainer (slow; subsample)

clf_type = type(classifier).__name__
print(f"Classifier type: {clf_type}")

if "HistGradientBoosting" in clf_type:
    try:
        explainer = shap.TreeExplainer(classifier)
        print("Using TreeExplainer.")
    except Exception as e:
        print(f"TreeExplainer failed ({e}); falling back to KernelExplainer.")
        background = shap.sample(X_test_t, 100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
elif "RandomForest" in clf_type:
    explainer = shap.TreeExplainer(classifier)
    print("Using TreeExplainer.")
elif "Logistic" in clf_type:
    explainer = shap.LinearExplainer(classifier, X_test_t)
    print("Using LinearExplainer.")
else:
    print("Using KernelExplainer (will subsample).")
    background = shap.sample(X_test_t, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(classifier.predict_proba, background)


# ============================================================
# 3. Compute SHAP values
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: COMPUTE SHAP VALUES")
print("=" * 60)

# Use a manageable sample for Kernel/Linear explainers; all rows for Tree
n_explain = len(X_test_t) if "Tree" in type(explainer).__name__ else 500
X_explain = X_test_t[:n_explain]

shap_values = explainer.shap_values(X_explain)

# Handle different SHAP output shapes
# Binary classifiers often return list [class0, class1] or a 3D array
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_pos = shap_values[:, :, 1]
else:
    shap_values_pos = shap_values

print(f"SHAP values shape: {shap_values_pos.shape}")

np.save(os.path.join(OUT_DIR, "shap_values_test.npy"), shap_values_pos)
print(f"Saved raw SHAP values → {OUT_DIR}/shap_values_test.npy")


# ============================================================
# 4. Global feature importance (mean |SHAP|)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

mean_abs = np.abs(shap_values_pos).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(os.path.join(OUT_DIR, "global_shap_importance.csv"), index=False)
print("\nTop 20 features by mean |SHAP|:")
print(importance_df.head(20).to_string(index=False))

# Bar plot (top 20)
top_k = min(20, len(importance_df))
plt.figure(figsize=(9, 8))
shap.summary_plot(
    shap_values_pos,
    features=X_explain,
    feature_names=feature_names,
    plot_type="bar",
    max_display=top_k,
    show=False,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_summary_bar.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved → global_summary_bar.png")

# Beeswarm (top 20)
plt.figure(figsize=(9, 8))
shap.summary_plot(
    shap_values_pos,
    features=X_explain,
    feature_names=feature_names,
    max_display=top_k,
    show=False,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_summary_beeswarm.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved → global_summary_beeswarm.png")


# ============================================================
# 5. Per-patient explanations for top-risk patients
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: PER-PATIENT EXPLANATIONS (TOP-RISK)")
print("=" * 60)

# Predicted probabilities
proba = classifier.predict_proba(X_explain)[:, 1]
test_ids = df.loc[X_test.index[:n_explain], ID_COL].values

# Rank by predicted risk
order = np.argsort(-proba)
top_n = 10

records = []
for rank, i in enumerate(order[:top_n], start=1):
    contrib = shap_values_pos[i]
    top_feats_idx = np.argsort(-np.abs(contrib))[:5]
    top_features = [
        {
            "feature": feature_names[j],
            "shap_value": float(contrib[j]),
            "direction": "increases risk" if contrib[j] > 0 else "decreases risk",
        }
        for j in top_feats_idx
    ]
    records.append({
        "rank": rank,
        "patient_id": test_ids[i],
        "predicted_probability": float(proba[i]),
        "top_features": top_features,
    })

# Waterfall plots for top 3
for r in records[:3]:
    i = order[r["rank"] - 1]
    plt.figure(figsize=(9, 6))
    try:
        explanation = shap.Explanation(
            values=shap_values_pos[i],
            base_values=float(np.mean(classifier.predict_proba(X_explain)[:, 1])),
            data=X_explain[i],
            feature_names=list(feature_names),
        )
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.tight_layout()
        fname = f"explanation_{r['patient_id']}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved waterfall → {fname}")
    except Exception as e:
        plt.close()
        print(f"Waterfall skipped for {r['patient_id']}: {e}")

# Export tabular explanations
explanation_rows = []
for r in records:
    for f in r["top_features"]:
        explanation_rows.append({
            "rank": r["rank"],
            "patient_id": r["patient_id"],
            "predicted_probability": r["predicted_probability"],
            "feature": f["feature"],
            "shap_value": f["shap_value"],
            "direction": f["direction"],
        })
expl_df = pd.DataFrame(explanation_rows)
expl_df.to_csv(os.path.join(OUT_DIR, "patient_explanations.csv"), index=False)
print(f"Saved tabular per-patient explanations → {OUT_DIR}/patient_explanations.csv")


# ============================================================
# 6. Console narrative for top patient
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SAMPLE EXPLANATION NARRATIVE")
print("=" * 60)

if records:
    top = records[0]
    print(f"\nPatient: {top['patient_id']}")
    print(f"Predicted 30-day readmission probability: {top['predicted_probability']:.1%}")
    print("Top drivers:")
    for f in top["top_features"]:
        sign = "+" if f["shap_value"] > 0 else "−"
        print(f"  {sign} {f['feature']:<40} (SHAP = {f['shap_value']:+.4f})  → {f['direction']}")
    print(
        "\nInterpretation: features with positive SHAP values push the "
        "prediction toward readmission; negative SHAP values pull it away. "
        "The sum of SHAP values plus the base rate equals the predicted "
        "log-odds."
    )


# ============================================================
# 7. Save model card snippet
# ============================================================
model_card_path = os.path.join(OUT_DIR, "explanation_model_card.txt")
lines = [
    "SHAP EXPLAINABILITY — MODEL CARD SNIPPET",
    "=" * 60,
    f"Model: {type(classifier).__name__}",
    f"Explainer: {type(explainer).__name__}",
    f"Samples explained: {n_explain}",
    f"Feature space: {len(feature_names)} (post-one-hot)",
    "",
    "Top 10 global features by mean |SHAP|:",
    importance_df.head(10).to_string(index=False),
    "",
    "Interpretation notes:",
    "- SHAP values are in log-odds units (positive = pushes toward readmission).",
    "- One-hot categories are shown as individual feature names (e.g., Heart_Failure_Type_HFrEF).",
    "- Per-patient explanations are stored in patient_explanations.csv.",
    "- Waterfall plots are saved per top-risk patient for review.",
    "=" * 60,
]
with open(model_card_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved → {model_card_path}")
print("\nDone.")