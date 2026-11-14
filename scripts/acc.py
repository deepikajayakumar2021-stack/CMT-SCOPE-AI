
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\CMT_PROJECT")
MANIFEST = ROOT / "processed" / "splits" / "stage5c_patient_manifest.csv"
OUTPUT = ROOT / "results" / "accuracy_improvement"
OUTPUT.mkdir(parents=True, exist_ok=True)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)
print("Manifest columns:", df.columns.tolist())

# Keep one row per patient. Resolve conflicting labels before proceeding.
required = {"ID", "severity_name"}
if not required.issubset(df.columns):
    raise ValueError(f"Required columns missing: {required - set(df.columns)}")

df["ID"] = df["ID"].astype(str).str.strip()
df["severity_name"] = df["severity_name"].astype(str).str.strip().str.title()
df = df[df["severity_name"].isin(["Mild", "Moderate", "Severe"])].copy()

conflicts = df.groupby("ID")["severity_name"].nunique()
conflicts = conflicts[conflicts > 1]
if len(conflicts):
    raise ValueError(
        "Conflicting severity labels found for patients: "
        + ", ".join(conflicts.index)
        + ". Reconcile the source labels before training."
    )

df = df.drop_duplicates("ID").reset_index(drop=True)

label_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
df["y"] = df["severity_name"].map(label_map)

# Only use independent clinical features. Do NOT include IDs,
# severity labels, CMTES, or any direct label-derived variables.
candidate_features = [
    "Age (Years)", "Gender_encoded", "Pinprick", "Vibration"
]
features = [c for c in candidate_features if c in df.columns]

if not features:
    raise ValueError(
        "No supported clinical features found. Check the manifest column names."
    )

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["y"].astype(int)

# Severe has only one patient, so reliable three-class generalisation
# cannot be established with this dataset alone.
print("\nPatients:", len(df))
print("Features:", features)
print("\nClass counts:")
print(df["severity_name"].value_counts().to_string())

if y.value_counts().min() < 2:
    print(
        "\nWARNING: At least one class has fewer than two patients. "
        "Ordinary stratified CV is not valid for all three classes."
    )

# Leave-one-patient-out avoids splitting multiple rows of the same patient
# across training and test sets. Each fold predicts one held-out patient.
cv = LeaveOneOut()

models = {
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=1,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=1,
        random_state=42,
    ),
    "SVM RBF": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            probability=False,
            random_state=42,
        ),
    ),
}

results = []
predictions = {}

for name, model in models.items():
    pipe = model
    if name in ("Extra Trees", "Random Forest"):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            clone(model),
        )

    # Every patient is held out once.
    pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    predictions[name] = pred

    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)
    macro = f1_score(y, pred, average="macro", zero_division=0)

    results.append({
        "model": name,
        "LOOCV_accuracy": acc,
        "LOOCV_balanced_accuracy": bal,
        "LOOCV_macro_f1": macro,
    })

    print(f"\n=== {name} ===")
    print("LOOCV accuracy:", round(acc, 4))
    print("LOOCV balanced accuracy:", round(bal, 4))
    print("LOOCV macro F1:", round(macro, 4))
    print(classification_report(
        y,
        pred,
        labels=[0, 1, 2],
        target_names=["Mild", "Moderate", "Severe"],
        zero_division=0,
    ))
    print("Confusion matrix (Mild, Moderate, Severe):")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))

result_df = pd.DataFrame(results).sort_values(
    "LOOCV_macro_f1", ascending=False
)
result_df.to_csv(OUTPUT / "clinical_loocv_results.csv", index=False)
print("\nSaved:", OUTPUT / "clinical_loocv_results.csv")
print(result_df.to_string(index=False))

