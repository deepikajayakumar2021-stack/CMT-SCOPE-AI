
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)


ROOT = Path(r"D:\CMT_PROJECT")
INPUT = ROOT / "results" / "random_forest_baseline" / "patient_features.csv"
OUT = ROOT / "results" / "svm_baseline"

OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Mild", "Moderate", "Severe"]

if not INPUT.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{INPUT}\n"
        "Run the patient-level feature extraction script first."
    )

df = pd.read_csv(INPUT)

print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())


target_candidates = [
    "severity_label",
    "severity",
    "severity_class",
    "actual_severity",
    "label",
    "target",
    "class"
]

target_col = next(
    (c for c in target_candidates if c in df.columns),
    None
)

if target_col is None:
    raise ValueError(
        "Could not identify the severity label column. "
        "Check the printed column names and set target_col manually."
    )

def normalize_label(value):
    value = str(value).strip().lower()

    mapping = {
        "0": "Mild", "0.0": "Mild",
        "class 0": "Mild", "mild": "Mild",
        "1": "Moderate", "1.0": "Moderate",
        "class 1": "Moderate", "moderate": "Moderate",
        "2": "Severe", "2.0": "Severe",
        "class 2": "Severe", "severe": "Severe"
    }

    return mapping.get(value)

labels = df[target_col].apply(normalize_label)

if labels.isna().any():
    bad = df.loc[labels.isna(), target_col].unique()
    raise ValueError(
        f"Unrecognized severity labels: {bad}. "
        "Check the label encoding before continuing."
    )



exclude_words = [
    "patient_id", "subject_id", "recording_id",
    "filename", "filepath", "path", "roc",
    "severity", "label", "target", "class",
    "cmtes", "cmt_es", "cmt-fom"
]

feature_cols = []

for col in df.columns:
    if col == target_col:
        continue

    name = col.lower()

    if any(word in name for word in exclude_words):
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        feature_cols.append(col)

if not feature_cols:
    raise ValueError("No usable numeric feature columns were found.")

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = labels.reset_index(drop=True)

if len(y) < 3:
    raise ValueError("Not enough labelled patients for evaluation.")

if y.nunique() < 2:
    raise ValueError("At least two severity classes are required.")

print("\nPatients:", len(y))
print("\nClass distribution:")
print(y.value_counts())

print("\nFeatures used:")
print(feature_cols)


model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
)

loo = LeaveOneOut()
predictions = []

for train_idx, test_idx in loo.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Every training fold must contain at least two classes.
    if y_train.nunique() < 2:
        raise ValueError(
            "A training fold contains fewer than two classes."
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]

    predictions.append({
        "Patient_Row": int(test_idx[0]),
        "Actual": y_test.iloc[0],
        "Predicted": pred,
        "Correct": pred == y_test.iloc[0]
    })

result = pd.DataFrame(predictions)

# ============================================================
# METRICS
# ============================================================

y_true = result["Actual"]
y_pred = result["Predicted"]

accuracy = accuracy_score(y_true, y_pred) * 100
macro_f1 = f1_score(
    y_true, y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred,
    labels=CLASSES
)

print("\n====================================")
print(" CMT-SCOPE AI - SVM BASELINE")
print("====================================")
print(f"Evaluated patients : {len(result)}")
print(f"Correct predictions: {int(result['Correct'].sum())}")
print(f"LOOCV accuracy     : {accuracy:.2f}%")
print(f"Macro F1-score     : {macro_f1:.4f}")
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=CLASSES,
    zero_division=0
))

# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(OUT / "loocv_predictions.csv", index=False)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(OUT / "confusion_matrix.csv")

pd.DataFrame([{
    "Model": "Class-weighted RBF SVM",
    "Evaluation": "Leave-One-Out patient-level CV",
    "Patients": len(result),
    "Correct": int(result["Correct"].sum()),
    "Accuracy_Percent": accuracy,
    "Macro_F1": macro_f1
}]).to_csv(OUT / "metrics.csv", index=False)

print("\nSaved results to:", OUT)
