# D:\CMT_PROJECT\scripts\improve_baseline.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(r"D:\CMT_PROJECT")
MANIFEST = ROOT / "processed" / "splits" / "stage5c_patient_manifest.csv"
GAIT_DIR = ROOT / "processed" / "gait_windows"

OUT_DIR = ROOT / "results" / "baseline_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. LOAD PATIENT MANIFEST
# --------------------------------------------------

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

# Match the column names used in your existing baseline.
id_col = next(
    (c for c in ["ID", "id", "patient_id", "subject_id"] if c in df.columns),
    None,
)
label_col = next(
    (c for c in ["severity_name", "severity", "Severity", "label"]
     if c in df.columns),
    None,
)

if id_col is None or label_col is None:
    raise ValueError(
        f"Could not find patient ID and severity columns.\n"
        f"Available columns: {list(df.columns)}"
    )

df = df[[id_col, label_col]].dropna().copy()
df[id_col] = df[id_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()

# Keep one consistent label per patient.
if df.groupby(id_col)[label_col].nunique().max() > 1:
    raise ValueError("A patient has multiple severity labels. Check the manifest.")

df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)

print("\nPatient label counts:")
print(df[label_col].value_counts())
print(f"\nPatients in manifest: {len(df)}")

# --------------------------------------------------
# 3. EXTRACT FEATURES FROM EACH GAIT WINDOW
# --------------------------------------------------

def window_features(window):
    """
    Input: one window with shape (time_steps, 3 axes).
    Output: 36 features = 12 features for each axis.
    """
    result = []

    for axis in range(3):
        x = np.asarray(window[:, axis], dtype=np.float64)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            result.extend([0.0] * 12)
            continue

        mean = np.mean(x)
        std = np.std(x)
        centered = x - mean
        rms = np.sqrt(np.mean(x ** 2))

        # Stable moment calculations when the signal is nearly constant.
        if std > 1e-10:
            z = centered / std
            skew = np.mean(z ** 3)
            kurtosis = np.mean(z ** 4)
        else:
            skew = 0.0
            kurtosis = 0.0

        zero_crossing = np.mean(
            np.signbit(x[1:]) != np.signbit(x[:-1])
        ) if len(x) > 1 else 0.0

        result.extend([
            mean,
            std,
            np.median(x),
            np.min(x),
            np.max(x),
            np.max(x) - np.min(x),
            rms,
            np.mean(np.abs(x)),
            np.percentile(x, 25),
            np.percentile(x, 75),
            zero_crossing,
            skew,
        ])

        # Replace the final feature with kurtosis, keeping 12 per axis.
        result[-1] = kurtosis

    return np.asarray(result, dtype=np.float64)


def load_gait_array(path):
    """Find a 3-axis gait array in an NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        candidates = []

        for key in data.files:
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                candidates.append(arr)
            elif arr.ndim == 3 and arr.shape[1] == 3:
                candidates.append(np.transpose(arr, (0, 2, 1)))

        if not candidates:
            raise ValueError(
                f"No 3D, three-axis gait array found in {path.name}. "
                f"Available arrays: {data.files}"
            )

        # Expected format: (windows, time_steps, 3).
        arr = candidates[0]

        if arr.shape[-1] != 3:
            raise ValueError(f"Unexpected gait shape {arr.shape} in {path.name}")

        return np.asarray(arr, dtype=np.float32)


def patient_features(gait):
    """
    First compute window features, then summarize their distribution
    for the patient. Each patient produces one feature vector.
    """
    all_window_features = []

    for window in gait:
        if window.shape[0] < 2:
            continue

        feat = window_features(window)

        if np.all(np.isfinite(feat)):
            all_window_features.append(feat)

    if not all_window_features:
        raise ValueError("No valid gait windows found.")

    features = np.vstack(all_window_features)

    # 36 window features summarized by mean, standard deviation,
    # and median = 108 patient-level features.
    summary = np.concatenate([
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.median(features, axis=0),
    ])

    return summary


# --------------------------------------------------
# 4. LOAD ONE FEATURE VECTOR PER PATIENT
# --------------------------------------------------

X_rows = []
y_rows = []
patient_ids = []

missing = []
failed = []

for _, row in df.iterrows():
    patient_id = row[id_col]
    label = row[label_col]

    # Example expected filename: ROC-011_gait.npz
    path = GAIT_DIR / f"{patient_id}_gait.npz"

    if not path.exists():
        missing.append(patient_id)
        continue

    try:
        gait = load_gait_array(path)
        features = patient_features(gait)

        X_rows.append(features)
        y_rows.append(label)
        patient_ids.append(patient_id)

        print(
            f"Loaded {patient_id}: "
            f"{gait.shape[0]} windows, {len(features)} features"
        )

    except Exception as exc:
        failed.append((patient_id, str(exc)))
        print(f"Skipped {patient_id}: {exc}")

if missing:
    print("\nMissing gait files:", missing)

if failed:
    print("\nFiles that could not be processed:")
    for patient_id, reason in failed:
        print(f"  {patient_id}: {reason}")

if len(X_rows) < 3:
    raise RuntimeError(
        "Too few patients loaded. Check gait filenames and manifest IDs."
    )

X = np.asarray(X_rows, dtype=np.float64)
y = np.asarray(y_rows)
patient_ids = np.asarray(patient_ids)

if not np.all(np.isfinite(X)):
    raise ValueError("Feature matrix contains NaN or infinite values.")

print("\nFinal evaluation dataset")
print("Patients:", len(y))
print("Features per patient:", X.shape[1])
print("Labels:", dict(zip(*np.unique(y, return_counts=True))))

# --------------------------------------------------
# 5. DEFINE CANDIDATE MODELS
# --------------------------------------------------

def make_pipeline(model, k=None, scale=False):
    steps = []

    if scale:
        steps.append(("scaler", StandardScaler()))

    if k is not None:
        steps.append((
            "select",
            SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        ))

    steps.append(("model", model))
    return Pipeline(steps)


candidates = {
    "ExtraTrees_Default": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),

    "ExtraTrees_Leaf2": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),

    "ExtraTrees_Leaf3": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),

    "ExtraTrees_Balanced": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),

    "ExtraTrees_Top10": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        k=10,
    ),

    "ExtraTrees_Top20": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        k=20,
    ),

    "ExtraTrees_Top30": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        k=30,
    ),

    "ExtraTrees_Top50": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        k=50,
    ),

    "RandomForest": make_pipeline(
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),

    "SVM_RBF": make_pipeline(
        SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
        ),
        scale=True,
    ),

    "SVM_RBF_C10_Top20": make_pipeline(
        SVC(
            C=10.0,
            kernel="rbf",
            class_weight="balanced",
        ),
        k=20,
        scale=True,
    ),
}

# --------------------------------------------------
# 6. LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION
# --------------------------------------------------

results = []
prediction_tables = []

for model_name, model in candidates.items():
    print(f"\nEvaluating: {model_name}")

    predictions = []

    for test_idx in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_idx:test_idx + 1]

        # Some folds may have no examples of a class, such as Severe.
        # The model cannot learn a class absent from its training fold.
        if len(np.unique(y_train)) < 2:
            pred = pd.Series(y_train).mode().iloc[0]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

        predictions.append(pred)

    predictions = np.asarray(predictions)

    acc = accuracy_score(y, predictions)
    bal_acc = balanced_accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)

    results.append({
        "Model": model_name,
        "Patients": len(y),
        "Correct": int(np.sum(predictions == y)),
        "Accuracy_Percent": round(acc * 100, 2),
        "Balanced_Accuracy_Percent": round(bal_acc * 100, 2),
        "Macro_F1_Percent": round(macro_f1 * 100, 2),
    })

    prediction_tables.append(pd.DataFrame({
        "Patient_ID": patient_ids,
        "Actual": y,
        "Predicted": predictions,
        "Correct": predictions == y,
        "Model": model_name,
    }))

    print(
        f"Accuracy: {acc * 100:.2f}% | "
        f"Balanced accuracy: {bal_acc * 100:.2f}% | "
        f"Macro F1: {macro_f1 * 100:.2f}%"
    )

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    by=["Balanced_Accuracy_Percent", "Macro_F1_Percent", "Accuracy_Percent"],
    ascending=False,
)

predictions_df = pd.concat(prediction_tables, ignore_index=True)

metrics_path = OUT_DIR / "tuning_metrics.csv"
predictions_path = OUT_DIR / "tuning_patient_predictions.csv"

results_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

print("\nSaved:")
print(metrics_path)
print(predictions_path)

best = results_df.iloc[0]
print("\nBest by balanced accuracy, then macro F1, then accuracy:")
print(best.to_string())

print(
    "\nReminder: this is LOOCV on the currently loaded labelled patients. "
    "It is not an independent external test, and results do not guarantee "
    "performance on new patients."
)