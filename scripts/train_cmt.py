
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"D:\CMT_PROJECT")
GAIT_DIR = ROOT / "processed" / "gait_windows"
OUTPUT_DIR = ROOT / "results" / "accuracy_improvement"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_NAMES = {
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}

ID_NAMES = {
    "id", "patient_id", "subject_id", "participant_id",
    "patient", "subject", "participant", "record_id"
}

LABEL_NAMES = {
    "severity", "severity_label", "severity_class",
    "severity_group", "class", "class_label",
    "disease_severity", "cmt_severity",
    "severity_category", "stage"
}


# ============================================================
# HELPERS
# ============================================================

def normalize_column(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def normalize_patient_id(value):
    """Normalize IDs such as ROC-011, ROC_011 and ROC-011_gait."""
    text = str(value).strip().upper()
    text = re.sub(r"_GAIT$", "", text)
    match = re.search(r"ROC[\s_-]*0*(\d+)", text)

    if match:
        return f"ROC-{int(match.group(1)):03d}"

    return None


def normalize_severity(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z]", "", text)

    # Accept explicit severity labels only.
    aliases = {
        "mild": "Mild",
        "mildlyaffected": "Mild",
        "moderate": "Moderate",
        "moderatelyaffected": "Moderate",
        "severe": "Severe",
        "severelyaffected": "Severe",
    }

    return aliases.get(text)


# ============================================================
# LOAD REAL SEVERITY LABELS
# ============================================================

def load_labels():
    label_map = {}
    sources = []

    files = list(ROOT.rglob("*.csv"))
    files += list(ROOT.rglob("*.xlsx"))
    files += list(ROOT.rglob("*.xls"))

    excluded_parts = {
        "results", "output", "outputs", "predictions",
        "baseline_comparison", "accuracy_improvement"
    }

    for file in files:
        if any(part.lower() in excluded_parts
               for part in file.relative_to(ROOT).parts[:-1]):
            continue

        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            if df.empty:
                continue

            original_columns = list(df.columns)
            normalized = {
                col: normalize_column(col)
                for col in original_columns
            }

            id_columns = [
                col for col in original_columns
                if normalized[col] in ID_NAMES
                or normalized[col].endswith("_patient_id")
                or normalized[col].endswith("_subject_id")
            ]

            label_columns = [
                col for col in original_columns
                if normalized[col] in LABEL_NAMES
                or "severity" in normalized[col]
            ]

            # Check candidate columns for actual Mild/Moderate/Severe values.
            for label_col in label_columns:
                for id_col in id_columns:
                    found = 0

                    for _, row in df[[id_col, label_col]].dropna().iterrows():
                        patient_id = normalize_patient_id(row[id_col])
                        severity = normalize_severity(row[label_col])

                        if patient_id and severity:
                            if patient_id in label_map:
                                if label_map[patient_id] != severity:
                                    print(
                                        "WARNING: Conflicting labels for",
                                        patient_id,
                                        "from", file
                                    )
                                    label_map.pop(patient_id, None)
                                    continue

                            label_map[patient_id] = severity
                            found += 1

                    if found:
                        sources.append(
                            f"{file.name}: ID={id_col}, label={label_col}"
                        )

        except Exception as exc:
            print(f"Skipped {file.name}: {exc}")

    print("\nLabel sources discovered:")
    for source in sorted(set(sources)):
        print("  ", source)

    print("\nUnique patient labels:", len(label_map))
    print(pd.Series(label_map).value_counts().to_string()
          if label_map else "No recognized labels found.")

    return label_map


# ============================================================
# GAIT FEATURE EXTRACTION
# ============================================================

def extract_features(windows):
    """
    Input shape: (number_of_windows, 256, 3)
    Output: one feature vector for the patient.
    """
    x = np.asarray(windows, dtype=np.float64)

    if x.ndim == 2 and x.shape[-1] == 3:
        x = x[np.newaxis, :, :]

    if x.ndim != 3:
        raise ValueError(f"Expected 3D gait array, got {x.shape}")

    if x.shape[-1] != 3 and x.shape[1] == 3:
        x = np.transpose(x, (0, 2, 1))

    if x.shape[-1] != 3:
        raise ValueError(f"Could not identify 3 acceleration axes: {x.shape}")

    x = x[np.isfinite(x).all(axis=(1, 2))]

    if len(x) == 0:
        raise ValueError("No finite gait windows")

    # Window-level descriptors, aggregated across windows.
    window_features = []

    for w in x:
        features = []

        for axis in range(3):
            signal = w[:, axis]
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.sqrt(np.mean(signal ** 2)),
                np.mean(np.abs(np.diff(signal))),
                np.std(np.diff(signal)),
            ])

            # Frequency-domain descriptors.
            centered = signal - np.mean(signal)
            power = np.abs(np.fft.rfft(centered)) ** 2

            if len(power) > 1:
                power = power[1:]  # exclude zero-frequency component
                total_power = power.sum() + 1e-12
                probabilities = power / total_power
                frequencies = np.arange(1, len(power) + 1)

                features.extend([
                    np.sum(power),
                    np.sum(frequencies * probabilities),
                    -np.sum(
                        probabilities * np.log(probabilities + 1e-12)
                    ),
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

        magnitude = np.sqrt(np.sum(w ** 2, axis=1))
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.sqrt(np.mean(magnitude ** 2)),
            np.mean(np.abs(np.diff(magnitude))),
        ])

        window_features.append(features)

    window_features = np.asarray(window_features)

    # A single patient-level vector prevents window-level leakage.
    patient_features = np.concatenate([
        np.mean(window_features, axis=0),
        np.std(window_features, axis=0),
        np.median(window_features, axis=0),
    ])

    return patient_features


# ============================================================
# LOAD ONE FEATURE VECTOR PER PATIENT
# ============================================================

def load_patient_data(label_map):
    X, y, patient_ids = [], [], []

    if not GAIT_DIR.exists():
        raise FileNotFoundError(f"Gait directory not found: {GAIT_DIR}")

    files = sorted(GAIT_DIR.glob("*.npz"))

    for file in files:
        patient_id = normalize_patient_id(file.stem)

        if not patient_id or patient_id not in label_map:
            continue

        try:
            with np.load(file, allow_pickle=False) as data:
                keys = list(data.keys())

                # Prefer common names; otherwise select a numeric 3D array.
                preferred = [
                    "windows", "X", "gait_windows", "arr_0", "data"
                ]
                key = next(
                    (k for k in preferred if k in keys),
                    None
                )

                if key is None:
                    candidates = []
                    for k in keys:
                        arr = data[k]
                        if arr.ndim == 3 and (
                            arr.shape[-1] == 3 or arr.shape[1] == 3
                        ):
                            candidates.append(k)

                    if not candidates:
                        print(f"Skipped {file.name}: keys/shapes "
                              f"{[(k, data[k].shape) for k in keys]}")
                        continue

                    key = candidates[0]

                windows = data[key]
                features = extract_features(windows)

            X.append(features)
            y.append(label_map[patient_id])
            patient_ids.append(patient_id)

        except Exception as exc:
            print(f"Skipped {file.name}: {exc}")

    if not X:
        raise RuntimeError(
            "No labelled gait files matched the clinical labels.\n"
            "Check your patient IDs, severity column, and gait file paths."
        )

    # Detect conflicting duplicate patient IDs across filenames.
    if len(patient_ids) != len(set(patient_ids)):
        duplicates = pd.Series(patient_ids)
        duplicates = duplicates[duplicates.duplicated()].unique().tolist()
        raise RuntimeError(
            "Multiple gait files map to the same patient: "
            + ", ".join(duplicates)
            + ". Combine that patient's recordings before evaluation."
        )

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y),
        np.asarray(patient_ids)
    )


# ============================================================
# PATIENT-INDEPENDENT MODEL COMPARISON
# ============================================================

def evaluate_models(X, y, patient_ids):
    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        ),
        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        ),
        "SVM RBF": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=1.0,
                kernel="rbf",
                class_weight="balanced"
            )
        ),
    }

    # Leave-one-patient-out: each patient is tested only when unseen.
    loo = LeaveOneOut()
    results = []
    predictions_by_model = {}

    for name, model in models.items():
        y_true, y_pred = [], []

        for train_idx, test_idx in loo.split(X, y):
            # One class in the training fold makes classification invalid.
            if len(np.unique(y[train_idx])) < 2:
                y_true.append(y[test_idx[0]])
                y_pred.append("__NO_TRAINING_CLASS__")
                continue

            fold_model = clone(model)
            fold_model.fit(X[train_idx], y[train_idx])
            pred = fold_model.predict(X[test_idx])[0]

            y_true.append(y[test_idx][0])
            y_pred.append(pred)

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=list(SEVERITY_NAMES.values()),
            average="macro", zero_division=0
        )
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        results.append({
            "Model": name,
            "Overall_Accuracy_Percent": round(accuracy * 100, 2),
            "Macro_F1_Percent": round(macro_f1 * 100, 2),
            "Balanced_Accuracy_Percent": round(balanced_acc * 100, 2),
        })

        predictions_by_model[name] = (y_true, y_pred)

        print(f"\n{name}")
        print(f"Overall accuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1:         {macro_f1 * 100:.2f}%")
        print(f"Balanced accuracy:{balanced_acc * 100:.2f}%")

        print("\nClassification report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=list(SEVERITY_NAMES.values()),
            zero_division=0
        ))

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(SEVERITY_NAMES.values())
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{c}" for c in SEVERITY_NAMES.values()],
            columns=[f"Predicted_{c}" for c in SEVERITY_NAMES.values()]
        )
        cm_df.to_csv(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion.csv"
        )

        pd.DataFrame({
            "Patient_ID": patient_ids,
            "Actual": y_true,
            "Predicted": y_pred
        }).to_csv(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_predictions.csv",
            index=False
        )

    results_df = pd.DataFrame(results).sort_values(
        "Overall_Accuracy_Percent", ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n================ MODEL COMPARISON ================")
    print(results_df.to_string(index=False))

    # Fit the top-performing model on all labelled patients for later use.
    # This final model is not an independent test result.
    best_name = results_df.iloc[0]["Model"]
    final_model = clone(models[best_name])
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "feature_count": X.shape[1],
            "classes": list(final_model.classes_),
            "note": "Fit on all labelled patients; not an evaluation model."
        },
        OUTPUT_DIR / "best_cmt_model.joblib"
    )

    print(f"\nBest measured baseline: {best_name}")
    print("Saved model: ", OUTPUT_DIR / "best_cmt_model.joblib")
    print("Saved results:", OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    labels = load_labels()

    if len(labels) < 3:
        raise SystemExit(
            "\nSTOP: Not enough real severity labels were found. "
            "No model was trained. Inspect your clinical CSV/Excel files "
            "and configure the label column names at the top of this script."
        )

    X, y, patient_ids = load_patient_data(labels)

    print("\nMatched labelled patients:", len(patient_ids))
    print("Class distribution:")
    print(pd.Series(y).value_counts().to_string())

    if len(np.unique(y)) != 3:
        raise SystemExit(
            "\nSTOP: All three severity classes must be present. "
            "Check the clinical labels and gait-file matches."
        )

    evaluate_models(X, y, patient_ids)
