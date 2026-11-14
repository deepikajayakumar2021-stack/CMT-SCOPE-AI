
import os
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import pearsonr, spearmanr



PROJECT_ROOT = r"D:\CMT_PROJECT"

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "stage5_patient_manifest.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

TARGET = "CMT-FOM_Total"

CLINICAL_FEATURES = [
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration"
]


if not os.path.exists(MANIFEST_FILE):
    raise FileNotFoundError(
        f"\nManifest not found:\n{MANIFEST_FILE}"
    )

manifest = pd.read_csv(MANIFEST_FILE)

print("=" * 70)
print("CMT1A RANDOM FOREST BASELINE")
print("=" * 70)

print(
    "\nTotal patients:",
    len(manifest)
)

print(
    "Train patients:",
    len(manifest[manifest["split"] == "train"])
)

print(
    "Validation patients:",
    len(manifest[manifest["split"] == "val"])
)

print(
    "Test patients:",
    len(manifest[manifest["split"] == "test"])
)



def extract_gait_features(gait_file):

    data = np.load(
        gait_file,
        allow_pickle=True
    )

    X = data["X"].astype(np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Unexpected gait shape: {X.shape}\n"
            f"Expected: (windows, 256, 3)"
        )

    # --------------------------------------------------------
    # Combine all windows into one patient-level signal
    # --------------------------------------------------------

    signal = X.reshape(-1, 3)

    acc_x = signal[:, 0]
    acc_y = signal[:, 1]
    acc_z = signal[:, 2]

    acc_mag = np.sqrt(
        acc_x ** 2 +
        acc_y ** 2 +
        acc_z ** 2
    )

    signals = {
        "ACC_X": acc_x,
        "ACC_Y": acc_y,
        "ACC_Z": acc_z,
        "ACC_MAG": acc_mag
    }

    features = {}

    for signal_name, values in signals.items():

        features[f"{signal_name}_mean"] = np.mean(values)

        features[f"{signal_name}_std"] = np.std(values)

        features[f"{signal_name}_min"] = np.min(values)

        features[f"{signal_name}_max"] = np.max(values)

        features[f"{signal_name}_median"] = np.median(values)

        features[f"{signal_name}_q25"] = np.percentile(
            values,
            25
        )

        features[f"{signal_name}_q75"] = np.percentile(
            values,
            75
        )

        features[f"{signal_name}_rms"] = np.sqrt(
            np.mean(values ** 2)
        )

    return features



rows = []

print()
print("=" * 70)
print("EXTRACTING PATIENT-LEVEL FEATURES")
print("=" * 70)

for _, row in manifest.iterrows():

    patient_id = str(row["ID"])

    gait_file = row["gait_file"]

    if not os.path.exists(gait_file):
        raise FileNotFoundError(
            f"\nGait file not found for {patient_id}:\n"
            f"{gait_file}"
        )

    gait_features = extract_gait_features(
        gait_file
    )

    patient_features = {
        "ID": patient_id,
        "split": row["split"],
        TARGET: row[TARGET]
    }

    # Add clinical features
    for feature in CLINICAL_FEATURES:

        patient_features[feature] = row[feature]

    # Add gait features
    patient_features.update(
        gait_features
    )

    rows.append(patient_features)

feature_df = pd.DataFrame(rows)


print(
    "\nFeature dataset shape:",
    feature_df.shape
)

print(
    "Number of features:",
    len(
        feature_df.columns
    ) - 3
)


 

FEATURE_COLUMNS = [
    column
    for column in feature_df.columns
    if column not in [
        "ID",
        "split",
        TARGET
    ]
]

print(
    "\nFeatures used by Random Forest:"
)

for feature in FEATURE_COLUMNS:
    print(
        " -",
        feature
    )


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df = feature_df[
    feature_df["split"] == "train"
].copy()

val_df = feature_df[
    feature_df["split"] == "val"
].copy()

test_df = feature_df[
    feature_df["split"] == "test"
].copy()


X_train = train_df[
    FEATURE_COLUMNS
].apply(
    pd.to_numeric,
    errors="coerce"
)

y_train = pd.to_numeric(
    train_df[TARGET],
    errors="coerce"
)

X_val = val_df[
    FEATURE_COLUMNS
].apply(
    pd.to_numeric,
    errors="coerce"
)

y_val = pd.to_numeric(
    val_df[TARGET],
    errors="coerce"
)

X_test = test_df[
    FEATURE_COLUMNS
].apply(
    pd.to_numeric,
    errors="coerce"
)

y_test = pd.to_numeric(
    test_df[TARGET],
    errors="coerce"
)


#  MEDIAN IMPUTATION

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_val = imputer.transform(
    X_val
)

X_test = imputer.transform(
    X_test
)


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


y_pred = model.predict(
    X_test
)


#  FUNCTION
# 

def calculate_metrics(
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    # Pearson
    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(predictions) > 0
    ):
        pearson_r, _ = pearsonr(
            y_true,
            predictions
        )
    else:
        pearson_r = np.nan

    
    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(predictions) > 0
    ):
        spearman_rho, _ = spearmanr(
            y_true,
            predictions
        )
    else:
        spearman_rho = np.nan

    return (
        mae,
        rmse,
        r2,
        pearson_r,
        spearman_rho
    )


(
    mae,
    rmse,
    r2,
    pearson_r,
    spearman_rho
) = calculate_metrics(
    y_test.values,
    y_pred
)



print()
print("=" * 70)
print("RANDOM FOREST CMT-FOM REGRESSION METRICS")
print("=" * 70)

print(
    f"MAE        : {mae:.4f}"
)

print(
    f"RMSE       : {rmse:.4f}"
)

print(
    f"R²         : {r2:.4f}"
)

print(
    f"Pearson r  : {pearson_r:.4f}"
)

print(
    f"Spearman ρ : {spearman_rho:.4f}"
)



prediction_df = pd.DataFrame({
    "Patient_ID": test_df["ID"].values,
    "Actual_CMT_FOM": y_test.values,
    "Predicted_CMT_FOM": y_pred
})

prediction_df["Error"] = (
    prediction_df["Actual_CMT_FOM"]
    - prediction_df["Predicted_CMT_FOM"]
)

prediction_df["Absolute_Error"] = (
    np.abs(
        prediction_df["Error"]
    )
)


print()
print("=" * 70)
print("RANDOM FOREST PATIENT-LEVEL PREDICTIONS")
print("=" * 70)

print(
    prediction_df.to_string(
        index=False
    )
)



importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


print()
print("=" * 70)
print("TOP RANDOM FOREST FEATURES")
print("=" * 70)

print(
    importance_df.head(15).to_string(
        index=False
    )
)



prediction_file = os.path.join(
    RESULTS_DIR,
    "random_forest_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)




metrics_df = pd.DataFrame({
    "Model": [
        "Random Forest Baseline"
    ],
    "Test_Patients": [
        len(y_test)
    ],
    "MAE": [
        mae
    ],
    "RMSE": [
        rmse
    ],
    "R2": [
        r2
    ],
    "Pearson_r": [
        pearson_r
    ],
    "Spearman_rho": [
        spearman_rho
    ]
})

metrics_file = os.path.join(
    RESULTS_DIR,
    "random_forest_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)




importance_file = os.path.join(
    RESULTS_DIR,
    "random_forest_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)




print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    prediction_file
)

print(
    metrics_file
)

print(
    importance_file
)

print()
print("=" * 70)
print("RANDOM FOREST BASELINE COMPLETE")
print("=" * 70)