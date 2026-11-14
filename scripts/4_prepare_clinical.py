
from pathlib import Path
import pandas as pd
import numpy as np




ROOT = Path(r"D:\CMT_PROJECT")
SPLIT_DIR = ROOT / "processed" / "splits"

INPUT_FILE = SPLIT_DIR / "clinical_aligned_baseline.csv"

TRAIN_FILE = SPLIT_DIR / "train_patients.csv"
VAL_FILE = SPLIT_DIR / "val_patients.csv"
TEST_FILE = SPLIT_DIR / "test_patients.csv"

OUTPUT_FILE = SPLIT_DIR / "clinical_model_ready.csv"



def map_severity(score):
    if pd.isna(score):
        return np.nan

    score = float(score)

    if 0 <= score <= 7:
        return 0

    elif 8 <= score <= 16:
        return 1

    elif 17 <= score <= 28:
        return 2

    else:
        return np.nan


severity_names = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}



print("=" * 70)
print("CMT1A STAGE 4: CLINICAL FEATURE PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nInput shape:", df.shape)


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMN
# ------------------------------------------------------------

required_columns = [
    "ID",
    "Status",
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration",
    "Foot_plantarflexion_left (Newtons)",
    "Foot_dorsiflexion_left (Newtons)",
    "Hip_flexion_left (MRC 0-5)",
    "Knee_extension_left (MRC 0-5)",
    "Ankle_dorsiflexion_left (MRC 0-5)",
    "Hip_abduction_left (MRC 0-5)",
    "Hip_extension_left (MRC 0-5)",
    "Knee_flexion_left (MRC 0-5)",
    "Ankle_plantarflexion_left (MRC 0-5)",
    "Cmtes_score",
    "CMT-FOM_Total"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required clinical columns:\n" + "\n".join(missing)
    )



df["ID"] = df["ID"].astype(str).str.strip()

df = df[df["Status"].astype(str).str.strip().str.upper() == "CMT1A"].copy()

print("CMT1A rows:", len(df))



train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

# Stage 2 uses patient_id
for split_df in [train, val, test]:

    if "patient_id" not in split_df.columns:
        raise ValueError(
            "Expected 'patient_id' column in Stage 2 split file."
        )

    split_df["patient_id"] = (
        split_df["patient_id"]
        .astype(str)
        .str.strip()
    )


train_ids = set(train["patient_id"])
val_ids = set(val["patient_id"])
test_ids = set(test["patient_id"])



all_split_ids = train_ids | val_ids | test_ids

clinical_ids = set(df["ID"])

missing_from_clinical = all_split_ids - clinical_ids

if missing_from_clinical:
    raise ValueError(
        "Patients in Stage 2 split but missing from clinical data: "
        + str(sorted(missing_from_clinical))
    )



df = df[df["ID"].isin(all_split_ids)].copy()

print("Patients aligned with Stage 2:", df["ID"].nunique())



df["severity_label"] = df["Cmtes_score"].apply(map_severity)

df["severity_name"] = df["severity_label"].map(severity_names)


# ------------------------------------------------------------
# 10. CHECK VALID SEVERITY LABELS
# ------------------------------------------------------------

if df["severity_label"].isna().any():

    bad_rows = df[df["severity_label"].isna()][
        ["ID", "Cmtes_score"]
    ]

    raise ValueError(
        "Invalid or missing Cmtes_score values:\n"
        + str(bad_rows)
    )


df["severity_label"] = df["severity_label"].astype(int)




def assign_split(patient_id):

    if patient_id in train_ids:
        return "train"

    elif patient_id in val_ids:
        return "val"

    elif patient_id in test_ids:
        return "test"

    return np.nan


df["split"] = df["ID"].apply(assign_split)



clinical_features = [
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration",
    "Foot_plantarflexion_left (Newtons)",
    "Foot_dorsiflexion_left (Newtons)",
    "Hip_flexion_left (MRC 0-5)",
    "Knee_extension_left (MRC 0-5)",
    "Ankle_dorsiflexion_left (MRC 0-5)",
    "Hip_abduction_left (MRC 0-5)",
    "Hip_extension_left (MRC 0-5)",
    "Knee_flexion_left (MRC 0-5)",
    "Ankle_plantarflexion_left (MRC 0-5)"
]



df["Gender"] = (
    df["Gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "female": 0,
        "male": 1
    })
)




missing_counts = df[clinical_features].isna().sum()

print("\nMissing clinical values:")

print(missing_counts[missing_counts > 0])




output_columns = [
    "ID",
    "split"
] + clinical_features + [
    "Cmtes_score",
    "severity_label",
    "severity_name"
]

output = df[output_columns].copy()



if output["ID"].duplicated().any():

    duplicates = output[
        output["ID"].duplicated(keep=False)
    ]["ID"].tolist()

    raise ValueError(
        "Duplicate patient IDs found: " + str(duplicates)
    )


if output["split"].isna().any():
    raise ValueError("Some patients have no train/val/test split.")



output.to_csv(OUTPUT_FILE, index=False)



print("\n" + "=" * 70)
print("STAGE 4 COMPLETE")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nOutput shape:", output.shape)

print("\nSeverity distribution:")
print(
    output["severity_name"]
    .value_counts()
    .reindex(["Mild", "Moderate", "Severe"])
    .fillna(0)
    .astype(int)
)

print("\nSplit distribution:")
print(
    output.groupby(["split", "severity_name"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["Mild", "Moderate", "Severe"], fill_value=0)
)

print("\nFinal columns:")
print(output.columns.tolist())

print("\nSaved successfully.")