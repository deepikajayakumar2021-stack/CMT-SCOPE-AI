import os
import pandas as pd
import numpy as np

PROJECT_ROOT = r"D:\CMT_PROJECT"

GAIT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "gait_windows"
)

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "clinical_aligned_baseline.csv"
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "train_patients.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "val_patients.csv"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "test_patients.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "stage5_patient_manifest.csv"
)



print("=" * 70)
print("STAGE 5A: PATIENT-LEVEL MULTIMODAL MANIFEST")
print("=" * 70)

clinical = pd.read_csv(CLINICAL_FILE)

print("\nClinical rows loaded:", len(clinical))



required_columns = [
    "ID",
    "Status",
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration",
    "Cmtes_score",
    "severity_label",
    "severity_name"
]

missing_columns = [
    column
    for column in required_columns
    if column not in clinical.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )



clinical["ID"] = (
    clinical["ID"]
    .astype(str)
    .str.strip()
)



clinical = clinical[
    clinical["Status"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "CMT1A"
].copy()


print(
    "CMT1A clinical patients:",
    clinical["ID"].nunique()
)



train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)



for split_df in [train, val, test]:

    if "patient_id" not in split_df.columns:

        raise ValueError(
            "Stage 2 split file must contain 'patient_id'."
        )

    split_df["patient_id"] = (
        split_df["patient_id"]
        .astype(str)
        .str.strip()
    )


train_ids = set(train["patient_id"])
val_ids = set(val["patient_id"])
test_ids = set(test["patient_id"])



if train_ids & val_ids:
    raise ValueError("Train/Validation overlap detected.")

if train_ids & test_ids:
    raise ValueError("Train/Test overlap detected.")

if val_ids & test_ids:
    raise ValueError("Validation/Test overlap detected.")


all_split_ids = train_ids | val_ids | test_ids



clinical = clinical[
    clinical["ID"].isin(all_split_ids)
].copy()


print(
    "Patients aligned with Stage 2:",
    clinical["ID"].nunique()
)



clinical_ids = set(clinical["ID"])

missing_clinical = (
    all_split_ids - clinical_ids
)

if missing_clinical:

    raise ValueError(
        "Patients missing from clinical data: "
        + str(sorted(missing_clinical))
    )


def find_gait_file(patient_id):

    filename = (
        f"{patient_id}_gait.npz"
    )

    path = os.path.join(
        GAIT_DIR,
        filename
    )

    if os.path.isfile(path):

        return path

    return ""


clinical["gait_file"] = (
    clinical["ID"]
    .apply(find_gait_file)
)



missing_gait = clinical[
    clinical["gait_file"] == ""
]

print(
    "\nMissing gait files:",
    len(missing_gait)
)

if len(missing_gait) > 0:

    print(
        missing_gait["ID"].tolist()
    )

    raise ValueError(
        "Some Stage 2 patients do not have gait files."
    )



if clinical["ID"].duplicated().any():

    duplicates = clinical[
        clinical["ID"].duplicated(keep=False)
    ]["ID"].tolist()

    raise ValueError(
        "Duplicate patient IDs found: "
        + str(duplicates)
    )



def assign_split(patient_id):

    if patient_id in train_ids:
        return "train"

    if patient_id in val_ids:
        return "val"

    if patient_id in test_ids:
        return "test"

    return np.nan


clinical["split"] = (
    clinical["ID"]
    .apply(assign_split)
)



if clinical["split"].isna().any():

    raise ValueError(
        "Some patients have no split assignment."
    )



manifest = clinical[
    [
        "ID",
        "Status",
        "Age (Years)",
        "Gender",
        "Pinprick",
        "Vibration",
        "Cmtes_score",
        "severity_label",
        "severity_name",
        "split",
        "gait_file"
    ]
].copy()


# ============================================================
# SORT
# ============================================================

split_order = {
    "train": 0,
    "val": 1,
    "test": 2
}

manifest["split_order"] = (
    manifest["split"]
    .map(split_order)
)

manifest = (
    manifest
    .sort_values(
        [
            "split_order",
            "severity_label",
            "ID"
        ]
    )
    .drop(columns=["split_order"])
    .reset_index(drop=True)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PATIENT SPLIT")
print("=" * 70)

split_table = pd.crosstab(
    manifest["split"],
    manifest["severity_name"]
)

split_table = split_table.reindex(
    index=["train", "val", "test"],
    columns=[
        "Mild",
        "Moderate",
        "Severe"
    ],
    fill_value=0
)

print(split_table)


# ============================================================
# SAFETY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

print(
    "\nTotal patients:",
    len(manifest)
)

print(
    "Duplicate patient IDs:",
    manifest["ID"].duplicated().sum()
)

print(
    "Missing gait files:",
    (manifest["gait_file"] == "").sum()
)

print(
    "Train ∩ Validation:",
    len(train_ids & val_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(val_ids & test_ids)
)


# ============================================================
# CHECK ALL THREE CLASSES IN TRAIN
# ============================================================

train_classes = set(
    manifest.loc[
        manifest["split"] == "train",
        "severity_label"
    ]
)

if train_classes != {0, 1, 2}:

    raise ValueError(
        "Training split does not contain Mild, Moderate, Severe."
    )


# ============================================================
# SEVERITY INFORMATION
# ============================================================

print("\nSeverity mapping:")
print("0 = Mild")
print("1 = Moderate")
print("2 = Severe")


# ============================================================
# DATASET LIMITATION
# ============================================================

print("\nDataset limitation:")
print(
    "Only one Severe CMT1A patient is available."
)

print(
    "The Severe patient is in TRAIN so the model can learn "
    "the Severe class."
)

print(
    "Therefore, TEST contains no independent Severe patient."
)


# ============================================================
# OUTPUT
# ============================================================

print("\nSaved:")
print(OUTPUT_FILE)

print("\nSTAGE 5A COMPLETE")