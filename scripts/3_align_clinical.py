from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path(r"D:\CMT_PROJECT")

PROCESSED_DIR = PROJECT_DIR / "processed"
GAIT_DIR = PROCESSED_DIR / "gait_windows"
SPLIT_DIR = PROCESSED_DIR / "splits"

CLINICAL_FILE = PROCESSED_DIR / "clinical_baseline.csv"

TRAIN_FILE = SPLIT_DIR / "train_patients.csv"
VAL_FILE = SPLIT_DIR / "val_patients.csv"
TEST_FILE = SPLIT_DIR / "test_patients.csv"

OUTPUT_FILE = SPLIT_DIR / "clinical_aligned_baseline.csv"


print("=" * 70)
print("CMT1A STAGE 3: CLINICAL + GAIT ALIGNMENT")
print("=" * 70)


required_files = [
    CLINICAL_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )



train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)


# Stage 2 uses patient_id
for df_split in [train, val, test]:

    if "patient_id" not in df_split.columns:

        raise ValueError(
            "Expected 'patient_id' column in Stage 2 split files."
        )

    df_split["patient_id"] = (
        df_split["patient_id"]
        .astype(str)
        .str.strip()
    )


train_ids = set(train["patient_id"])
val_ids = set(val["patient_id"])
test_ids = set(test["patient_id"])


verified_ids = sorted(
    train_ids |
    val_ids |
    test_ids
)


print("\nStage 2 verified patients:", len(verified_ids))



if len(train_ids & val_ids) > 0:

    raise ValueError(
        "Train and validation patients overlap."
    )

if len(train_ids & test_ids) > 0:

    raise ValueError(
        "Train and test patients overlap."
    )

if len(val_ids & test_ids) > 0:

    raise ValueError(
        "Validation and test patients overlap."
    )


print("✅ Stage 2 patient split has no overlap.")



clinical = pd.read_csv(CLINICAL_FILE)

print("\nClinical dataset shape:", clinical.shape)


if "ID" not in clinical.columns:

    raise ValueError(
        "ID column is missing from clinical_baseline.csv."
    )

if "Status" not in clinical.columns:

    raise ValueError(
        "Status column is missing from clinical_baseline.csv."
    )

if "Time" not in clinical.columns:

    raise ValueError(
        "Time column is missing from clinical_baseline.csv."
    )


clinical["ID"] = (
    clinical["ID"]
    .astype(str)
    .str.strip()
)


baseline = clinical[
    clinical["Time"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "baseline"
].copy()


baseline_cmt1a = baseline[
    baseline["Status"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "CMT1A"
].copy()


print("Baseline CMT1A records:", len(baseline_cmt1a))



if "Cmtes_score" not in baseline_cmt1a.columns:

    raise ValueError(
        "Cmtes_score column is required."
    )


baseline_cmt1a = baseline_cmt1a[
    baseline_cmt1a["Cmtes_score"].notna()
].copy()


print(
    "Baseline CMT1A records with valid CMTES:",
    len(baseline_cmt1a)
)



aligned = baseline_cmt1a[
    baseline_cmt1a["ID"].isin(verified_ids)
].copy()


aligned = aligned.sort_values("ID")


print("\n" + "=" * 70)
print("CLINICAL ALIGNMENT")
print("=" * 70)

print(
    "Stage 2 verified patients:",
    len(verified_ids)
)

print(
    "Matched clinical patients:",
    len(aligned)
)



matched_ids = set(aligned["ID"])


missing_clinical = sorted(
    set(verified_ids) -
    matched_ids
)


if missing_clinical:

    print("\n Missing clinical data:")

    for patient_id in missing_clinical:
        print("  ", patient_id)

    raise ValueError(
        "Some Stage 2 patients do not have matching clinical data."
    )

else:

    print(
        "\n✅ Every Stage 2 patient has baseline clinical data."
    )



missing_gait = []


for patient_id in verified_ids:

    gait_file = (
        GAIT_DIR /
        f"{patient_id}_gait.npz"
    )

    if not gait_file.exists():

        missing_gait.append(patient_id)


if missing_gait:

    print("\n Missing gait files:")

    for patient_id in missing_gait:
        print("  ", patient_id)

    raise ValueError(
        "Some Stage 2 patients do not have processed gait files."
    )

else:

    print(
        "✅ Every Stage 2 patient has a processed gait file."
    )


# ============================================================
# 10. CHECK DUPLICATE CLINICAL RECORDS
# ============================================================

duplicate_counts = (
    aligned["ID"]
    .value_counts()
)


duplicate_ids = duplicate_counts[
    duplicate_counts > 1
]


if len(duplicate_ids) > 0:

    print("\n❌ Duplicate baseline records:")

    print(
        duplicate_ids.to_string()
    )

    raise ValueError(
        "Duplicate baseline clinical records detected."
    )

else:

    print(
        " No duplicate baseline clinical records."
    )



# CMTES:
#     0-7   = Mild
#     8-16  = Moderate
#     17-28 = Severe


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


aligned["severity_label"] = (
    aligned["Cmtes_score"]
    .apply(map_severity)
)


aligned["severity_name"] = (
    aligned["severity_label"]
    .map({
        0: "Mild",
        1: "Moderate",
        2: "Severe"
    })
)



if aligned["severity_label"].isna().any():

    invalid = aligned[
        aligned["severity_label"].isna()
    ][
        ["ID", "Cmtes_score"]
    ]

    print("\n❌ Invalid CMTES values:")

    print(
        invalid.to_string(index=False)
    )

    raise ValueError(
        "Invalid CMTES severity value detected."
    )



def get_split(patient_id):

    if patient_id in train_ids:

        return "train"

    elif patient_id in val_ids:

        return "val"

    elif patient_id in test_ids:

        return "test"

    return None


aligned["split"] = (
    aligned["ID"]
    .apply(get_split)
)


if aligned["split"].isna().any():

    raise ValueError(
        "Some patients do not belong to a Stage 2 split."
    )


# ============================================================
# 14. SEVERITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SEVERITY DISTRIBUTION")
print("=" * 70)


severity_distribution = (
    aligned["severity_name"]
    .value_counts()
    .reindex(
        ["Mild", "Moderate", "Severe"],
        fill_value=0
    )
)


print(
    severity_distribution.to_string()
)


# ============================================================
# 15. SEVERITY BY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SEVERITY BY SPLIT")
print("=" * 70)


split_table = pd.crosstab(
    aligned["split"],
    aligned["severity_name"]
).reindex(
    index=["train", "val", "test"],
    columns=["Mild", "Moderate", "Severe"],
    fill_value=0
)


print(
    split_table.to_string()
)


# ============================================================
# 16. IMPORTANT CLINICAL VARIABLES
# ============================================================

important_columns = [

    "ID",
    "Status",
    "Time",
    "Gender",
    "Age (Years)",
    "Cmtes_score",
    "CMT-FOM_Total",
    "MMT_average",
    "Six_minute_walk_distance (meters)",
    "Pinprick",
    "Vibration",
    "severity_label",
    "severity_name",
    "split"
]


available_columns = [
    col
    for col in important_columns
    if col in aligned.columns
]


print("\n" + "=" * 70)
print("CLINICAL VARIABLES")
print("=" * 70)


print(
    aligned[available_columns]
    .to_string(index=False)
)




print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)


missing = (
    aligned[available_columns]
    .isna()
    .sum()
)


print(
    missing.to_string()
)


numeric_columns = [

    "Age (Years)",
    "Cmtes_score",
    "CMT-FOM_Total",
    "MMT_average",
    "Six_minute_walk_distance (meters)",
    "Pinprick",
    "Vibration"
]


numeric_columns = [
    col
    for col in numeric_columns
    if col in aligned.columns
]


print("\n" + "=" * 70)
print("CLINICAL STATISTICS")
print("=" * 70)


print(
    aligned[numeric_columns]
    .describe()
    .round(3)
    .to_string()
)



if len(aligned) != len(verified_ids):

    raise ValueError(
        "Final aligned patient count does not match Stage 2."
    )


if aligned["ID"].duplicated().any():

    raise ValueError(
        "Duplicate patient IDs remain."
    )


if set(aligned["ID"]) != set(verified_ids):

    raise ValueError(
        "Clinical IDs do not exactly match Stage 2 IDs."
    )


print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

print(
    " Patient count matches Stage 2."
)

print(
    " No duplicate patient IDs."
)

print(
    " All patients have clinical data."
)

print(
    "✅ All patients have gait files."
)

print(
    "✅ Severity labels are valid."
)

print(
    "✅ Train/validation/test patient IDs do not overlap."
)


# ============================================================
# 20. SAVE
# ============================================================

aligned.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("STAGE 3 COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print(
    "\nAligned patients:",
    len(aligned)
)