from pathlib import Path
import numpy as np
import pandas as pd

# Severity mapping:
#   CMTES 0-7   -> Mild
#   CMTES 8-16  -> Moderate
#   CMTES 17-28 -> Severe

PROJECT_DIR = Path(
    r"D:\CMT_PROJECT"
)

WINDOW_DIR = (
    PROJECT_DIR /
    "processed" /
    "gait_windows"
)

SPLIT_DIR = (
    PROJECT_DIR /
    "processed" /
    "splits"
)

CLINICAL_FILE = (
    PROJECT_DIR /
    "processed" /
    "clinical_baseline.csv"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print(
    "CMT1A STAGE 2: DATASET VALIDATION + "
    "SEVERITY-AWARE PATIENT-WISE SPLIT"
)
print("=" * 70)



files = sorted(
    WINDOW_DIR.glob("*_gait.npz")
)

print(
    f"\nAll processed gait files: {len(files)}"
)

if len(files) == 0:

    raise FileNotFoundError(
        "No .npz gait files found."
    )


if not CLINICAL_FILE.exists():

    raise FileNotFoundError(
        f"Clinical file not found:\n{CLINICAL_FILE}"
    )


clinical = pd.read_csv(
    CLINICAL_FILE
)


print(
    "\nClinical dataset shape:",
    clinical.shape
)



required_columns = [
    "ID",
    "Status",
    "Time",
    "Cmtes_score"
]


missing_columns = [
    column
    for column in required_columns
    if column not in clinical.columns
]


if missing_columns:

    raise ValueError(
        "Missing clinical columns:\n"
        + "\n".join(missing_columns)
    )



clinical["ID"] = (
    clinical["ID"]
    .astype(str)
    .str.strip()
)

clinical["Status"] = (
    clinical["Status"]
    .astype(str)
    .str.strip()
)

clinical["Time"] = (
    clinical["Time"]
    .astype(str)
    .str.strip()
)


 

eligible_clinical = clinical[
    (
        clinical["Status"]
        .str.upper()
        == "CMT1A"
    )
    &
    (
        clinical["Time"]
        .str.lower()
        == "baseline"
    )
    &
    (
        clinical["Cmtes_score"]
        .notna()
    )
].copy()


print(
    "\nEligible baseline CMT1A clinical patients:",
    len(eligible_clinical)
)


if len(eligible_clinical) == 0:

    raise ValueError(
        "No eligible baseline CMT1A patients "
        "with Cmtes_score were found."
    )


 

def cmtes_to_class(score):

    score = float(score)

    if 0 <= score <= 7:

        return 0

    elif 8 <= score <= 16:

        return 1

    elif 17 <= score <= 28:

        return 2

    else:

        return None


eligible_clinical["severity_label"] = (
    eligible_clinical["Cmtes_score"]
    .apply(cmtes_to_class)
)


eligible_clinical["severity_name"] = (
    eligible_clinical["severity_label"]
    .map(
        {
            0: "Mild",
            1: "Moderate",
            2: "Severe"
        }
    )
)


if (
    eligible_clinical["severity_label"]
    .isna()
    .any()
):

    raise ValueError(
        "Invalid Cmtes_score value found."
    )


eligible_clinical["severity_label"] = (
    eligible_clinical["severity_label"]
    .astype(int)
)



all_gait_patients = []

for file in files:

    patient_id = file.stem.replace(
        "_gait",
        ""
    )

    all_gait_patients.append(
        patient_id
    )


all_gait_patients = sorted(
    set(all_gait_patients)
)




eligible_patient_ids = set(
    eligible_clinical["ID"]
)

gait_patients = sorted(
    set(all_gait_patients)
    .intersection(
        eligible_patient_ids
    )
)


print(
    "\nAll processed gait patients:",
    len(all_gait_patients)
)

print(
    "Labeled CMT1A gait patients:",
    len(gait_patients)
)


if len(gait_patients) == 0:

    raise ValueError(
        "No gait patients match the labeled "
        "baseline CMT1A clinical cohort."
    )


# ============================================================
# 9. IDENTIFY CLINICAL PATIENTS WITHOUT GAIT
# ============================================================

clinical_without_gait = sorted(
    eligible_patient_ids -
    set(all_gait_patients)
)


if clinical_without_gait:

    print(
        "\n⚠️ Labeled CMT1A patients without "
        "processed gait data:"
    )

    print(
        clinical_without_gait
    )


# ============================================================
# 10. VALIDATE ONLY THE ELIGIBLE GAIT FILES
# ============================================================

patient_info = []

total_windows = 0


for file in files:

    patient_id_from_file = (
        file.stem.replace(
            "_gait",
            ""
        )
    )

    # --------------------------------------------------------
    # Ignore gait files that do not belong to the
    # labeled baseline CMT1A cohort.
    # --------------------------------------------------------

    if (
        patient_id_from_file
        not in gait_patients
    ):

        continue


    data = np.load(
        file,
        allow_pickle=True
    )


    # --------------------------------------------------------
    # CHECK REQUIRED NPZ KEYS
    # --------------------------------------------------------

    required_keys = [
        "X",
        "y",
        "patient_id"
    ]


    missing_keys = [
        key
        for key in required_keys
        if key not in data.files
    ]


    if missing_keys:

        raise ValueError(
            f"{file.name} is missing keys: "
            f"{missing_keys}"
        )


    X = data["X"]

    y = data["y"]

    patient_id = str(
        data["patient_id"]
    ).strip()


    # --------------------------------------------------------
    # CHECK PATIENT ID
    # --------------------------------------------------------

    if patient_id != patient_id_from_file:

        raise ValueError(
            f"Patient ID mismatch in {file.name}: "
            f"filename={patient_id_from_file}, "
            f"stored={patient_id}"
        )


    print(
        f"\n{patient_id}"
    )

    print(
        f"  X shape: {X.shape}"
    )

    print(
        f"  y shape: {y.shape}"
    )


    # --------------------------------------------------------
    # BASIC SHAPE CHECK
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"{patient_id}: X must be 3-dimensional."
        )


    if X.shape[1] != 256:

        raise ValueError(
            f"{patient_id}: expected "
            f"window length 256, got {X.shape[1]}."
        )


    if X.shape[2] != 3:

        raise ValueError(
            f"{patient_id}: expected "
            f"3 acceleration channels, got {X.shape[2]}."
        )


    if len(X) != len(y):

        raise ValueError(
            f"{patient_id}: X and y length mismatch."
        )


    # --------------------------------------------------------
    # NaN / INF
    # --------------------------------------------------------

    nan_count = np.isnan(
        X
    ).sum()

    inf_count = np.isinf(
        X
    ).sum()


    print(
        f"  NaN values: {nan_count}"
    )

    print(
        f"  Inf values: {inf_count}"
    )


    if nan_count > 0:

        raise ValueError(
            f"{patient_id}: NaN values detected."
        )


    if inf_count > 0:

        raise ValueError(
            f"{patient_id}: Inf values detected."
        )


    # --------------------------------------------------------
    # ACCELERATION RANGE
    # --------------------------------------------------------

    print(
        f"  Accel range: "
        f"{X.min():.4f} to {X.max():.4f} g"
    )


    # --------------------------------------------------------
    # ACTIVITY DISTRIBUTION
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )


    label_distribution = dict(
        zip(
            unique_labels.astype(str),
            counts
        )
    )


    print(
        f"  Activities: "
        f"{label_distribution}"
    )


    # --------------------------------------------------------
    # WINDOW COUNT
    # --------------------------------------------------------

    n_windows = len(X)

    total_windows += n_windows


    patient_info.append(
        {
            "patient_id": patient_id,
            "windows": n_windows,
            "nan_values": int(
                nan_count
            ),
            "inf_values": int(
                inf_count
            ),
            "min_accel": float(
                X.min()
            ),
            "max_accel": float(
                X.max()
            ),
            "activities": str(
                label_distribution
            )
        }
    )


# ============================================================
# 11. CHECK VALIDATED PATIENT COUNT
# ============================================================

summary = pd.DataFrame(
    patient_info
)


if len(summary) != len(gait_patients):

    raise ValueError(
        "Not all eligible gait patients were validated."
    )


# ============================================================
# 12. CREATE PATIENT SUMMARY
# ============================================================

summary_path = (
    SPLIT_DIR /
    "patient_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)


print(
    f"Eligible patients: {len(summary)}"
)


print(
    f"Total windows: {total_windows}"
)


print(
    "Average windows/patient: "
    f"{summary['windows'].mean():.2f}"
)


print(
    "Minimum windows/patient: "
    f"{summary['windows'].min()}"
)


print(
    "Maximum windows/patient: "
    f"{summary['windows'].max()}"
)


# ============================================================
# 13. MERGE GAIT SUMMARY WITH SEVERITY
# ============================================================

summary["patient_id"] = (
    summary["patient_id"]
    .astype(str)
    .str.strip()
)


patient_data = summary.merge(
    eligible_clinical[
        [
            "ID",
            "Cmtes_score",
            "severity_label",
            "severity_name"
        ]
    ],
    left_on="patient_id",
    right_on="ID",
    how="left"
)


# ============================================================
# 14. CHECK MISSING SEVERITY
# ============================================================

missing_severity = patient_data[
    patient_data["severity_label"].isna()
]


if len(missing_severity) > 0:

    print(
        "\nPatients without CMTES severity:"
    )

    print(
        missing_severity[
            ["patient_id"]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Some labeled gait patients do not "
        "have a valid Cmtes_score."
    )


# ============================================================
# 15. OVERALL SEVERITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SEVERITY DISTRIBUTION")
print("=" * 70)


for label, name in [
    (0, "Mild"),
    (1, "Moderate"),
    (2, "Severe")
]:

    count = int(
        (
            patient_data["severity_label"]
            == label
        ).sum()
    )

    print(
        f"{name}: {count}"
    )


# ============================================================
# 16. CHECK EXPECTED COHORT
# ============================================================

mild = patient_data[
    patient_data["severity_label"] == 0
].copy()


moderate = patient_data[
    patient_data["severity_label"] == 1
].copy()


severe = patient_data[
    patient_data["severity_label"] == 2
].copy()


if len(mild) < 7:

    raise ValueError(
        f"Expected at least 7 Mild patients, "
        f"found {len(mild)}."
    )


if len(moderate) < 7:

    raise ValueError(
        f"Expected at least 7 Moderate patients, "
        f"found {len(moderate)}."
    )


if len(severe) != 1:

    raise ValueError(
        f"Expected exactly 1 Severe patient, "
        f"found {len(severe)}."
    )


# ============================================================
# 17. PATIENT-WISE SEVERITY-AWARE SPLIT
#
# Mild:
#   4 -> train
#   1 -> validation
#   2 -> test
#
# Moderate:
#   4 -> train
#   1 -> validation
#   2 -> test
#
# Severe:
#   1 -> train
#
# Final:
#   Train = 9
#   Validation = 2
#   Test = 4
#
# The only Severe patient is kept in TRAIN so that the
# classifier can learn the Severe class.
# ============================================================


rng = np.random.default_rng(
    42
)


# ------------------------------------------------------------
# SHUFFLE WITHIN EACH SEVERITY CLASS
# ------------------------------------------------------------

mild_indices = np.arange(
    len(mild)
)

moderate_indices = np.arange(
    len(moderate)
)


rng.shuffle(
    mild_indices
)

rng.shuffle(
    moderate_indices
)


mild = mild.iloc[
    mild_indices
].reset_index(
    drop=True
)


moderate = moderate.iloc[
    moderate_indices
].reset_index(
    drop=True
)


# ============================================================
# 18. ASSIGN MILD
# ============================================================

mild_train = mild.iloc[
    :4
].copy()

mild_val = mild.iloc[
    4:5
].copy()

mild_test = mild.iloc[
    5:
].copy()


# ============================================================
# 19. ASSIGN MODERATE
# ============================================================

moderate_train = moderate.iloc[
    :4
].copy()

moderate_val = moderate.iloc[
    4:5
].copy()

moderate_test = moderate.iloc[
    5:
].copy()


# ============================================================
# 20. ASSIGN SEVERE
# ============================================================

severe_train = severe.copy()


# ============================================================
# 21. ADD SPLIT LABEL
# ============================================================

mild_train["split"] = "train"
mild_val["split"] = "val"
mild_test["split"] = "test"

moderate_train["split"] = "train"
moderate_val["split"] = "val"
moderate_test["split"] = "test"

severe_train["split"] = "train"


# ============================================================
# 22. COMBINE SPLITS
# ============================================================

split_data = pd.concat(
    [
        mild_train,
        moderate_train,
        severe_train,
        mild_val,
        moderate_val,
        mild_test,
        moderate_test
    ],
    ignore_index=True
)


# ============================================================
# 23. CREATE PATIENT LISTS
# ============================================================

train_patients = sorted(
    split_data.loc[
        split_data["split"] == "train",
        "patient_id"
    ].tolist()
)


val_patients = sorted(
    split_data.loc[
        split_data["split"] == "val",
        "patient_id"
    ].tolist()
)


test_patients = sorted(
    split_data.loc[
        split_data["split"] == "test",
        "patient_id"
    ].tolist()
)


# ============================================================
# 24. VERIFY NO PATIENT OVERLAP
# ============================================================

train_set = set(
    train_patients
)

val_set = set(
    val_patients
)

test_set = set(
    test_patients
)


assert train_set.isdisjoint(
    val_set
)

assert train_set.isdisjoint(
    test_set
)

assert val_set.isdisjoint(
    test_set
)


print(
    "\n✅ No patient overlap detected."
)


# ============================================================
# 25. VERIFY ALL ELIGIBLE PATIENTS ARE USED ONCE
# ============================================================

all_split_patients = (
    train_set |
    val_set |
    test_set
)


assert (
    all_split_patients
    == set(gait_patients)
)


assert (
    len(all_split_patients)
    == len(gait_patients)
)


print(
    "✅ Every eligible patient appears "
    "in exactly one split."
)


# ============================================================
# 26. SAVE SPLIT FILES
# ============================================================

pd.DataFrame(
    {
        "patient_id":
        train_patients
    }
).to_csv(
    SPLIT_DIR /
    "train_patients.csv",
    index=False
)


pd.DataFrame(
    {
        "patient_id":
        val_patients
    }
).to_csv(
    SPLIT_DIR /
    "val_patients.csv",
    index=False
)


pd.DataFrame(
    {
        "patient_id":
        test_patients
    }
).to_csv(
    SPLIT_DIR /
    "test_patients.csv",
    index=False
)


# ============================================================
# 27. WINDOWS PER SPLIT
# ============================================================

def count_windows(
    patient_list
):

    return int(
        summary[
            summary["patient_id"]
            .isin(patient_list)
        ]["windows"].sum()
    )


train_windows = count_windows(
    train_patients
)

val_windows = count_windows(
    val_patients
)

test_windows = count_windows(
    test_patients
)


# ============================================================
# 28. FINAL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT")
print("=" * 70)


print(
    f"Training   : "
    f"{len(train_patients):2d} patients | "
    f"{train_windows} windows"
)


print(
    f"Validation : "
    f"{len(val_patients):2d} patients | "
    f"{val_windows} windows"
)


print(
    f"Testing    : "
    f"{len(test_patients):2d} patients | "
    f"{test_windows} windows"
)


print(
    f"\nTotal      : "
    f"{total_windows} windows"
)


# ============================================================
# 29. SEVERITY BY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SEVERITY BY SPLIT")
print("=" * 70)


split_table = pd.crosstab(
    split_data["split"],
    split_data["severity_name"]
)


split_table = split_table.reindex(
    index=[
        "train",
        "val",
        "test"
    ],
    columns=[
        "Mild",
        "Moderate",
        "Severe"
    ],
    fill_value=0
)


print(
    split_table
)


# ============================================================
# 30. VERIFY TRAIN HAS ALL THREE CLASSES
# ============================================================

train_classes = set(
    split_data.loc[
        split_data["split"] == "train",
        "severity_label"
    ]
)


if train_classes != {
    0,
    1,
    2
}:

    raise ValueError(
        "Training split does not contain "
        "all three severity classes."
    )


print(
    "\n✅ Training split contains Mild, "
    "Moderate, and Severe."
)


# ============================================================
# 31. DATASET LIMITATION
# ============================================================

print(
    "\nIMPORTANT DATASET LIMITATION:"
)

print(
    "Only one Severe CMT1A patient has a "
    "valid baseline CMTES score."
)

print(
    "That Severe patient is included in TRAIN "
    "so the model can learn the Severe class."
)

print(
    "Therefore, TEST has no independent Severe "
    "patient."
)


# ============================================================
# 32. FILES CREATED
# ============================================================

print(
    "\nFiles created:"
)

print(
    SPLIT_DIR /
    "patient_summary.csv"
)

print(
    SPLIT_DIR /
    "train_patients.csv"
)

print(
    SPLIT_DIR /
    "val_patients.csv"
)

print(
    SPLIT_DIR /
    "test_patients.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 COMPLETE")
print("=" * 70)