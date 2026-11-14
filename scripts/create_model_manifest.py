import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# STEP 4
# CREATE MODEL-READY WINDOW MANIFEST
# ============================================================

PROJECT_ROOT = r"D:\CMT_PROJECT"

SEQUENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "baseline_sequences"
)

SEQUENCE_SUMMARY = os.path.join(
    SEQUENCE_DIR,
    "baseline_sequence_summary.csv"
)

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "clinical_baseline.csv"
)

FOLD_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "cmt1a_patient_folds.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "processed",
    "model_ready"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_MANIFEST_FILE = os.path.join(
    OUTPUT_DIR,
    "cmt1a_window_manifest.csv"
)

BALANCED_TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "cmt1a_balanced_train_manifest.csv"
)

# ============================================================
# SETTINGS
# ============================================================

# Maximum number of windows from each
# patient + sensor + activity combination
MAX_TRAIN_WINDOWS_PER_GROUP = 100

# Reproducible sampling
RANDOM_STATE = 42

# Expected sequence shape
EXPECTED_WINDOW_SIZE = 64
EXPECTED_CHANNELS = 3

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 4 - MODEL-READY WINDOW MANIFEST")
print("=" * 75)

print("\nLoading sequence summary...")

sequence_df = pd.read_csv(SEQUENCE_SUMMARY)

print(f"Sequence records: {len(sequence_df)}")

print("\nLoading clinical data...")

clinical_df = pd.read_csv(CLINICAL_FILE)

print(f"Clinical records: {len(clinical_df)}")

print("\nLoading patient folds...")

fold_df = pd.read_csv(FOLD_FILE)

print(f"Fold records: {len(fold_df)}")

# ============================================================
# FILTER CMT1A PATIENTS
# ============================================================

cmt1a_clinical = clinical_df[
    clinical_df["Status"] == "CMT1A"
].copy()

print("\nCMT1A patients:", len(cmt1a_clinical))

# ============================================================
# VALIDATE CMT1A PATIENT COUNT
# ============================================================

if len(cmt1a_clinical) != 15:
    raise ValueError(
        f"Expected 15 CMT1A patients, "
        f"but found {len(cmt1a_clinical)}"
    )

# ============================================================
# FILTER SEQUENCE DATA TO CMT1A
# ============================================================

cmt1a_ids = set(
    cmt1a_clinical["ID"].astype(str)
)

sequence_df["patient_id"] = (
    sequence_df["patient_id"].astype(str)
)

cmt1a_sequences = sequence_df[
    sequence_df["patient_id"].isin(cmt1a_ids)
].copy()

print(
    "CMT1A sequence records:",
    len(cmt1a_sequences)
)

# ============================================================
# CHECK THAT ALL CMT1A PATIENTS HAVE SEQUENCES
# ============================================================

sequence_patients = set(
    cmt1a_sequences["patient_id"]
)

missing_patients = sorted(
    cmt1a_ids - sequence_patients
)

if missing_patients:
    raise ValueError(
        "CMT1A patients without gait sequences:\n"
        + "\n".join(missing_patients)
    )

print("All CMT1A patients have gait sequences.")

# ============================================================
# PREPARE FOLD INFORMATION
# ============================================================

fold_df["ID"] = fold_df["ID"].astype(str)

fold_df = fold_df[
    fold_df["ID"].isin(cmt1a_ids)
].copy()

# ============================================================
# VALIDATE FOLD FILE
# ============================================================

expected_fold_rows = 15 * 5

if len(fold_df) != expected_fold_rows:
    raise ValueError(
        f"Expected {expected_fold_rows} fold rows, "
        f"but found {len(fold_df)}"
    )

# ============================================================
# CREATE PATIENT -> FOLD LOOKUP
# ============================================================

# Every patient has one test fold.
patient_test_fold = {}

for patient_id in cmt1a_ids:

    patient_rows = fold_df[
        fold_df["ID"] == patient_id
    ]

    test_rows = patient_rows[
        patient_rows["split"] == "test"
    ]

    if len(test_rows) != 1:
        raise ValueError(
            f"Patient {patient_id} does not have "
            f"exactly one test fold."
        )

    patient_test_fold[patient_id] = int(
        test_rows.iloc[0]["fold"]
    )

# ============================================================
# ADD CMTES SCORE
# ============================================================

score_lookup = dict(
    zip(
        cmt1a_clinical["ID"].astype(str),
        cmt1a_clinical["Cmtes_score"]
    )
)

cmt1a_sequences["Cmtes_score"] = (
    cmt1a_sequences["patient_id"]
    .map(score_lookup)
)

# ============================================================
# CHECK FOR MISSING TARGETS
# ============================================================

missing_scores = cmt1a_sequences[
    cmt1a_sequences["Cmtes_score"].isna()
]

if len(missing_scores) > 0:
    raise ValueError(
        "Some CMT1A sequence records have missing CMTES."
    )

# ============================================================
# ADD FOLD AND SPLIT
# ============================================================

cmt1a_sequences["test_fold"] = (
    cmt1a_sequences["patient_id"]
    .map(patient_test_fold)
)

# ============================================================
# EXPAND EACH NPY FILE INTO WINDOW ROWS
# ============================================================

print("\n" + "=" * 75)
print("EXPANDING NPY FILES INTO WINDOWS")
print("=" * 75)

manifest_rows = []

total_files = len(cmt1a_sequences)

processed_files = 0
total_windows = 0

for row_index, row in cmt1a_sequences.iterrows():

    patient_id = str(row["patient_id"])
    sensor = str(row["sensor"])
    activity = str(row["activity"])
    trial_number = int(row["trial_number"])

    file_name = str(row["file"])

    # --------------------------------------------------------
    # Locate NPY file
    # --------------------------------------------------------

    npy_path = os.path.join(
        SEQUENCE_DIR,
        file_name
    )

    # If the stored path is relative or has another form,
    # try basename as a fallback.
    if not os.path.exists(npy_path):

        possible_files = glob.glob(
            os.path.join(
                SEQUENCE_DIR,
                "**",
                os.path.basename(file_name)
            ),
            recursive=True
        )

        if possible_files:
            npy_path = possible_files[0]

        else:
            raise FileNotFoundError(
                f"Could not find sequence file:\n{file_name}"
            )

    # --------------------------------------------------------
    # Load sequence
    # --------------------------------------------------------

    data = np.load(
        npy_path,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if data.ndim != 3:
        raise ValueError(
            f"Unexpected dimensions for {file_name}: "
            f"{data.shape}"
        )

    if data.shape[1] != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            f"Unexpected window size for {file_name}: "
            f"{data.shape}"
        )

    if data.shape[2] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Unexpected channel count for {file_name}: "
            f"{data.shape}"
        )

    number_of_windows = data.shape[0]

    # --------------------------------------------------------
    # Determine split for each fold
    # --------------------------------------------------------

    # This patient's test fold
    test_fold = patient_test_fold[patient_id]

    # --------------------------------------------------------
    # Create one manifest row per window
    # --------------------------------------------------------

    for window_index in range(number_of_windows):

        manifest_rows.append({
            "patient_id": patient_id,
            "sensor": sensor,
            "activity": activity,
            "trial_number": trial_number,
            "window_index": window_index,
            "file": file_name,
            "npy_path": npy_path,
            "window_size": EXPECTED_WINDOW_SIZE,
            "channels": EXPECTED_CHANNELS,
            "Cmtes_score": float(row["Cmtes_score"]),
            "test_fold": test_fold
        })

    total_windows += number_of_windows
    processed_files += 1

    if processed_files % 50 == 0:
        print(
            f"Processed {processed_files}/{total_files} files | "
            f"Windows: {total_windows:,}"
        )

# ============================================================
# CREATE MANIFEST DATAFRAME
# ============================================================

manifest_df = pd.DataFrame(manifest_rows)

print(
    f"\nTotal model windows: "
    f"{len(manifest_df):,}"
)

# ============================================================
# ADD FOLD-SPECIFIC SPLIT
# ============================================================
#
# IMPORTANT:
#
# A patient is TEST only in their designated fold.
#
# Example:
#
# Patient ROC-008:
#   Fold 1 -> TEST
#   Fold 2 -> TRAIN
#   Fold 3 -> TRAIN
#   Fold 4 -> TRAIN
#   Fold 5 -> TRAIN
#
# ============================================================

# ============================================================
# SAVE ALL WINDOWS
# ============================================================

manifest_df.to_csv(
    ALL_MANIFEST_FILE,
    index=False
)

print("\nSaved all-window manifest:")
print(ALL_MANIFEST_FILE)

# ============================================================
# DISPLAY BASIC STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("ALL-WINDOW MANIFEST SUMMARY")
print("=" * 75)

print(
    "\nPatients:",
    manifest_df["patient_id"].nunique()
)

print(
    "Sensors:",
    manifest_df["sensor"].nunique()
)

print(
    "Activities:",
    manifest_df["activity"].nunique()
)

print(
    "Total windows:",
    len(manifest_df)
)

print("\nWindows by activity:")

activity_summary = (
    manifest_df
    .groupby("activity")
    .agg(
        windows=("window_index", "count"),
        patients=("patient_id", "nunique")
    )
    .sort_values("windows", ascending=False)
)

print(activity_summary)

# ============================================================
# CREATE BALANCED TRAINING MANIFEST
# ============================================================
#
# We do NOT balance the whole dataset.
#
# We only balance TRAINING windows.
#
# TEST WINDOWS ARE NOT REDUCED.
#
# Maximum:
# 100 windows per
# patient + sensor + activity
#
# ============================================================

print("\n" + "=" * 75)
print("CREATING BALANCED TRAINING MANIFEST")
print("=" * 75)

balanced_parts = []

for fold in range(1, 6):

    print(f"\nProcessing Fold {fold}...")

    # --------------------------------------------------------
    # Patients that are TRAIN in this fold
    # --------------------------------------------------------

    train_patients = fold_df[
        (fold_df["fold"] == fold) &
        (fold_df["split"] == "train")
    ]["ID"].astype(str).tolist()

    train_data = manifest_df[
        manifest_df["patient_id"].isin(train_patients)
    ].copy()

    print(
        "Original training windows:",
        len(train_data)
    )

    # --------------------------------------------------------
    # Sample maximum 100 windows from each
    # patient + sensor + activity
    # --------------------------------------------------------

    balanced_fold_parts = []

    grouped = train_data.groupby(
        [
            "patient_id",
            "sensor",
            "activity"
        ],
        sort=False
    )

    for (
        patient_id,
        sensor,
        activity
    ), group in grouped:

        if len(group) > MAX_TRAIN_WINDOWS_PER_GROUP:

            sampled = group.sample(
                n=MAX_TRAIN_WINDOWS_PER_GROUP,
                random_state=RANDOM_STATE
            )

        else:

            sampled = group.copy()

        balanced_fold_parts.append(sampled)

    balanced_fold = pd.concat(
        balanced_fold_parts,
        ignore_index=True
    )

    # Add fold number
    balanced_fold["fold"] = fold
    balanced_fold["split"] = "train"

    balanced_fold_parts = balanced_fold

    print(
        "Balanced training windows:",
        len(balanced_fold)
    )

    balanced_parts.append(
        balanced_fold
    )

# ============================================================
# COMBINE BALANCED TRAINING DATA
# ============================================================

balanced_train_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# ============================================================
# SAVE BALANCED TRAINING MANIFEST
# ============================================================

balanced_train_df.to_csv(
    BALANCED_TRAIN_FILE,
    index=False
)

print("\nSaved balanced training manifest:")
print(BALANCED_TRAIN_FILE)

# ============================================================
# VALIDATION 1
# NO CONTROL PATIENTS
# ============================================================

if not set(
    balanced_train_df["patient_id"]
).issubset(cmt1a_ids):

    raise ValueError(
        "Non-CMT1A patient found in balanced manifest."
    )

print("\nValidation 1: PASS")
print("Only CMT1A patients are included.")

# ============================================================
# VALIDATION 2
# EACH FOLD HAS 12 TRAIN PATIENTS
# ============================================================

print("\n" + "=" * 75)
print("FOLD VALIDATION")
print("=" * 75)

for fold in range(1, 6):

    fold_train = balanced_train_df[
        balanced_train_df["fold"] == fold
    ]

    patients = fold_train[
        "patient_id"
    ].nunique()

    print(
        f"Fold {fold}: "
        f"{patients} training patients | "
        f"{len(fold_train):,} windows"
    )

    if patients != 12:
        raise ValueError(
            f"Fold {fold} should contain "
            f"12 training patients."
        )

print("\nValidation 2: PASS")

# ============================================================
# VALIDATION 3
# PATIENTS MUST NEVER BE TEST IN THEIR OWN TRAINING FOLD
# ============================================================

for fold in range(1, 6):

    train_patients = set(
        balanced_train_df[
            balanced_train_df["fold"] == fold
        ]["patient_id"]
    )

    test_patients = set(
        fold_df[
            (fold_df["fold"] == fold) &
            (fold_df["split"] == "test")
        ]["ID"].astype(str)
    )

    overlap = train_patients.intersection(
        test_patients
    )

    if overlap:
        raise ValueError(
            f"DATA LEAKAGE in Fold {fold}: "
            f"{sorted(overlap)}"
        )

    print(
        f"Fold {fold}: No patient leakage"
    )

print("\nValidation 3: PASS")

# ============================================================
# VALIDATION 4
# TEST PATIENTS
# ============================================================

print("\n" + "=" * 75)
print("TEST PATIENT VALIDATION")
print("=" * 75)

for fold in range(1, 6):

    test_patients = fold_df[
        (fold_df["fold"] == fold) &
        (fold_df["split"] == "test")
    ]["ID"].astype(str).tolist()

    print(
        f"Fold {fold} test patients: "
        f"{', '.join(test_patients)}"
    )

    if len(test_patients) != 3:
        raise ValueError(
            f"Fold {fold} should have "
            f"3 test patients."
        )

print("\nValidation 4: PASS")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(
    "\n1. ALL WINDOWS:"
)
print(
    ALL_MANIFEST_FILE
)

print(
    "\n2. BALANCED TRAINING:"
)
print(
    BALANCED_TRAIN_FILE
)

print("\nStatistics:")

print(
    f"Total CMT1A patients: "
    f"{manifest_df['patient_id'].nunique()}"
)

print(
    f"Total windows: "
    f"{len(manifest_df):,}"
)

print(
    f"Balanced training rows: "
    f"{len(balanced_train_df):,}"
)

print("\n" + "=" * 75)
print("DO NOT START MODEL TRAINING YET")
print("=" * 75)