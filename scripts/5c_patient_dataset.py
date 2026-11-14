import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = r"D:\CMT_PROJECT"

MANIFEST_PATH = os.path.join(
    ROOT,
    "processed",
    "splits",
    "stage5_patient_manifest.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "processed",
    "splits",
    "stage5c_patient_manifest.csv"
)

GAIT_DIR = os.path.join(
    ROOT,
    "processed",
    "gait_windows"
)



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


print("=" * 70)
print("STAGE 5C: CMT1A PATIENT DATASET")
print("=" * 70)



if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(
        f"Stage 5A manifest not found:\n{MANIFEST_PATH}"
    )



print("\nLoading Stage 5A patient manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

print(f"Patients loaded: {len(manifest)}")
print(f"Columns: {manifest.columns.tolist()}")



required_columns = [
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

missing_columns = [
    col for col in required_columns
    if col not in manifest.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )



if manifest["ID"].duplicated().any():
    duplicate_ids = manifest.loc[
        manifest["ID"].duplicated(),
        "ID"
    ].tolist()

    raise ValueError(
        f"Duplicate patient IDs found: {duplicate_ids}"
    )



#  CMT1A ONLY


manifest["Status"] = (
    manifest["Status"]
    .astype(str)
    .str.strip()
)

cmt1a = manifest[
    manifest["Status"].str.upper() == "CMT1A"
].copy()

if len(cmt1a) == 0:
    raise ValueError("No CMT1A patients found.")

if len(cmt1a) != len(manifest):
    print(
        f"\nWarning: {len(manifest) - len(cmt1a)} "
        "non-CMT1A rows were removed."
    )

manifest = cmt1a.reset_index(drop=True)

#  VALIDATE SEVERITY LABELS


valid_labels = {0, 1, 2}

if not set(manifest["severity_label"].dropna().astype(int)).issubset(
    valid_labels
):
    raise ValueError(
        "Invalid severity_label found. "
        "Expected only 0, 1, 2."
    )


expected_names = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}

for _, row in manifest.iterrows():

    label = int(row["severity_label"])
    name = str(row["severity_name"]).strip()

    if label not in expected_names:
        raise ValueError(
            f"Invalid severity label for {row['ID']}: {label}"
        )

    if name != expected_names[label]:
        raise ValueError(
            f"Severity mismatch for {row['ID']}: "
            f"label={label}, name={name}"
        )



# VALIDATE SPLITS


valid_splits = {"train", "val", "test"}

actual_splits = set(
    manifest["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if not actual_splits.issubset(valid_splits):
    raise ValueError(
        f"Invalid split values found: {actual_splits}"
    )

manifest["split"] = (
    manifest["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)



clinical_features = [
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration"
]



#  ENCODE GENDER

# Female = 0
# Male   = 1


def encode_gender(value):

    value = str(value).strip().lower()

    if value == "female":
        return 0.0

    if value == "male":
        return 1.0

    raise ValueError(
        f"Unknown Gender value: {value}"
    )


manifest["Gender_encoded"] = manifest["Gender"].apply(
    encode_gender
)



#  CONVERT NUMERIC CLINICAL FEATURES


numeric_features = [
    "Age (Years)",
    "Pinprick",
    "Vibration"
]

for feature in numeric_features:

    manifest[feature] = pd.to_numeric(
        manifest[feature],
        errors="coerce"
    )



# TRAINING-ONLY MEDIAN IMPUTATION


train_rows = manifest[
    manifest["split"] == "train"
].copy()

if len(train_rows) == 0:
    raise ValueError("Training split is empty.")


print("\nTraining-only clinical medians:")

training_medians = {}

for feature in numeric_features:

    median_value = train_rows[feature].median()

    if pd.isna(median_value):
        raise ValueError(
            f"Training median is NaN for {feature}"
        )

    training_medians[feature] = float(median_value)

    manifest[feature] = manifest[feature].fillna(
        median_value
    )

    print(
        f"{feature}: {median_value}"
    )


# . VALIDATE CLINICAL DATA


for feature in numeric_features + ["Gender_encoded"]:

    if manifest[feature].isna().any():

        bad_ids = manifest.loc[
            manifest[feature].isna(),
            "ID"
        ].tolist()

        raise ValueError(
            f"Missing clinical values remain in "
            f"{feature}: {bad_ids}"
        )



# . CHECK GAIT FILES


print("\nChecking gait files...")

missing_gait = []

for _, row in manifest.iterrows():

    gait_file = str(row["gait_file"])

    # Handle both absolute and relative paths
    if os.path.isabs(gait_file):
        full_path = gait_file
    else:
        full_path = os.path.join(
            ROOT,
            gait_file
        )

        # If that does not exist, try gait directory
        if not os.path.exists(full_path):
            full_path = os.path.join(
                GAIT_DIR,
                os.path.basename(gait_file)
            )

    if not os.path.exists(full_path):
        missing_gait.append(
            (row["ID"], full_path)
        )


if missing_gait:

    print("\nMissing gait files:")

    for patient_id, path in missing_gait:
        print(patient_id, "->", path)

    raise FileNotFoundError(
        f"{len(missing_gait)} gait files are missing."
    )


print(f"Gait files found: {len(manifest)}")



#  STORE NORMALIZED GAIT PATH

normalized_gait_paths = []

for _, row in manifest.iterrows():

    gait_file = str(row["gait_file"])

    if os.path.isabs(gait_file):
        full_path = gait_file
    else:

        full_path = os.path.join(
            ROOT,
            gait_file
        )

        if not os.path.exists(full_path):

            full_path = os.path.join(
                GAIT_DIR,
                os.path.basename(gait_file)
            )

    normalized_gait_paths.append(
        os.path.abspath(full_path)
    )

manifest["gait_file"] = normalized_gait_paths



#  SEVERITY DISTRIBUTION


print("\n" + "=" * 70)
print("SEVERITY DISTRIBUTION")
print("=" * 70)

severity_counts = (
    manifest["severity_name"]
    .value_counts()
    .reindex(
        ["Mild", "Moderate", "Severe"],
        fill_value=0
    )
)

for severity_name, count in severity_counts.items():

    print(
        f"{severity_name}: {count}"
    )


# ============================================================
#  PATIENT SPLIT
# ============================================================

print("\n" + "=" * 70)
print("PATIENT SPLIT")
print("=" * 70)

split_table = pd.crosstab(
    manifest["split"],
    manifest["severity_name"]
).reindex(
    index=["train", "val", "test"],
    columns=["Mild", "Moderate", "Severe"],
    fill_value=0
)

print(split_table)


# ============================================================
# SPLIT OVERLAP CHECK
# ============================================================

train_ids = set(
    manifest.loc[
        manifest["split"] == "train",
        "ID"
    ]
)

val_ids = set(
    manifest.loc[
        manifest["split"] == "val",
        "ID"
    ]
)

test_ids = set(
    manifest.loc[
        manifest["split"] == "test",
        "ID"
    ]
)

if train_ids & val_ids:
    raise ValueError(
        "Train/Validation patient overlap detected."
    )

if train_ids & test_ids:
    raise ValueError(
        "Train/Test patient overlap detected."
    )

if val_ids & test_ids:
    raise ValueError(
        "Validation/Test patient overlap detected."
    )



#  PATIENT DATASET CLASS


class CMT1APatientDataset(Dataset):
    """
    Patient-level dataset.

    Each item contains:

        patient_id
        gait
        clinical
        severity_label

    Gait:
        [num_windows, 256, 3]

    Clinical:
        [4]

    Severity:
        0 = Mild
        1 = Moderate
        2 = Severe
    """

    def __init__(self, dataframe):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        patient_id = str(row["ID"])

        # ----------------------------------------------------
        # Load gait data
        # ----------------------------------------------------

        gait_path = row["gait_file"]

        data = np.load(
            gait_path,
            allow_pickle=False
        )

        if "X" not in data:

            raise KeyError(
                f"'X' not found in gait file: "
                f"{gait_path}"
            )

        X = data["X"]

        # ----------------------------------------------------
        # Validate gait
        # ----------------------------------------------------

        if X.ndim != 3:

            raise ValueError(
                f"Invalid gait shape for {patient_id}: "
                f"{X.shape}"
            )

        if X.shape[1] != 256:

            raise ValueError(
                f"Expected 256 samples per gait window "
                f"for {patient_id}, got {X.shape[1]}"
            )

        if X.shape[2] != 3:

            raise ValueError(
                f"Expected 3 gait channels "
                f"for {patient_id}, got {X.shape[2]}"
            )

        if not np.isfinite(X).all():

            raise ValueError(
                f"NaN or Inf found in gait data "
                f"for {patient_id}"
            )

        # ----------------------------------------------------
        # Convert gait to tensor
        # ----------------------------------------------------

        gait_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        clinical_values = np.array(
            [
                float(row["Age (Years)"]),
                float(row["Gender_encoded"]),
                float(row["Pinprick"]),
                float(row["Vibration"])
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate clinical values
        # ----------------------------------------------------

        if not np.isfinite(clinical_values).all():

            raise ValueError(
                f"NaN or Inf found in clinical data "
                f"for {patient_id}"
            )

        clinical_tensor = torch.tensor(
            clinical_values,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Severity label
        # ----------------------------------------------------

        severity_label = int(
            row["severity_label"]
        )

        if severity_label not in [0, 1, 2]:

            raise ValueError(
                f"Invalid severity label "
                f"for {patient_id}: "
                f"{severity_label}"
            )

        severity_tensor = torch.tensor(
            severity_label,
            dtype=torch.long
        )

        return (
            patient_id,
            gait_tensor,
            clinical_tensor,
            severity_tensor
        )



#  CREATE DATAFRAMES


train_df = manifest[
    manifest["split"] == "train"
].copy()

val_df = manifest[
    manifest["split"] == "val"
].copy()

test_df = manifest[
    manifest["split"] == "test"
].copy()



#  CREATE DATASET OBJECTS


train_dataset = CMT1APatientDataset(
    train_df
)

val_dataset = CMT1APatientDataset(
    val_df
)

test_dataset = CMT1APatientDataset(
    test_df
)


print("\n" + "=" * 70)
print("DATASET OBJECTS")
print("=" * 70)

print(
    f"Training patients: {len(train_dataset)}"
)

print(
    f"Validation patients: {len(val_dataset)}"
)

print(
    f"Testing patients: {len(test_dataset)}"
)



print("\n" + "=" * 70)
print("SAMPLE CHECK")
print("=" * 70)

if len(train_dataset) == 0:

    raise ValueError(
        "Training dataset is empty."
    )

sample = train_dataset[0]

sample_patient_id = sample[0]
sample_gait = sample[1]
sample_clinical = sample[2]
sample_severity = sample[3]


print(
    f"Patient ID: {sample_patient_id}"
)

print(
    f"Gait shape: {sample_gait.shape}"
)

print(
    f"Clinical shape: {sample_clinical.shape}"
)

print(
    f"Clinical values: {sample_clinical.tolist()}"
)

print(
    f"Severity label: {sample_severity.item()}"
)

print(
    f"Severity name: "
    f"{expected_names[sample_severity.item()]}"
)



# . TEST ALL DATASET OBJECTS


print("\nValidating dataset samples...")

for dataset_name, dataset in [
    ("train", train_dataset),
    ("val", val_dataset),
    ("test", test_dataset)
]:

    for index in range(len(dataset)):

        patient_id, gait, clinical, severity = dataset[index]

        if gait.ndim != 3:
            raise ValueError(
                f"Invalid gait dimensions for "
                f"{dataset_name} patient {patient_id}"
            )

        if clinical.shape != (4,):
            raise ValueError(
                f"Invalid clinical shape for "
                f"{dataset_name} patient {patient_id}: "
                f"{clinical.shape}"
            )

        if severity.item() not in [0, 1, 2]:
            raise ValueError(
                f"Invalid severity for "
                f"{dataset_name} patient {patient_id}"
            )

print("All dataset samples validated successfully.")



#  SAVE FINAL STAGE 5C MANIFEST


# Save only useful manifest information.
# Gender_encoded is included for reproducibility.

save_columns = [
    "ID",
    "Status",
    "Age (Years)",
    "Gender",
    "Gender_encoded",
    "Pinprick",
    "Vibration",
    "Cmtes_score",
    "severity_label",
    "severity_name",
    "split",
    "gait_file"
]

final_manifest = manifest[
    save_columns
].copy()

final_manifest.to_csv(
    OUTPUT_PATH,
    index=False
)



# FINAL SAFETY CHECKS


print("\n" + "=" * 70)
print("FINAL SAFETY CHECKS")
print("=" * 70)

print(
    f"Total patients: {len(final_manifest)}"
)

print(
    f"Training patients: "
    f"{len(train_df)}"
)

print(
    f"Validation patients: "
    f"{len(val_df)}"
)

print(
    f"Testing patients: "
    f"{len(test_df)}"
)

print(
    f"Duplicate patient IDs: "
    f"{final_manifest['ID'].duplicated().sum()}"
)

print(
    f"Train ∩ Validation: "
    f"{len(train_ids & val_ids)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_ids & test_ids)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(val_ids & test_ids)}"
)



#  SEVERITY MAPPING


print("\nSeverity mapping:")

print("0 = Mild")
print("1 = Moderate")
print("2 = Severe")



#  DATASET LIMITATION


if int(
    (final_manifest["severity_name"] == "Severe").sum()
) == 1:

    print("\nDataset limitation:")
    print(
        "Only one Severe CMT1A patient is available."
    )
    print(
        "The Severe patient is in TRAIN so the model "
        "can learn the Severe class."
    )
    print(
        "TEST contains no independent Severe patient."
    )



print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STAGE 5C COMPLETE")
print("=" * 70)