import pandas as pd
from sklearn.model_selection import KFold
import os

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"D:\CMT_PROJECT"

CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "clinical_baseline.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "cmt1a_patient_folds.csv"
)

# ============================================================
# SETTINGS
# ============================================================

N_FOLDS = 5
RANDOM_STATE = 42

# ============================================================
# LOAD CLINICAL DATA
# ============================================================

df = pd.read_csv(CLINICAL_FILE)

print("=" * 70)
print("CMT1A PATIENT-LEVEL 5-FOLD SPLIT")
print("=" * 70)

# Keep only CMT1A patients
cmt1a = df[df["Status"] == "CMT1A"].copy()

# Sort patients for reproducibility
cmt1a = cmt1a.sort_values("ID").reset_index(drop=True)

print(f"\nCMT1A patients: {len(cmt1a)}")

# ============================================================
# BASIC VALIDATION
# ============================================================

if len(cmt1a) != 15:
    raise ValueError(
        f"Expected 15 CMT1A patients, but found {len(cmt1a)}"
    )

if cmt1a["ID"].duplicated().any():
    raise ValueError("Duplicate CMT1A patient IDs found.")

if cmt1a["Cmtes_score"].isna().any():
    raise ValueError("Missing CMTES scores found.")

# ============================================================
# CREATE 5-FOLD SPLIT
# ============================================================

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

rows = []

print("\n" + "=" * 70)
print("FOLD INFORMATION")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(
    kf.split(cmt1a), start=1
):

    train_patients = cmt1a.iloc[train_idx]
    test_patients = cmt1a.iloc[test_idx]

    print(f"\nFold {fold}")
    print("-" * 40)

    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients : {len(test_patients)}")

    print("\nTrain:")
    print(", ".join(train_patients["ID"].tolist()))

    print("\nTest:")
    print(", ".join(test_patients["ID"].tolist()))

    # Store train/test information
    for _, row in cmt1a.iterrows():

        patient_id = row["ID"]

        if patient_id in test_patients["ID"].values:
            split = "test"
        else:
            split = "train"

        rows.append({
            "ID": patient_id,
            "fold": fold,
            "split": split,
            "Cmtes_score": row["Cmtes_score"]
        })

# ============================================================
# CREATE DATAFRAME
# ============================================================

fold_df = pd.DataFrame(rows)

# Sort for easy inspection
fold_df = fold_df.sort_values(
    ["fold", "split", "ID"]
).reset_index(drop=True)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# Every patient should occur once per fold
patient_counts = fold_df.groupby("ID")["fold"].nunique()

if not (patient_counts == N_FOLDS).all():
    raise ValueError(
        "Some patients do not appear in all 5 folds."
    )

# Every fold should have 12 train and 3 test
for fold in range(1, N_FOLDS + 1):

    fold_data = fold_df[fold_df["fold"] == fold]

    train_count = (fold_data["split"] == "train").sum()
    test_count = (fold_data["split"] == "test").sum()

    if train_count != 12 or test_count != 3:
        raise ValueError(
            f"Fold {fold} has {train_count} train and "
            f"{test_count} test patients."
        )

    print(
        f"Fold {fold}: "
        f"{train_count} train + {test_count} test = "
        f"{train_count + test_count} patients"
    )

# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_FILE)

print(f"\nShape: {fold_df.shape}")

print("\nFirst rows:")
print(fold_df.head(15).to_string(index=False))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)