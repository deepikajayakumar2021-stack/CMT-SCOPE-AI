import os
import re
import zipfile
import pandas as pd
import numpy as np



BASE_DIR = r"D:\CMT_PROJECT\Baseline_CMT1A-BioStampNPoint2023"
CLINICAL_FILE = r"D:\CMT_PROJECT\ClinicData_CMT1A-BioStampNPoint2023.csv"

OUTPUT_DIR = r"D:\CMT_PROJECT\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "baseline_gait_features.csv"
)


GAIT_KEYWORDS = [
    "10m walk",
    "10m walk/run",
    "TUG",
    "6MWT",
    "Stairs (ascending)",
    "Stairs (descending)"
]


def is_gait_activity(name):

    if pd.isna(name):
        return False

    name = str(name).strip().lower()

    return any(
        keyword.lower() in name
        for keyword in GAIT_KEYWORDS
    )


def calculate_features(df, activity):

    if len(df) == 0:
        return None

    x = pd.to_numeric(df["Accel X (g)"], errors="coerce")
    y = pd.to_numeric(df["Accel Y (g)"], errors="coerce")
    z = pd.to_numeric(df["Accel Z (g)"], errors="coerce")

    valid = x.notna() & y.notna() & z.notna()

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(x) == 0:
        return None

    magnitude = np.sqrt(
        x**2 + y**2 + z**2
    )

    features = {

        "activity": activity,

        "sample_count": len(x),

        "acc_x_mean": x.mean(),
        "acc_y_mean": y.mean(),
        "acc_z_mean": z.mean(),

        "acc_x_std": x.std(),
        "acc_y_std": y.std(),
        "acc_z_std": z.std(),

        "acc_x_rms": np.sqrt(np.mean(x**2)),
        "acc_y_rms": np.sqrt(np.mean(y**2)),
        "acc_z_rms": np.sqrt(np.mean(z**2)),

        "magnitude_mean": magnitude.mean(),
        "magnitude_std": magnitude.std(),
        "magnitude_rms": np.sqrt(
            np.mean(magnitude**2)
        ),

        "magnitude_max": magnitude.max(),
        "magnitude_min": magnitude.min()
    }

    return features


def load_annotations(ann_file):

    ann = pd.read_csv(ann_file)

    ann.columns = ann.columns.str.strip()

    gait_ann = ann[
        ann["name"].apply(is_gait_activity)
    ].copy()

    return gait_ann


# ============================================================
# PROCESS ONE ROC FOLDER
# ============================================================

def process_patient(roc_folder):

    roc_id = os.path.basename(roc_folder)

    print("\n========================================")
    print("Processing:", roc_id)
    print("========================================")

    ann_file = os.path.join(
        roc_folder,
        f"AnnotID_{roc_id}.csv"
    )

    sensor_file = os.path.join(
        roc_folder,
        f"ch_ID_{roc_id}_Accel.csv"
    )

    if not os.path.exists(ann_file):

        print("Annotation missing:", roc_id)

        return []

    if not os.path.exists(sensor_file):

        print("Sensor missing:", roc_id)

        return []

    # -------------------------------
    # annotations
    # -------------------------------

    gait_ann = load_annotations(ann_file)

    print(
        "Gait annotations:",
        len(gait_ann)
    )

    if len(gait_ann) == 0:

        print("No gait activities")

        return []



    results = []

    required_columns = [
        "Timestamp (microseconds)",
        "Accel X (g)",
        "Accel Y (g)",
        "Accel Z (g)"
    ]

    print("Reading sensor in chunks...")

    for chunk in pd.read_csv(
        sensor_file,
        chunksize=200_000
    ):

        chunk.columns = chunk.columns.str.strip()

        timestamp = pd.to_numeric(
            chunk["Timestamp (microseconds)"],
            errors="coerce"
        )

        
        for _, ann_row in gait_ann.iterrows():

            start = float(
                ann_row["timestamp_start"]
            )

            stop = float(
                ann_row["timestamp_stop"]
            )

            activity = str(
                ann_row["name"]
            ).strip()

            mask = (
                (timestamp >= start) &
                (timestamp <= stop)
            )

            selected = chunk.loc[
                mask,
                required_columns
            ]

            if len(selected) == 0:
                continue

            feature_row = calculate_features(
                selected,
                activity
            )

            if feature_row is not None:

                feature_row["ID"] = roc_id

                results.append(
                    feature_row
                )

    print(
        "Feature rows:",
        len(results)
    )

    return results



print("\nLoading clinical data...")

clinical = pd.read_csv(
    CLINICAL_FILE
)

clinical.columns = (
    clinical.columns
    .str.strip()
)


clinical_baseline = clinical[
    (clinical["Status"].str.strip() == "CMT1A") &
    (clinical["Time"].str.strip() == "Baseline")
].copy()

print(
    "Baseline CMT1A clinical records:",
    len(clinical_baseline)
)

print(
    clinical_baseline["ID"].tolist()
)



all_features = []

patient_ids = (
    clinical_baseline["ID"]
    .astype(str)
    .str.strip()
    .tolist()
)

print("\nPatients to process:")
print(patient_ids)

for patient_id in patient_ids:

    folder = os.path.join(
        BASE_DIR,
        patient_id
    )

    if not os.path.isdir(folder):

        print(
            "\nWARNING: Folder not found:",
            patient_id
        )

        continue

    patient_features = process_patient(
        folder
    )

    all_features.extend(
        patient_features
    )



print("\n========================================")
print("Creating final feature dataframe")
print("========================================")

features_df = pd.DataFrame(
    all_features
)

print(
    "Raw feature shape:",
    features_df.shape
)


if len(features_df) > 0:

    clinical_merge = clinical_baseline.copy()

    clinical_merge["ID"] = (
        clinical_merge["ID"]
        .astype(str)
        .str.strip()
    )

    features_df["ID"] = (
        features_df["ID"]
        .astype(str)
        .str.strip()
    )

    final_df = features_df.merge(
        clinical_merge,
        on="ID",
        how="left"
    )

else:

    final_df = features_df



final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("DONE")
print("========================================")

print(
    "Final shape:",
    final_df.shape
)

print(
    "Saved to:"
)

print(OUTPUT_FILE)

print("\nActivity counts:")

if len(final_df) > 0:

    print(
        final_df["activity"]
        .value_counts()
    )

print("\nFirst rows:")

print(
    final_df.head()
)