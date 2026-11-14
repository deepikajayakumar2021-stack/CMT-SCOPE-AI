import os
import pandas as pd
import numpy as np

# ============================================================
# PATH
# ============================================================

ROC_FOLDER = r"D:\CMT_PROJECT\Baseline_CMT1A-BioStampNPoint2023\ROC-011"

OUTPUT_DIR = r"D:\CMT_PROJECT\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ROC011_test_features.csv"
)

# ============================================================
# GAIT ACTIVITIES
# ============================================================

GAIT_KEYWORDS = [
    "10m walk",
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(df, activity, sensor):

    if len(df) == 0:
        return None

    x = pd.to_numeric(
        df["Accel X (g)"],
        errors="coerce"
    )

    y = pd.to_numeric(
        df["Accel Y (g)"],
        errors="coerce"
    )

    z = pd.to_numeric(
        df["Accel Z (g)"],
        errors="coerce"
    )

    valid = (
        x.notna() &
        y.notna() &
        z.notna()
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if len(x) == 0:
        return None

    magnitude = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    return {

        "ID": "ROC-011",

        "sensor": sensor,

        "activity": activity,

        "sample_count": len(x),

        # -------------------------------
        # Mean
        # -------------------------------

        "acc_x_mean": x.mean(),
        "acc_y_mean": y.mean(),
        "acc_z_mean": z.mean(),

        # -------------------------------
        # Standard deviation
        # -------------------------------

        "acc_x_std": x.std(),
        "acc_y_std": y.std(),
        "acc_z_std": z.std(),

        # -------------------------------
        # RMS
        # -------------------------------

        "acc_x_rms": np.sqrt(
            np.mean(x**2)
        ),

        "acc_y_rms": np.sqrt(
            np.mean(y**2)
        ),

        "acc_z_rms": np.sqrt(
            np.mean(z**2)
        ),

        # -------------------------------
        # Magnitude
        # -------------------------------

        "magnitude_mean": magnitude.mean(),

        "magnitude_std": magnitude.std(),

        "magnitude_rms": np.sqrt(
            np.mean(magnitude**2)
        ),

        "magnitude_max": magnitude.max(),

        "magnitude_min": magnitude.min()
    }


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("\n========================================")
print("Loading annotations")
print("========================================")

annotation_file = os.path.join(
    ROC_FOLDER,
    "AnnotID_ROC-011.csv"
)

annotations = pd.read_csv(
    annotation_file
)

annotations.columns = (
    annotations.columns
    .str.strip()
)

print("\nAll annotations:")
print(
    annotations[
        [
            "timestamp_start",
            "timestamp_stop",
            "name"
        ]
    ].to_string(index=False)
)

# ============================================================
# SELECT GAIT ACTIVITIES
# ============================================================

gait_annotations = annotations[
    annotations["name"].apply(
        is_gait_activity
    )
].copy()

print("\n========================================")
print("Gait annotations")
print("========================================")

print(
    gait_annotations[
        [
            "timestamp_start",
            "timestamp_stop",
            "name"
        ]
    ].to_string(index=False)
)

print(
    "\nNumber of gait annotations:",
    len(gait_annotations)
)

# ============================================================
# SENSOR FILES
# ============================================================

sensor_files = {

    "ch": os.path.join(
        ROC_FOLDER,
        "ch_ID_ROC-011_Accel.csv"
    ),

    "ll": os.path.join(
        ROC_FOLDER,
        "ll_ID_ROC-011_Accel.csv"
    ),

    "lt": os.path.join(
        ROC_FOLDER,
        "lt_ID_ROC-011_Accel.csv"
    )
}

# ============================================================
# PROCESS SENSORS
# ============================================================

all_results = []

for sensor_name, sensor_file in sensor_files.items():

    print("\n========================================")
    print("Processing sensor:", sensor_name)
    print("========================================")

    if not os.path.exists(sensor_file):

        print(
            "WARNING: File not found:",
            sensor_file
        )

        continue

    print(
        "File:",
        os.path.basename(sensor_file)
    )

    # --------------------------------------------------------
    # Read sensor in chunks
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            sensor_file,
            chunksize=200_000
        ),
        start=1
    ):

        print(
            "Reading chunk:",
            chunk_number
        )

        chunk.columns = (
            chunk.columns
            .str.strip()
        )

        timestamp = pd.to_numeric(
            chunk[
                "Timestamp (microseconds)"
            ],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Match annotations
        # ----------------------------------------------------

        for _, annotation in gait_annotations.iterrows():

            start = float(
                annotation[
                    "timestamp_start"
                ]
            )

            stop = float(
                annotation[
                    "timestamp_stop"
                ]
            )

            activity = str(
                annotation["name"]
            ).strip()

            mask = (
                (timestamp >= start) &
                (timestamp <= stop)
            )

            selected = chunk.loc[
                mask,
                [
                    "Timestamp (microseconds)",
                    "Accel X (g)",
                    "Accel Y (g)",
                    "Accel Z (g)"
                ]
            ]

            if len(selected) == 0:
                continue

            features = calculate_features(
                selected,
                activity,
                sensor_name
            )

            if features is not None:

                all_results.append(
                    features
                )

# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n========================================")
print("Creating result dataframe")
print("========================================")

result_df = pd.DataFrame(
    all_results
)

print(
    "\nRaw result shape:",
    result_df.shape
)

# ============================================================
# IMPORTANT:
# Combine duplicate rows belonging to the same
# sensor + activity.
# ============================================================

if len(result_df) > 0:

    numeric_columns = [
        "sample_count",
        "acc_x_mean",
        "acc_y_mean",
        "acc_z_mean",
        "acc_x_std",
        "acc_y_std",
        "acc_z_std",
        "acc_x_rms",
        "acc_y_rms",
        "acc_z_rms",
        "magnitude_mean",
        "magnitude_std",
        "magnitude_rms",
        "magnitude_max",
        "magnitude_min"
    ]

    result_df = (
        result_df
        .groupby(
            [
                "ID",
                "sensor",
                "activity"
            ],
            as_index=False
        )[numeric_columns]
        .mean()
    )

# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print(
    "Final shape:",
    result_df.shape
)

print(
    "\nSaved to:"
)

print(OUTPUT_FILE)

if len(result_df) > 0:

    print("\nRows by sensor:")

    print(
        result_df[
            "sensor"
        ].value_counts()
    )

    print("\nRows by activity:")

    print(
        result_df[
            "activity"
        ].value_counts()
    )

    print("\nFinal data:")

    print(
        result_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nWARNING: No features were extracted."
    )