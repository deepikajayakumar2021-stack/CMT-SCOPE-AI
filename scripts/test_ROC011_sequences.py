import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\CMT_PROJECT\Baseline_CMT1A-BioStampNPoint2023"
PATIENT_ID = "ROC-011"

PATIENT_DIR = os.path.join(BASE_DIR, PATIENT_ID)

ANNOT_FILE = os.path.join(
    PATIENT_DIR,
    f"AnnotID_{PATIENT_ID}.csv"
)

OUTPUT_DIR = r"D:\CMT_PROJECT\processed\ROC011_sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Number of samples in one sequence
WINDOW_SIZE = 64

# Overlap between consecutive windows
# 64 means 50% overlap
STEP_SIZE = 32

# Sensors
SENSORS = ["ch", "ll", "lt"]


# ============================================================
# GAIT ACTIVITIES
# ============================================================

GAIT_KEYWORDS = [
    "10m walk",
    "10m walk/run",
    "TUG",
    "6MWT",
    "Stairs (ascending)",
    "Stairs (descending)"
]


# ============================================================
# FUNCTION: CHECK GAIT ACTIVITY
# ============================================================

def is_gait_activity(activity):

    activity = str(activity).strip().lower()

    for keyword in GAIT_KEYWORDS:
        if keyword.lower() in activity:
            return True

    return False


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("=" * 70)
print("ROC-011 TIME-SERIES SEQUENCE EXTRACTION")
print("=" * 70)

print("\nLoading annotation file...")
print(ANNOT_FILE)

annotations = pd.read_csv(ANNOT_FILE)

print("\nAnnotation columns:")
print(list(annotations.columns))


# ============================================================
# FILTER GAIT ACTIVITIES
# ============================================================

gait_annotations = annotations[
    annotations["name"].apply(is_gait_activity)
].copy()

print("\nNumber of gait annotations:", len(gait_annotations))

print("\nGait activities found:")
print(gait_annotations["name"].tolist())


# ============================================================
# CONVERT TIMESTAMPS
# ============================================================

gait_annotations["timestamp_start"] = pd.to_numeric(
    gait_annotations["timestamp_start"],
    errors="coerce"
)

gait_annotations["timestamp_stop"] = pd.to_numeric(
    gait_annotations["timestamp_stop"],
    errors="coerce"
)

gait_annotations = gait_annotations.dropna(
    subset=["timestamp_start", "timestamp_stop"]
)


# ============================================================
# PROCESS EACH SENSOR
# ============================================================

all_sequence_info = []

for sensor in SENSORS:

    sensor_file = os.path.join(
        PATIENT_DIR,
        f"{sensor}_ID_{PATIENT_ID}_Accel.csv"
    )

    print("\n" + "=" * 70)
    print("PROCESSING SENSOR:", sensor)
    print("=" * 70)

    if not os.path.exists(sensor_file):
        print("WARNING: Sensor file not found:")
        print(sensor_file)
        continue

    print("Sensor file:")
    print(sensor_file)

    # --------------------------------------------------------
    # Load sensor data
    # --------------------------------------------------------

    print("\nReading sensor data...")

    sensor_df = pd.read_csv(sensor_file)

    print("Total sensor samples:", len(sensor_df))

    # Required columns
    required_columns = [
        "Timestamp (microseconds)",
        "Accel X (g)",
        "Accel Y (g)",
        "Accel Z (g)"
    ]

    missing = [
        col for col in required_columns
        if col not in sensor_df.columns
    ]

    if missing:
        print("ERROR: Missing columns:")
        print(missing)
        continue

    # --------------------------------------------------------
    # Convert timestamp and acceleration
    # --------------------------------------------------------

    sensor_df["Timestamp (microseconds)"] = pd.to_numeric(
        sensor_df["Timestamp (microseconds)"],
        errors="coerce"
    )

    for col in [
        "Accel X (g)",
        "Accel Y (g)",
        "Accel Z (g)"
    ]:
        sensor_df[col] = pd.to_numeric(
            sensor_df[col],
            errors="coerce"
        )

    sensor_df = sensor_df.dropna(
        subset=[
            "Timestamp (microseconds)",
            "Accel X (g)",
            "Accel Y (g)",
            "Accel Z (g)"
        ]
    )

    timestamps = sensor_df["Timestamp (microseconds)"].to_numpy()

    acceleration = sensor_df[
        [
            "Accel X (g)",
            "Accel Y (g)",
            "Accel Z (g)"
        ]
    ].to_numpy(dtype=np.float32)


    # --------------------------------------------------------
    # Process each gait annotation
    # --------------------------------------------------------

    for annotation_index, annotation in gait_annotations.iterrows():

        activity = str(annotation["name"]).strip()

        start_time = annotation["timestamp_start"]
        stop_time = annotation["timestamp_stop"]

        print(
            f"\nActivity: {activity}"
        )

        print(
            f"Timestamp: {start_time:.0f} -> {stop_time:.0f}"
        )

        # ----------------------------------------------------
        # Find samples belonging to this activity
        # ----------------------------------------------------

        mask = (
            (timestamps >= start_time)
            &
            (timestamps <= stop_time)
        )

        activity_data = acceleration[mask]

        print(
            "Samples found:",
            len(activity_data)
        )

        # ----------------------------------------------------
        # Check enough samples
        # ----------------------------------------------------

        if len(activity_data) < WINDOW_SIZE:

            print(
                "Not enough samples for a 128-sample window."
            )

            continue


        # ----------------------------------------------------
        # Create sliding windows
        # ----------------------------------------------------

        windows = []

        for start in range(
            0,
            len(activity_data) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            end = start + WINDOW_SIZE

            window = activity_data[start:end]

            windows.append(window)


        windows = np.asarray(
            windows,
            dtype=np.float32
        )

        print(
            "Windows created:",
            len(windows)
        )

        print(
            "Window shape:",
            windows.shape
        )


        # ----------------------------------------------------
        # Create safe activity name
        # ----------------------------------------------------

        safe_activity = (
            activity
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )


        # ----------------------------------------------------
        # Save sequence file
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{PATIENT_ID}_{sensor}_{safe_activity}.npy"
        )

        np.save(
            output_file,
            windows
        )

        print(
            "Saved:",
            output_file
        )


        # ----------------------------------------------------
        # Save information
        # ----------------------------------------------------

        all_sequence_info.append({

            "ID": PATIENT_ID,

            "sensor": sensor,

            "activity": activity,

            "samples": len(activity_data),

            "window_size": WINDOW_SIZE,

            "step_size": STEP_SIZE,

            "number_of_windows": len(windows),

            "shape": str(windows.shape),

            "file": output_file
        })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_sequence_info)

summary_file = os.path.join(
    OUTPUT_DIR,
    "ROC011_sequence_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE EXTRACTION COMPLETED")
print("=" * 70)

print("\nTotal sequence files created:")
print(len(summary_df))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSummary file:")
print(summary_file)

print("\nSequence summary:")

if len(summary_df) > 0:
    print(
        summary_df[
            [
                "sensor",
                "activity",
                "samples",
                "number_of_windows",
                "shape"
            ]
        ].to_string(index=False)
    )
else:
    print("No sequences were created.")