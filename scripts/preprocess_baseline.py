import os
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_DIR = Path(
    r"D:\CMT_PROJECT\Baseline_CMT1A-BioStampNPoint2023"
)

OUTPUT_DIR = Path(
    r"D:\CMT_PROJECT\processed\gait_windows"
)

LOG_DIR = Path(
    r"D:\CMT_PROJECT\logs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GAIT ACTIVITIES
# ============================================================

GAIT_ACTIVITIES = {
    "10m walk",
    "10m walk/run",
    "tug",
    "6mwt",
    "stairs (ascending)",
    "stairs (descending)",
}


# ============================================================
# WINDOW PARAMETERS
# ============================================================

WINDOW_SIZE = 256
STEP_SIZE = 128

# Number of rows read from CSV at one time.
# 250,000 is usually a good balance between speed and RAM.
CHUNK_SIZE = 250_000


# ============================================================
# REQUIRED SENSOR COLUMNS
# ============================================================

SENSOR_COLUMNS = [
    "Timestamp (microseconds)",
    "Accel X (g)",
    "Accel Y (g)",
    "Accel Z (g)",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_activity_name(name):
    """
    Convert activity name to lowercase standardized format.
    """

    if pd.isna(name):
        return ""

    return str(name).strip().lower()


# ------------------------------------------------------------
# Find annotation file
# ------------------------------------------------------------

def find_annotation_file(patient_dir):
    """
    Find AnnotID_*.csv inside patient directory.
    """

    files = list(patient_dir.glob("AnnotID_*.csv"))

    if not files:
        return None

    return files[0]


# ------------------------------------------------------------
# Find sensor files
# ------------------------------------------------------------

def find_sensor_files(patient_dir):
    """
    Find all accelerometer CSV files.
    """

    return sorted(
        patient_dir.glob("*_Accel.csv")
    )


# ------------------------------------------------------------
# Read annotations
# ------------------------------------------------------------

def read_annotations(annotation_file):

    print(
        f"Reading annotations: {annotation_file.name}"
    )

    ann = pd.read_csv(
        annotation_file
    )

    required = [
        "timestamp_start",
        "timestamp_stop",
        "name",
    ]

    missing = [
        col for col in required
        if col not in ann.columns
    ]

    if missing:

        raise ValueError(
            f"Missing annotation columns: {missing}"
        )

    # Normalize activity name

    ann["activity"] = ann["name"].apply(
        normalize_activity_name
    )

    # Keep only relevant gait activities

    ann = ann[
        ann["activity"].isin(
            GAIT_ACTIVITIES
        )
    ].copy()

    # Remove invalid timestamps

    ann = ann.dropna(
        subset=[
            "timestamp_start",
            "timestamp_stop",
        ]
    )

    # Ensure correct ordering

    ann = ann.sort_values(
        "timestamp_start"
    ).reset_index(drop=True)

    return ann


# ============================================================
# WINDOW CREATION
# ============================================================

def create_windows_from_array(values):
    """
    Convert acceleration array into overlapping
    windows of shape:

        (N_windows, 256, 3)
    """

    if len(values) < WINDOW_SIZE:

        return np.empty(
            (0, WINDOW_SIZE, 3),
            dtype=np.float32,
        )

    n_windows = (
        (len(values) - WINDOW_SIZE)
        // STEP_SIZE
    ) + 1

    windows = np.empty(
        (
            n_windows,
            WINDOW_SIZE,
            3,
        ),
        dtype=np.float32,
    )

    for i in range(n_windows):

        start = i * STEP_SIZE
        end = start + WINDOW_SIZE

        windows[i] = values[start:end]

    return windows


# ============================================================
# PROCESS ONE SENSOR CHUNK
# ============================================================

def process_chunk(
    chunk,
    annotations,
    patient_windows,
    patient_labels,
    sensor_file_name,
):
    """
    Process one sensor CSV chunk.

    Only samples that overlap with gait annotations
    are retained.

    This prevents the complete sensor CSV from being
    loaded into RAM.
    """

    if chunk.empty:
        return

    # Remove missing values

    chunk = chunk.dropna(
        subset=SENSOR_COLUMNS
    )

    if chunk.empty:
        return

    timestamp = chunk[
        "Timestamp (microseconds)"
    ].to_numpy()

    # --------------------------------------------------------
    # Check each gait annotation
    # --------------------------------------------------------

    for _, ann in annotations.iterrows():

        activity = ann["activity"]

        start_time = ann[
            "timestamp_start"
        ]

        stop_time = ann[
            "timestamp_stop"
        ]

        # Find samples belonging to this annotation

        mask = (
            (timestamp >= start_time)
            &
            (timestamp <= stop_time)
        )

        if not np.any(mask):
            continue

        segment = chunk.loc[mask]

        if len(segment) < WINDOW_SIZE:
            continue

        values = segment[
            [
                "Accel X (g)",
                "Accel Y (g)",
                "Accel Z (g)",
            ]
        ].to_numpy(
            dtype=np.float32
        )

        windows = create_windows_from_array(
            values
        )

        if len(windows) == 0:
            continue

        patient_windows.append(
            windows
        )

        patient_labels.extend(
            [activity] * len(windows)
        )


# ============================================================
# PROCESS ONE SENSOR CSV
# ============================================================

def process_sensor_file(
    sensor_file,
    annotations,
    patient_windows,
    patient_labels,
):
    """
    Read a large sensor CSV in chunks.

    IMPORTANT:
    The entire CSV is NEVER concatenated into RAM.
    """

    print(
        f"\nReading: {sensor_file.name}"
    )

    total_rows = 0
    chunk_number = 0

    try:

        reader = pd.read_csv(
            sensor_file,
            usecols=SENSOR_COLUMNS,
            chunksize=CHUNK_SIZE,
        )

        for chunk in reader:

            chunk_number += 1

            total_rows += len(chunk)

            process_chunk(
                chunk,
                annotations,
                patient_windows,
                patient_labels,
                sensor_file.name,
            )

            if chunk_number % 5 == 0:

                print(
                    f"  Processed chunks: "
                    f"{chunk_number:,} | "
                    f"Rows: {total_rows:,}"
                )

    except ValueError as e:

        print(
            f"❌ Column error in "
            f"{sensor_file.name}"
        )

        raise e

    print(
        f"  Finished: {sensor_file.name}"
    )

    print(
        f"  Total rows scanned: "
        f"{total_rows:,}"
    )


# ============================================================
# PROCESS ONE PATIENT
# ============================================================

def process_patient(patient_dir):

    patient_id = patient_dir.name

    print("\n" + "=" * 70)
    print(
        f"Processing {patient_id}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    annotation_file = find_annotation_file(
        patient_dir
    )

    sensor_files = find_sensor_files(
        patient_dir
    )

    if annotation_file is None:

        print(
            "❌ Annotation file missing"
        )

        return 0

    if not sensor_files:

        print(
            "❌ Sensor files missing"
        )

        return 0

    print(
        f"Annotation: "
        f"{annotation_file.name}"
    )

    print(
        f"Sensor files: "
        f"{len(sensor_files)}"
    )

    # --------------------------------------------------------
    # Read annotations
    # --------------------------------------------------------

    annotations = read_annotations(
        annotation_file
    )

    print(
        f"Gait annotations: "
        f"{len(annotations)}"
    )

    if annotations.empty:

        print(
            "⚠ No gait annotations found"
        )

        return 0

    # Display activities

    print(
        "Activities found:"
    )

    for activity in sorted(
        annotations["activity"].unique()
    ):

        count = (
            annotations["activity"]
            == activity
        ).sum()

        print(
            f"  {activity}: {count}"
        )

    # --------------------------------------------------------
    # Store windows
    # --------------------------------------------------------

    patient_windows = []
    patient_labels = []

    # --------------------------------------------------------
    # Process every sensor CSV
    # --------------------------------------------------------

    for sensor_file in sensor_files:

        process_sensor_file(
            sensor_file,
            annotations,
            patient_windows,
            patient_labels,
        )

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    if not patient_windows:

        print(
            f"\n❌ No usable windows "
            f"for {patient_id}"
        )

        return 0

    # --------------------------------------------------------
    # Combine windows from chunks/files
    # --------------------------------------------------------

    X = np.concatenate(
        patient_windows,
        axis=0,
    )

    y = np.array(
        patient_labels,
        dtype=str,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"{patient_id}_gait.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        patient_id=patient_id,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n✅ Saved:")
    print(
        f"   {output_file}"
    )

    print(
        f"   Windows: {len(X):,}"
    )

    print(
        f"   Shape: {X.shape}"
    )

    print(
        f"   Output size: "
        f"{output_file.stat().st_size / (1024 ** 2):.2f} MB"
    )

    # Activity distribution

    print(
        "\nWindow distribution:"
    )

    unique_labels, counts = np.unique(
        y,
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):

        print(
            f"   {label:25s}: "
            f"{count:,}"
        )

    return len(X)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CMT1A BASELINE GAIT PREPROCESSING")
    print("OPTIMIZED CHUNK-BASED VERSION")
    print("=" * 70)

    print(
        f"\nBaseline directory:"
    )

    print(
        BASELINE_DIR
    )

    print(
        f"\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nChunk size:"
        f" {CHUNK_SIZE:,} rows"
    )

    print(
        f"Window size:"
        f" {WINDOW_SIZE}"
    )

    print(
        f"Step size:"
        f" {STEP_SIZE}"
    )

    # --------------------------------------------------------
    # Check baseline directory
    # --------------------------------------------------------

    if not BASELINE_DIR.exists():

        print(
            "\n❌ BASELINE DIRECTORY NOT FOUND"
        )

        print(
            BASELINE_DIR
        )

        return

    # --------------------------------------------------------
    # Find patients
    # --------------------------------------------------------

    patients = sorted(
        [
            p
            for p in BASELINE_DIR.iterdir()
            if p.is_dir()
            and p.name.upper().startswith(
                "ROC-"
            )
        ]
    )

    print(
        f"\nPatients found: "
        f"{len(patients)}"
    )

    if not patients:

        print(
            "❌ No ROC patient folders found."
        )

        return

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    total_windows = 0
    successful = 0

    log_rows = []

    for index, patient in enumerate(
        patients,
        start=1,
    ):

        print(
            f"\n\nPatient "
            f"{index}/{len(patients)}"
        )

        try:

            windows = process_patient(
                patient
            )

            if windows > 0:

                successful += 1

                status = "success"

            else:

                status = "no_windows"

            total_windows += windows

            log_rows.append(
                {
                    "patient": patient.name,
                    "windows": windows,
                    "status": status,
                }
            )

        except KeyboardInterrupt:

            print(
                "\n\n⚠ Processing interrupted "
                "by user."
            )

            print(
                "Already-created patient files "
                "remain saved."
            )

            break

        except Exception as e:

            print(
                f"\n❌ ERROR in "
                f"{patient.name}:"
            )

            print(
                str(e)
            )

            log_rows.append(
                {
                    "patient": patient.name,
                    "windows": 0,
                    "status":
                        f"error: {e}",
                }
            )

    # --------------------------------------------------------
    # Save processing log
    # --------------------------------------------------------

    log_file = (
        LOG_DIR
        / "baseline_preprocessing_log.csv"
    )

    pd.DataFrame(
        log_rows
    ).to_csv(
        log_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Patients found: "
        f"{len(patients)}"
    )

    print(
        f"Patients processed successfully: "
        f"{successful}"
    )

    print(
        f"Total gait windows: "
        f"{total_windows:,}"
    )

    print(
        f"\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nLog:"
    )

    print(
        log_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()