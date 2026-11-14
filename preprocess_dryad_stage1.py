import re
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_DIR = Path(r"D:\CMT_PROJECT")

BASELINE_DIR = PROJECT_DIR / "Baseline_CMT1A-BioStampNPoint2023"
LONGITUDINAL_DIR = PROJECT_DIR / "Longitudinal_CMT1A-BioStampNPoint2023"

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



CHUNK_SIZE = 100_000

GAIT_ACTIVITIES = {
    "10m walk",
    "10m walk/run",
    "tug",
    "6mwt",
    "stairs (ascending)",
    "stairs (descending)"
}




def clean_column_name(column):
    column = str(column).strip().lower()
    column = column.replace("\ufeff", "")
    return column


def find_roc_folders(root):

    folders = []

    if not root.exists():
        print(f"ERROR: Folder not found: {root}")
        return folders

    for item in root.iterdir():

        if (
            item.is_dir()
            and re.fullmatch(
                r"ROC-\d+",
                item.name,
                re.IGNORECASE
            )
        ):
            folders.append(item)

    return sorted(
        folders,
        key=lambda x: x.name
    )
def find_files(roc_folder):

    accel_files = []
    annotation_file = None

    for file in roc_folder.glob("*.csv"):

        name = file.name.lower()

        if "annotid" in name:
            annotation_file = file

        elif "accel" in name:
            accel_files.append(file)

    return accel_files, annotation_file


def load_annotations(annotation_file):

    print(f"Reading annotation: {annotation_file.name}")

    annotations = pd.read_csv(annotation_file)

    annotations.columns = [
        clean_column_name(c)
        for c in annotations.columns
    ]

    print("Annotation columns:", list(annotations.columns))

    if "timestamp_start" not in annotations.columns:
        raise ValueError(
            "timestamp_start column not found"
        )

    if "name" not in annotations.columns:
        raise ValueError(
            "name column not found"
        )

  
    annotations["start_us"] = pd.to_numeric(
        annotations["timestamp_start"],
        errors="coerce"
    )

    if "timestamp" in annotations.columns:

        annotations["end_us"] = pd.to_numeric(
            annotations["timestamp"],
            errors="coerce"
        )

    else:
        annotations["end_us"] = np.nan


    if "duration" in annotations.columns:

        duration_seconds = pd.to_timedelta(
            annotations["duration"],
            errors="coerce"
        ).dt.total_seconds()

        missing_end = annotations["end_us"].isna()

        annotations.loc[
            missing_end,
            "end_us"
        ] = (
            annotations.loc[
                missing_end,
                "start_us"
            ]
            + duration_seconds.loc[
                missing_end
            ] * 1_000_000
        )

    # Activity
    annotations["activity"] = (
        annotations["name"]
        .astype(str)
        .str.strip()
    )

    
    activity_lower = (
        annotations["activity"]
        .str.lower()
        .str.strip()
    )

    annotations = annotations[
        activity_lower.isin(GAIT_ACTIVITIES)
    ].copy()

    # Remove invalid timestamps
    annotations = annotations.dropna(
        subset=[
            "start_us",
            "end_us"
        ]
    )

    annotations = annotations[
        annotations["end_us"] > annotations["start_us"]
    ]

    annotations.reset_index(
        drop=True,
        inplace=True
    )

    return annotations




def process_accelerometer(
    accel_file,
    annotations,
    participant_id,
    dataset_type
):

    print(f"Processing: {accel_file.name}")

    extracted = []

    total_rows = 0

    for chunk in pd.read_csv(
        accel_file,
        chunksize=CHUNK_SIZE
    ):

        chunk.columns = [
            clean_column_name(c)
            for c in chunk.columns
        ]

        timestamp_col = None
        x_col = None
        y_col = None
        z_col = None

        # Find columns
        for col in chunk.columns:

            if "timestamp" in col:
                timestamp_col = col

            elif col in [
                "accel x (g)",
                "accel_x"
            ]:
                x_col = col

            elif col in [
                "accel y (g)",
                "accel_y"
            ]:
                y_col = col

            elif col in [
                "accel z (g)",
                "accel_z"
            ]:
                z_col = col

        if not all([
            timestamp_col,
            x_col,
            y_col,
            z_col
        ]):

            print("ERROR: Required columns not found.")

            print(
                "Available columns:",
                list(chunk.columns)
            )

            return None

        # Convert to numeric
        for col in [
            timestamp_col,
            x_col,
            y_col,
            z_col
        ]:

            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

        chunk.dropna(
            subset=[
                timestamp_col,
                x_col,
                y_col,
                z_col
            ],
            inplace=True
        )

        if len(chunk) == 0:
            continue

        total_rows += len(chunk)

        # Match annotations
        for annotation_id, annotation in annotations.iterrows():

            start_us = annotation["start_us"]
            end_us = annotation["end_us"]

            mask = (
                (chunk[timestamp_col] >= start_us)
                &
                (chunk[timestamp_col] <= end_us)
            )

            selected = chunk.loc[
                mask,
                [
                    timestamp_col,
                    x_col,
                    y_col,
                    z_col
                ]
            ].copy()

            if len(selected) == 0:
                continue

            # Standard column names
            selected.rename(
                columns={
                    timestamp_col: "timestamp_us",
                    x_col: "acc_x_g",
                    y_col: "acc_y_g",
                    z_col: "acc_z_g"
                },
                inplace=True
            )

            # Acceleration magnitude
            selected["acc_magnitude_g"] = np.sqrt(
                selected["acc_x_g"] ** 2
                + selected["acc_y_g"] ** 2
                + selected["acc_z_g"] ** 2
            )

            # Metadata
            selected["participant_id"] = participant_id
            selected["dataset_type"] = dataset_type
            selected["sensor_file"] = accel_file.name
            selected["activity"] = annotation["activity"]
            selected["annotation_id"] = annotation_id

            selected["annotation_start_us"] = start_us
            selected["annotation_end_us"] = end_us

            extracted.append(selected)

    if not extracted:

        print("No gait data extracted.")

        return None

    result = pd.concat(
        extracted,
        ignore_index=True
    )

    # Remove duplicates
    result.drop_duplicates(
        subset=[
            "participant_id",
            "sensor_file",
            "timestamp_us",
            "activity"
        ],
        inplace=True
    )

    result.sort_values(
        [
            "participant_id",
            "sensor_file",
            "timestamp_us"
        ],
        inplace=True
    )

    print(f"Total rows read: {total_rows:,}")
    print(f"Gait rows extracted: {len(result):,}")

    return result


def process_participant(
    roc_folder,
    dataset_type
):

    participant_id = roc_folder.name

    print("\n" + "=" * 65)
    print(
        f"{dataset_type.upper()} | {participant_id}"
    )
    print("=" * 65)

    accel_files, annotation_file = find_files(
        roc_folder
    )

    if annotation_file is None:

        print("Annotation file not found.")

        return []

    if not accel_files:

        print("Accelerometer files not found.")

        return []

    annotations = load_annotations(
        annotation_file
    )

    print(
        f"Gait annotations: {len(annotations)}"
    )

    if len(annotations) == 0:
        return []

    results = []

    for accel_file in accel_files:

        result = process_accelerometer(
            accel_file,
            annotations,
            participant_id,
            dataset_type
        )

        if result is not None:
            results.append(result)

    return results
def process_dataset(
    dataset_dir,
    dataset_type
):

    roc_folders = find_roc_folders(
        dataset_dir
    )

    print("\n" + "#" * 65)
    print(f"{dataset_type.upper()} DATASET")
    print(
        f"Participants found: {len(roc_folders)}"
    )
    print("#" * 65)

    all_results = []

    for roc_folder in roc_folders:

        results = process_participant(
            roc_folder,
            dataset_type
        )

        all_results.extend(results)

    if not all_results:
        return None

    return pd.concat(
        all_results,
        ignore_index=True
    )



def analyze_sampling_rate(data):

    print("\n" + "=" * 65)
    print("SAMPLING RATE ANALYSIS")
    print("=" * 65)

    results = []

    group_columns = [
        "participant_id",
        "dataset_type",
        "sensor_file"
    ]

    for keys, group in data.groupby(
        group_columns
    ):

        timestamps = (
            group["timestamp_us"]
            .sort_values()
            .drop_duplicates()
        )

        if len(timestamps) < 2:
            continue

        differences = (
            timestamps.diff()
            .dropna()
        )

        differences = differences[
            differences > 0
        ]

        if len(differences) == 0:
            continue

        median_interval = differences.median()

        frequency = (
            1_000_000 / median_interval
        )

        results.append({
            "participant_id": keys[0],
            "dataset_type": keys[1],
            "sensor_file": keys[2],
            "median_interval_us": median_interval,
            "estimated_sampling_rate_hz": frequency,
            "number_of_samples": len(timestamps)
        })

    sampling = pd.DataFrame(results)

    if len(sampling) > 0:

        print("\nSampling rate summary:")

        print(
            sampling[
                "estimated_sampling_rate_hz"
            ].describe()
        )

    return sampling

def main():

    print("\n")
    print("=" * 65)
    print("CMT1A DRYAD STAGE-1 PREPROCESSING")
    print("=" * 65)

    BASELINE_DIR = (
    PROJECT_DIR
    / "Baseline_CMT1A-BioStampNPoint2023"
    / "Baseline_CMT1A-BioStampNPoint2023"
)

LONGITUDINAL_DIR = (
    PROJECT_DIR
    / "Longitudinal_CMT1A-BioStampNPoint2023"
    / "Longitudinal_CMT1A-BioStampNPoint2023"
)
    datasets = []

    if baseline_data is not None:
        datasets.append(baseline_data)

    if longitudinal_data is not None:
        datasets.append(longitudinal_data)

    if not datasets:

        print(
            "\nERROR: No gait data was extracted."
        )

        return

    # Combine baseline + longitudinal
    gait_data = pd.concat(
        datasets,
        ignore_index=True
    )

    gait_data.sort_values(
        [
            "participant_id",
            "dataset_type",
            "sensor_file",
            "timestamp_us"
        ],
        inplace=True
    )

    # Save processed gait data
    output_file = (
        OUTPUT_DIR /
        "dryad_gait_annotated.csv"
    )

    gait_data.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:")
    print(output_file)

    # Sampling rate
    sampling = analyze_sampling_rate(
        gait_data
    )

    sampling_file = (
        OUTPUT_DIR /
        "dryad_sampling_rate.csv"
    )

    sampling.to_csv(
        sampling_file,
        index=False
    )

    print("\nSaved:")
    print(sampling_file)

    # Activity summary
    summary = (
        gait_data
        .groupby(
            [
                "participant_id",
                "dataset_type",
                "activity"
            ]
        )
        .size()
        .reset_index(
            name="sample_count"
        )
    )

    summary_file = (
        OUTPUT_DIR /
        "dryad_gait_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nSaved:")
    print(summary_file)

    # Final information
    print("\n" + "=" * 65)
    print("STAGE 1 COMPLETED")
    print("=" * 65)

    print(
        f"\nParticipants: "
        f"{gait_data['participant_id'].nunique()}"
    )

    print(
        f"Total extracted rows: "
        f"{len(gait_data):,}"
    )

    print("\nActivities:")

    print(
        gait_data["activity"].value_counts()
    )

    print("\nOutput folder:")
    print(OUTPUT_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()