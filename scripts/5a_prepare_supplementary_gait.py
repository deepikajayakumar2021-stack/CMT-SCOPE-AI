import os
import json
import random
import warnings

import numpy as np
import pandas as pd



PROJECT_ROOT = r"D:\CMT_PROJECT"

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "temp_inspect",
    "dataset",
    "data"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "processed",
    "supplementary_gait"
)

WINDOW_DIR = os.path.join(
    OUTPUT_ROOT,
    "windows"
)

MANIFEST_FILE = os.path.join(
    OUTPUT_ROOT,
    "supplementary_gait_manifest.csv"
)

SPLIT_FILE = os.path.join(
    OUTPUT_ROOT,
    "supplementary_gait_split.csv"
)

STATS_FILE = os.path.join(
    OUTPUT_ROOT,
    "supplementary_gait_statistics.csv"
)

SEED = 42

WINDOW_SIZE = 256
STRIDE = 128

EXPECTED_FREQ = 100.0


CHANNELS = [
    "Acc_X",
    "Acc_Y",
    "Acc_Z"
]



random.seed(SEED)
np.random.seed(SEED)



os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(WINDOW_DIR, exist_ok=True)



COHORTS = {
    "healthy": {
        "path": os.path.join(DATA_ROOT, "healthy", "HS"),
        "condition": "healthy"
    },

    "neuro": {
        "path": os.path.join(DATA_ROOT, "neuro"),
        "condition": None
    },

    "ortho": {
        "path": os.path.join(DATA_ROOT, "ortho", "KOA"),
        "condition": "KOA"
    }
}



def find_meta_file(trial_dir):
    files = os.listdir(trial_dir)

    for f in files:
        if f.endswith("_meta.json"):
            return os.path.join(trial_dir, f)

    return None



def find_rf_file(trial_dir):
    files = os.listdir(trial_dir)

    for f in files:

        if f.endswith("_raw_data_RF.txt"):
            return os.path.join(trial_dir, f)

    return None



def load_metadata(meta_file):

    try:

        with open(
            meta_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        warnings.warn(
            f"Could not read metadata: {meta_file}\n{e}"
        )

        return {}



def load_rf_acceleration(raw_file):

    try:

        df = pd.read_csv(
            raw_file,
            sep=r"\s+",
            engine="python"
        )

    except Exception as e:

        warnings.warn(
            f"Could not read RF file: {raw_file}\n{e}"
        )

        return None

    required = [
        "Acc_X",
        "Acc_Y",
        "Acc_Z"
    ]

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        warnings.warn(
            f"Missing columns {missing}: {raw_file}"
        )

        return None

    data = df[required].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    if len(data) == 0:
        return None

    return data.to_numpy(
        dtype=np.float32
    )


def normalize_window(window):

    """
    Per-window z-score normalization.

    Input:
        [256, 3]

    Output:
        [256, 3]
    """

    window = window.astype(
        np.float32,
        copy=True
    )

    mean = np.mean(
        window,
        axis=0,
        keepdims=True
    )

    std = np.std(
        window,
        axis=0,
        keepdims=True
    )

    std[std < 1e-6] = 1.0

    window = (
        window - mean
    ) / std

    return window.astype(
        np.float32
    )



def create_windows(signal):

    """
    Input:
        [N, 3]

    Output:
        list of [256, 3]
    """

    windows = []

    n_samples = signal.shape[0]

    if n_samples < WINDOW_SIZE:
        return windows

    start = 0

    while start + WINDOW_SIZE <= n_samples:

        end = start + WINDOW_SIZE

        window = signal[
            start:end
        ]

        windows.append(
            window
        )

        start += STRIDE

    return windows



records = []

print("=" * 70)
print("SUPPLEMENTARY GAIT DATASET PREPROCESSING")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Dataset root:")
print(DATA_ROOT)

print()
print("Window size:", WINDOW_SIZE)
print("Stride:", STRIDE)
print("Expected sampling frequency:", EXPECTED_FREQ)
print()



for cohort_name, config in COHORTS.items():

    cohort_path = config["path"]

    if not os.path.isdir(cohort_path):

        print(
            f"[WARNING] Missing cohort: {cohort_path}"
        )

        continue

    print()
    print("=" * 70)
    print(f"PROCESSING COHORT: {cohort_name}")
    print("=" * 70)

   
    if cohort_name == "neuro":

        

        condition_dirs = [
            d for d in os.listdir(cohort_path)
            if os.path.isdir(
                os.path.join(
                    cohort_path,
                    d
                )
            )
        ]

        subject_entries = []

        for condition in condition_dirs:

            condition_path = os.path.join(
                cohort_path,
                condition
            )

            subjects = [
                d for d in os.listdir(condition_path)
                if os.path.isdir(
                    os.path.join(
                        condition_path,
                        d
                    )
                )
            ]

            for subject in subjects:

                subject_entries.append(
                    (
                        subject,
                        condition,
                        os.path.join(
                            condition_path,
                            subject
                        )
                    )
                )

    else:

        subjects = [
            d for d in os.listdir(cohort_path)
            if os.path.isdir(
                os.path.join(
                    cohort_path,
                    d
                )
            )
        ]

        subject_entries = []

        for subject in subjects:

            subject_entries.append(
                (
                    subject,
                    config["condition"],
                    os.path.join(
                        cohort_path,
                        subject
                    )
                )
            )

    print(
        "Subjects found:",
        len(subject_entries)
    )




    for subject_id, condition, subject_path in subject_entries:

        trial_dirs = [
            d for d in os.listdir(subject_path)
            if os.path.isdir(
                os.path.join(
                    subject_path,
                    d
                )
            )
        ]

        for trial_id in sorted(trial_dirs):

            trial_path = os.path.join(
                subject_path,
                trial_id
            )

            meta_file = find_meta_file(
                trial_path
            )

            rf_file = find_rf_file(
                trial_path
            )

            if meta_file is None:

                print(
                    f"[SKIP] No metadata: "
                    f"{trial_id}"
                )

                continue

            if rf_file is None:

                print(
                    f"[SKIP] No RF sensor file: "
                    f"{trial_id}"
                )

                continue


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            meta = load_metadata(
                meta_file
            )

            frequency = meta.get(
                "freq",
                np.nan
            )

            if (
                frequency != frequency
                or frequency is None
            ):

                frequency = EXPECTED_FREQ

            try:
                frequency = float(
                    frequency
                )
            except:
                frequency = EXPECTED_FREQ


            # ------------------------------------------------
            # Load signal
            # ------------------------------------------------

            signal = load_rf_acceleration(
                rf_file
            )

            if signal is None:

                print(
                    f"[SKIP] Invalid signal: "
                    f"{trial_id}"
                )

                continue


            # ------------------------------------------------
            # Check frequency
            # ------------------------------------------------

            if abs(
                frequency - EXPECTED_FREQ
            ) > 1.0:

                print(
                    f"[WARNING] "
                    f"{trial_id}: "
                    f"{frequency} Hz"
                )


            # ------------------------------------------------
            # Create windows
            # ------------------------------------------------

            windows = create_windows(
                signal
            )

            if len(windows) == 0:

                print(
                    f"[SKIP] Too short: "
                    f"{trial_id}"
                )

                continue


            
            safe_trial_id = (
                f"{cohort_name}_"
                f"{condition}_"
                f"{trial_id}"
            )

            trial_window_count = 0

            for window_idx, window in enumerate(
                windows
            ):

                window = normalize_window(
                    window
                )

                
                window_model = window.T


                output_name = (
                    f"{safe_trial_id}_"
                    f"window_{window_idx:04d}.npz"
                )

                output_path = os.path.join(
                    WINDOW_DIR,
                    output_name
                )


                np.savez_compressed(
                    output_path,
                    signal=window_model,
                    subject_id=subject_id,
                    trial_id=trial_id,
                    cohort=cohort_name,
                    condition=condition,
                    sampling_rate=frequency,
                    window_index=window_idx
                )


                records.append({

                    "subject_id":
                        subject_id,

                    "trial_id":
                        trial_id,

                    "cohort":
                        cohort_name,

                    "condition":
                        condition,

                    "sampling_rate":
                        frequency,

                    "window_index":
                        window_idx,

                    "window_file":
                        output_name,

                    "n_samples":
                        WINDOW_SIZE,

                    "channels":
                        3
                })


                trial_window_count += 1


            print(
                f"[OK] "
                f"{subject_id} | "
                f"{trial_id} | "
                f"{len(signal)} samples | "
                f"{trial_window_count} windows"
            )


# ============================================================
# 13. CREATE MANIFEST
# ============================================================

if len(records) == 0:

    raise RuntimeError(
        "No valid gait windows were created."
    )


manifest = pd.DataFrame(
    records
)


manifest.to_csv(
    MANIFEST_FILE,
    index=False
)


# ============================================================
# 14. CREATE PATIENT-LEVEL SPLIT
# ============================================================

subjects = (
    manifest[
        [
            "subject_id",
            "cohort",
            "condition"
        ]
    ]
    .drop_duplicates(
        subset=["subject_id"]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Shuffle subjects
# ------------------------------------------------------------

subjects = subjects.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)


# ------------------------------------------------------------
# Split 70 / 15 / 15
# ------------------------------------------------------------

n_subjects = len(subjects)

n_train = int(
    0.70 * n_subjects
)

n_val = int(
    0.15 * n_subjects
)


def assign_split(index):

    if index < n_train:
        return "train"

    elif index < n_train + n_val:
        return "val"

    else:
        return "test"


subjects["split"] = [
    assign_split(i)
    for i in range(n_subjects)
]


# ------------------------------------------------------------
# Save split file
# ------------------------------------------------------------

subjects.to_csv(
    SPLIT_FILE,
    index=False
)


# ============================================================
# 15. MERGE SPLIT INTO MANIFEST
# ============================================================

manifest = manifest.merge(
    subjects[
        [
            "subject_id",
            "split"
        ]
    ],
    on="subject_id",
    how="left"
)


manifest.to_csv(
    MANIFEST_FILE,
    index=False
)


# ============================================================
# 16. STATISTICS
# ============================================================

stats = []

for split_name in [
    "train",
    "val",
    "test"
]:

    subset = manifest[
        manifest["split"] == split_name
    ]

    stats.append({

        "split":
            split_name,

        "unique_subjects":
            subset[
                "subject_id"
            ].nunique(),

        "unique_trials":
            subset[
                "trial_id"
            ].nunique(),

        "windows":
            len(subset),

        "cohorts":
            subset[
                "cohort"
            ].nunique(),

        "conditions":
            subset[
                "condition"
            ].nunique()
    })


stats_df = pd.DataFrame(
    stats
)


stats_df.to_csv(
    STATS_FILE,
    index=False
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print()
print(
    "Unique subjects:",
    manifest["subject_id"].nunique()
)

print(
    "Unique trials:",
    manifest["trial_id"].nunique()
)

print(
    "Total windows:",
    len(manifest)
)

print()

print(
    manifest.groupby(
        [
            "cohort",
            "condition"
        ]
    )[
        "subject_id"
    ]
    .nunique()
)

print()
print("Split statistics:")
print(stats_df)

print()
print("Manifest:")
print(MANIFEST_FILE)

print()
print("Windows:")
print(WINDOW_DIR)

print()
print("Split:")
print(SPLIT_FILE)

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print(
    "These supplementary subjects are NOT CMT1A labels."
)
print(
    "Use them only for general gait representation "
    "pretraining."
)
print("=" * 70)