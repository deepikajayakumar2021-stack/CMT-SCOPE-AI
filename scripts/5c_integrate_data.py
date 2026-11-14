import os
import re
import zipfile
import tempfile
import shutil
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"D:\CMT_PROJECT"

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "processed"
)

SPLIT_DIR = os.path.join(
    PROCESSED_DIR,
    "splits"
)

GAIT_DIR = os.path.join(
    PROCESSED_DIR,
    "gait_windows"
)

INTEGRATED_DIR = os.path.join(
    PROCESSED_DIR,
    "integrated"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(GAIT_DIR, exist_ok=True)
os.makedirs(INTEGRATED_DIR, exist_ok=True)


CLINICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "ClinicData_CMT1A-BioStampNPoint2023.csv"
)

BASELINE_ZIP = os.path.join(
    PROJECT_ROOT,
    "Baseline_CMT1A-BioStampNPoint2023.zip"
)

LONGITUDINAL_ZIP = os.path.join(
    PROJECT_ROOT,
    "Longitudinal_CMT1A-BioStampNPoint2023.zip"
)



MULTI_GAIT_FILE = os.path.join(
    PROCESSED_DIR,
    "multi_gait_windows.npz"
)

SECONDARY_GAIT_FILE = os.path.join(
    PROCESSED_DIR,
    "secondary_gait_windows.npz"
)

COMBINED_GAIT_FILE = os.path.join(
    PROCESSED_DIR,
    "combined_gait_windows.npz"
)



EXTERNAL_SEARCH_DIRS = [

    PROJECT_ROOT,

    os.path.join(
        PROJECT_ROOT,
        "datasets"
    ),

    os.path.join(
        PROJECT_ROOT,
        "data"
    ),

    os.path.join(
        PROJECT_ROOT,
        "raw_data"
    ),

    os.path.join(
        PROJECT_ROOT,
        "external_datasets"
    ),

]


EXTERNAL_DATASET_PATTERNS = {

    "OU_InertialGaitData": [
        "OU-IneritialGaitData",
        "OU-InertialGaitData",
        "OU_ISIR",
        "OU-Inertial"
    ],

    "WBDSascii": [
        "WBDSascii",
        "WBDS_ascii"
    ],

    "WBDSc3d": [
        "WBDSc3d",
        "WBDS_c3d"
    ],

    "dataset_zip": [
        "dataset.zip"
    ],

}

WINDOW_SIZE = 256
CHANNELS = 3

MAX_WINDOWS_PER_EXTERNAL_SOURCE = 100000

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)



# 4. PRINT HEADER

print()
print("=" * 78)
print("CMT-SCOPE AI - COMPLETE DATASET INTEGRATION")
print("=" * 78)
print()

print("Project:")
print(PROJECT_ROOT)

print()

print("Output:")
print(INTEGRATED_DIR)

print()


def safe_float_array(array):
    """
    Convert array to float32 and remove invalid values.
    """

    arr = np.asarray(
        array,
        dtype=np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return arr


# ------------------------------------------------------------
# Find numeric columns
# ------------------------------------------------------------

def find_numeric_columns(df):

    numeric_cols = []

    for col in df.columns:

        try:

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            valid_count = converted.notna().sum()

            if valid_count > 20:
                numeric_cols.append(
                    col
                )

        except Exception:
            pass

    return numeric_cols


# ------------------------------------------------------------
# Find accelerometer columns
# ------------------------------------------------------------

def detect_accelerometer_columns(df):

    columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    original = list(df.columns)

    x_candidates = []
    y_candidates = []
    z_candidates = []

    for i, col in enumerate(columns):

        # X
        if (
            "accel x" in col
            or "accelerometer x" in col
            or "acc_x" in col
            or "accx" in col
            or col == "x"
        ):
            x_candidates.append(
                original[i]
            )

        # Y
        if (
            "accel y" in col
            or "accelerometer y" in col
            or "acc_y" in col
            or "accy" in col
            or col == "y"
        ):
            y_candidates.append(
                original[i]
            )

        # Z
        if (
            "accel z" in col
            or "accelerometer z" in col
            or "acc_z" in col
            or "accz" in col
            or col == "z"
        ):
            z_candidates.append(
                original[i]
            )

    if (
        len(x_candidates) > 0
        and len(y_candidates) > 0
        and len(z_candidates) > 0
    ):

        return [
            x_candidates[0],
            y_candidates[0],
            z_candidates[0]
        ]

    # --------------------------------------------------------
    # Generic X/Y/Z fallback
    # --------------------------------------------------------

    exact_xyz = {}

    for i, col in enumerate(columns):

        if col in ["x", "x-axis", "x axis"]:
            exact_xyz["x"] = original[i]

        if col in ["y", "y-axis", "y axis"]:
            exact_xyz["y"] = original[i]

        if col in ["z", "z-axis", "z axis"]:
            exact_xyz["z"] = original[i]

    if len(exact_xyz) == 3:

        return [
            exact_xyz["x"],
            exact_xyz["y"],
            exact_xyz["z"]
        ]

    # --------------------------------------------------------
    # Numeric fallback
    # --------------------------------------------------------

    numeric_cols = find_numeric_columns(
        df
    )

    if len(numeric_cols) >= 3:

        return numeric_cols[:3]

    return None


# ------------------------------------------------------------
# Normalize one window
# ------------------------------------------------------------

def normalize_window(window):

    window = safe_float_array(
        window
    )

    # Per-channel normalization
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

    std[
        std < 1e-6
    ] = 1.0

    window = (
        window - mean
    ) / std

    return window.astype(
        np.float32
    )


# ------------------------------------------------------------
# Create windows
# ------------------------------------------------------------

def make_windows(
    signal,
    window_size=WINDOW_SIZE,
    max_windows=None
):

    signal = safe_float_array(
        signal
    )

    if signal.ndim != 2:
        return np.empty(
            (0, window_size, CHANNELS),
            dtype=np.float32
        )

    # Make sure exactly 3 channels exist
    if signal.shape[1] < 3:

        return np.empty(
            (0, window_size, CHANNELS),
            dtype=np.float32
        )

    signal = signal[:, :3]

    if len(signal) < window_size:
        return np.empty(
            (0, window_size, CHANNELS),
            dtype=np.float32
        )

    windows = []

    step = window_size // 2

    for start in range(
        0,
        len(signal) - window_size + 1,
        step
    ):

        window = signal[
            start:start + window_size
        ]

        window = normalize_window(
            window
        )

        windows.append(
            window
        )

        if (
            max_windows is not None
            and len(windows) >= max_windows
        ):
            break

    if len(windows) == 0:

        return np.empty(
            (0, window_size, CHANNELS),
            dtype=np.float32
        )

    return np.asarray(
        windows,
        dtype=np.float32
    )


# ============================================================
# 6. LOAD EXISTING NPZ DATASET
# ============================================================

def load_npz_windows(
    path,
    source_name
):

    if not os.path.exists(path):

        print(
            f"[NOT FOUND] {source_name}: {path}"
        )

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    print()
    print(
        f"[LOADING] {source_name}"
    )

    try:

        data = np.load(
            path,
            allow_pickle=True
        )

        print(
            "Keys:",
            list(data.keys())
        )

        X = None

        # ----------------------------------------------------
        # Search common keys
        # ----------------------------------------------------

        for key in [
            "X",
            "x",
            "windows",
            "gait_windows",
            "data"
        ]:

            if key in data:

                X = data[key]

                break

        if X is None:

            print(
                "[WARNING] No gait tensor found."
            )

            return np.empty(
                (0, WINDOW_SIZE, CHANNELS),
                dtype=np.float32
            )

        X = safe_float_array(
            X
        )

        print(
            "Original shape:",
            X.shape
        )

        # ----------------------------------------------------
        # Convert common shapes
        # ----------------------------------------------------

        if X.ndim == 2:

            X = make_windows(
                X
            )

        elif X.ndim == 3:

            # [N,256,3]
            if (
                X.shape[1] == WINDOW_SIZE
                and X.shape[2] >= 3
            ):

                X = X[:, :, :3]

            # [N,3,256]
            elif (
                X.shape[1] >= 3
                and X.shape[2] == WINDOW_SIZE
            ):

                X = np.transpose(
                    X[:, :3, :],
                    (0, 2, 1)
                )

            else:

                print(
                    "[WARNING] Unsupported 3D shape:",
                    X.shape
                )

                return np.empty(
                    (0, WINDOW_SIZE, CHANNELS),
                    dtype=np.float32
                )

        else:

            print(
                "[WARNING] Unsupported dimensions:",
                X.ndim
            )

            return np.empty(
                (0, WINDOW_SIZE, CHANNELS),
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        X = X[
            :,
            :WINDOW_SIZE,
            :CHANNELS
        ]

        if X.shape[1] != WINDOW_SIZE:
            print(
                "[WARNING] Wrong window length."
            )

            return np.empty(
                (0, WINDOW_SIZE, CHANNELS),
                dtype=np.float32
            )

        print(
            "Final shape:",
            X.shape
        )

        print(
            "Windows:",
            len(X)
        )

        return X.astype(
            np.float32
        )

    except Exception as e:

        print(
            "[ERROR]",
            source_name,
            ":",
            e
        )

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )


# ============================================================
# 7. FIND EXTERNAL DATASETS
# ============================================================

def find_external_datasets():

    found = {}

    print()
    print("=" * 78)
    print("SEARCHING FOR EXTERNAL DATASETS")
    print("=" * 78)

    for dataset_name, patterns in (
        EXTERNAL_DATASET_PATTERNS.items()
    ):

        found[dataset_name] = []

        for search_dir in EXTERNAL_SEARCH_DIRS:

            if not os.path.exists(
                search_dir
            ):
                continue

            for root, dirs, files in os.walk(
                search_dir
            ):

                # Avoid scanning generated output
                if (
                    "integrated"
                    in root.lower()
                ):
                    continue

                for file in files:

                    lower = file.lower()

                    for pattern in patterns:

                        if pattern.lower() in lower:

                            full_path = os.path.join(
                                root,
                                file
                            )

                            if full_path not in found[
                                dataset_name
                            ]:

                                found[
                                    dataset_name
                                ].append(
                                    full_path
                                )

    for name, files in found.items():

        print()
        print(name)

        if len(files) == 0:

            print(
                "  Not found"
            )

        else:

            for f in files:

                print(
                    "  ",
                    f
                )

    return found


def process_csv_file(
    csv_path,
    source_name
):

    try:

        df = pd.read_csv(
            csv_path,
            low_memory=False
        )

    except Exception:

        try:

            df = pd.read_csv(
                csv_path,
                header=None,
                low_memory=False
            )

        except Exception as e:

            print(
                "[CSV ERROR]",
                csv_path,
                e
            )

            return np.empty(
                (0, WINDOW_SIZE, CHANNELS),
                dtype=np.float32
            )

    if len(df) < WINDOW_SIZE:

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    columns = detect_accelerometer_columns(
        df
    )

    if columns is None:

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    try:

        signal = df[
            columns
        ].apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna().values

    except Exception:

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    if len(signal) < WINDOW_SIZE:

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    windows = make_windows(
        signal,
        max_windows=500
    )

    return windows


# ============================================================
# 9. PROCESS ZIP DATASET
# ============================================================

def process_zip_dataset(
    zip_path,
    source_name
):

    print()
    print(
        "=" * 78
    )

    print(
        "PROCESSING:",
        source_name
    )

    print(
        zip_path
    )

    print(
        "=" * 78
    )

    if not os.path.exists(
        zip_path
    ):

        print(
            "[NOT FOUND]"
        )

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    all_windows = []

    temp_dir = tempfile.mkdtemp(
        prefix="cmt_integrate_"
    )

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            members = z.namelist()

            print(
                "Archive entries:",
                len(members)
            )

            # ------------------------------------------------
            # Process CSV files
            # ------------------------------------------------

            csv_members = [
                m for m in members
                if m.lower().endswith(".csv")
            ]

            print(
                "CSV files:",
                len(csv_members)
            )

            # Avoid extracting gigantic irrelevant files
            for index, member in enumerate(
                csv_members
            ):

                try:

                    target = os.path.join(
                        temp_dir,
                        os.path.basename(member)
                    )

                    with z.open(
                        member
                    ) as source:

                        with open(
                            target,
                            "wb"
                        ) as dest:

                            shutil.copyfileobj(
                                source,
                                dest
                            )

                    windows = process_csv_file(
                        target,
                        source_name
                    )

                    if len(windows) > 0:

                        all_windows.append(
                            windows
                        )

                        print(
                            f"  {index + 1}/{len(csv_members)}",
                            os.path.basename(member),
                            "->",
                            len(windows),
                            "windows"
                        )

                except Exception as e:

                    print(
                        "[SKIP]",
                        member,
                        e
                    )

                current_count = sum(
                    len(x)
                    for x in all_windows
                )

                if (
                    current_count
                    >= MAX_WINDOWS_PER_EXTERNAL_SOURCE
                ):

                    print(
                        "Maximum window limit reached."
                    )

                    break

    except Exception as e:

        print(
            "[ZIP ERROR]",
            e
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

    if len(all_windows) == 0:

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    X = np.concatenate(
        all_windows,
        axis=0
    )

    if len(X) > MAX_WINDOWS_PER_EXTERNAL_SOURCE:

        indices = rng.choice(
            len(X),
            size=MAX_WINDOWS_PER_EXTERNAL_SOURCE,
            replace=False
        )

        X = X[
            indices
        ]

    print(
        "Final external windows:",
        X.shape
    )

    return X.astype(
        np.float32
    )


# ============================================================
# 10. LOAD CMT1A PATIENT GAIT WINDOWS
# ============================================================

def load_cmt_patient_windows(
    patient_id
):

    file_path = os.path.join(
        GAIT_DIR,
        f"{patient_id}_gait.npz"
    )

    if not os.path.exists(
        file_path
    ):

        return np.empty(
            (0, WINDOW_SIZE, CHANNELS),
            dtype=np.float32
        )

    return load_npz_windows(
        file_path,
        f"CMT1A-{patient_id}"
    )


# ============================================================
# 11. LOAD CMT CLINICAL DATA
# ============================================================

def load_clinical_data():

    print()
    print("=" * 78)
    print("LOADING CMT1A CLINICAL DATA")
    print("=" * 78)

    if not os.path.exists(
        CLINICAL_FILE
    ):

        raise FileNotFoundError(
            f"Clinical file not found:\n{CLINICAL_FILE}"
        )

    clinical = pd.read_csv(
        CLINICAL_FILE
    )

    print(
        "Clinical shape:",
        clinical.shape
    )

    print(
        "Columns:"
    )

    print(
        clinical.columns.tolist()
    )

    # --------------------------------------------------------
    # Only CMT1A
    # --------------------------------------------------------

    clinical = clinical[
        clinical["Status"]
        .astype(str)
        .str.upper()
        .str.contains("CMT1A")
    ].copy()

    clinical[
        "ID"
    ] = clinical[
        "ID"
    ].astype(str).str.strip()

    print()
    print(
        "CMT1A records:",
        len(clinical)
    )

    print()
    print(
        "CMT1A patients:",
        clinical["ID"].nunique()
    )

    return clinical


# ============================================================
# 12. CREATE SEVERITY LABEL
# ============================================================

def assign_severity(
    cmtes_score
):

    if pd.isna(
        cmtes_score
    ):

        return np.nan

    score = float(
        cmtes_score
    )

    # --------------------------------------------------------
    # Current project severity mapping
    #
    # IMPORTANT:
    # If your original validated threshold definition differs,
    # replace ONLY this function.
    #
    # --------------------------------------------------------

    if score <= 10:

        return 0

    elif score <= 20:

        return 1

    else:

        return 2


def severity_name(
    label
):

    mapping = {
        0: "Mild",
        1: "Moderate",
        2: "Severe"
    }

    return mapping.get(
        label,
        "Unknown"
    )


# ============================================================
# 13. BUILD 15-PATIENT CMT MANIFEST
# ============================================================

def build_cmt_manifest(
    clinical
):

    print()
    print("=" * 78)
    print("BUILDING 15-PATIENT CMT1A MANIFEST")
    print("=" * 78)

    # --------------------------------------------------------
    # Select baseline CMT1A
    # --------------------------------------------------------

    baseline = clinical[
        clinical["Time"]
        .astype(str)
        .str.lower()
        .str.contains("baseline")
    ].copy()

    # --------------------------------------------------------
    # If Time does not contain baseline,
    # use first record per patient.
    # --------------------------------------------------------

    if len(baseline) == 0:

        baseline = (
            clinical
            .sort_values(
                ["ID"]
            )
            .groupby(
                "ID",
                as_index=False
            )
            .first()
        )

    # --------------------------------------------------------
    # Keep one row per CMT patient
    # --------------------------------------------------------

    baseline = (
        baseline
        .drop_duplicates(
            subset=["ID"]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Make sure we don't silently lose patients
    # --------------------------------------------------------

    print(
        "Baseline CMT1A patients:",
        len(baseline)
    )

    # --------------------------------------------------------
    # Required clinical features
    # --------------------------------------------------------

    feature_map = {}

    def get_column(
        preferred,
        alternatives
    ):

        if preferred in baseline.columns:
            return preferred

        for alt in alternatives:

            if alt in baseline.columns:
                return alt

        return None

    age_col = get_column(
        "Age (Years)",
        ["Age"]
    )

    gender_col = get_column(
        "Gender",
        ["Sex"]
    )

    pinprick_col = get_column(
        "Pinprick",
        []
    )

    vibration_col = get_column(
        "Vibration",
        []
    )

    cmtes_col = get_column(
        "Cmtes_score",
        ["CMTES", "CMTES_score"]
    )

    cmt_fom_col = get_column(
        "CMT-FOM_Total",
        []
    )

    mmt_col = get_column(
        "MMT_average",
        []
    )

    # --------------------------------------------------------
    # Construct manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame()

    manifest[
        "ID"
    ] = baseline[
        "ID"
    ].astype(str)

    manifest[
        "Status"
    ] = "CMT1A"

    if age_col is not None:

        manifest[
            "Age (Years)"
        ] = pd.to_numeric(
            baseline[age_col],
            errors="coerce"
        )

    else:

        manifest[
            "Age (Years)"
        ] = np.nan

    if gender_col is not None:

        manifest[
            "Gender"
        ] = baseline[
            gender_col
        ].astype(str)

    else:

        manifest[
            "Gender"
        ] = "Unknown"

    if pinprick_col is not None:

        manifest[
            "Pinprick"
        ] = pd.to_numeric(
            baseline[pinprick_col],
            errors="coerce"
        )

    else:

        manifest[
            "Pinprick"
        ] = np.nan

    if vibration_col is not None:

        manifest[
            "Vibration"
        ] = pd.to_numeric(
            baseline[vibration_col],
            errors="coerce"
        )

    else:

        manifest[
            "Vibration"
        ] = np.nan

    if cmtes_col is not None:

        manifest[
            "Cmtes_score"
        ] = pd.to_numeric(
            baseline[cmtes_col],
            errors="coerce"
        )

    else:

        manifest[
            "Cmtes_score"
        ] = np.nan

    if cmt_fom_col is not None:

        manifest[
            "CMT-FOM_Total"
        ] = pd.to_numeric(
            baseline[cmt_fom_col],
            errors="coerce"
        )

    else:

        manifest[
            "CMT-FOM_Total"
        ] = np.nan

    if mmt_col is not None:

        manifest[
            "MMT_average"
        ] = pd.to_numeric(
            baseline[mmt_col],
            errors="coerce"
        )

    else:

        manifest[
            "MMT_average"
        ] = np.nan

    # --------------------------------------------------------
    # Gender encoding
    # --------------------------------------------------------

    manifest[
        "Gender_encoded"
    ] = (
        manifest["Gender"]
        .astype(str)
        .str.lower()
        .map(
            {
                "male": 1,
                "m": 1,
                "female": 0,
                "f": 0
            }
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    manifest[
        "severity_label"
    ] = manifest[
        "Cmtes_score"
    ].apply(
        assign_severity
    )

    manifest[
        "severity_name"
    ] = manifest[
        "severity_label"
    ].apply(
        severity_name
    )

    # --------------------------------------------------------
    # Gait files
    # --------------------------------------------------------

    gait_files = []

    gait_windows = []

    for patient_id in manifest[
        "ID"
    ]:

        path = os.path.join(
            GAIT_DIR,
            f"{patient_id}_gait.npz"
        )

        gait_files.append(
            path
        )

        if os.path.exists(
            path
        ):

            try:

                d = np.load(
                    path,
                    allow_pickle=True
                )

                X = d["X"]

                gait_windows.append(
                    len(X)
                )

            except Exception:

                gait_windows.append(
                    0
                )

        else:

            gait_windows.append(
                0
            )

    manifest[
        "gait_file"
    ] = gait_files

    manifest[
        "gait_windows"
    ] = gait_windows

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    manifest = manifest.sort_values(
        "ID"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Patient-level split
    #
    # IMPORTANT:
    # all longitudinal records for a patient must use the
    # same split.
    # --------------------------------------------------------

    patient_ids = manifest[
        "ID"
    ].tolist()

    shuffled = patient_ids.copy()

    local_rng = np.random.default_rng(
        RANDOM_SEED
    )

    local_rng.shuffle(
        shuffled
    )

    n = len(
        shuffled
    )

    n_train = int(
        round(
            n * 0.60
        )
    )

    n_val = int(
        round(
            n * 0.20
        )
    )

    train_ids = set(
        shuffled[:n_train]
    )

    val_ids = set(
        shuffled[
            n_train:n_train + n_val
        ]
    )

    test_ids = set(
        shuffled[
            n_train + n_val:
        ]
    )

    def assign_split(
        patient_id
    ):

        if patient_id in train_ids:
            return "train"

        if patient_id in val_ids:
            return "val"

        return "test"

    manifest[
        "split"
    ] = manifest[
        "ID"
    ].apply(
        assign_split
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        SPLIT_DIR,
        "stage5c_integrated_patient_manifest.csv"
    )

    manifest.to_csv(
        output_path,
        index=False
    )

    # Also replace the manifest used by Stage 6
    current_manifest = os.path.join(
        SPLIT_DIR,
        "stage5c_patient_manifest.csv"
    )

    manifest.to_csv(
        current_manifest,
        index=False
    )

    print()
    print(
        "Manifest saved:"
    )

    print(
        output_path
    )

    print()
    print(
        "Stage-6 manifest updated:"
    )

    print(
        current_manifest
    )

    print()
    print(
        "Patients:",
        len(manifest)
    )

    print()
    print(
        "Split distribution:"
    )

    print(
        manifest["split"].value_counts()
    )

    print()
    print(
        "Severity distribution:"
    )

    print(
        manifest["severity_name"].value_counts()
    )

    return manifest


# ============================================================
# 14. LOAD LONGITUDINAL CMT1A INFORMATION
# ============================================================

def build_longitudinal_table(
    clinical
):

    print()
    print("=" * 78)
    print("BUILDING LONGITUDINAL CMT1A TABLE")
    print("=" * 78)

    long_df = clinical[
        clinical["Time"]
        .astype(str)
        .str.lower()
        .str.contains(
            "longitudinal"
        )
    ].copy()

    if len(long_df) == 0:

        print(
            "No longitudinal records found."
        )

        return pd.DataFrame()

    long_df[
        "ID"
    ] = long_df[
        "ID"
    ].astype(str)

    output = os.path.join(
        INTEGRATED_DIR,
        "cmt1a_longitudinal_clinical.csv"
    )

    long_df.to_csv(
        output,
        index=False
    )

    print(
        "Longitudinal records:",
        len(long_df)
    )

    print(
        "Longitudinal patients:",
        long_df["ID"].nunique()
    )

    print(
        "Saved:",
        output
    )

    return long_df


# ============================================================
# 15. COLLECT CMT1A GAIT WINDOWS
# ============================================================

def collect_cmt_gait(
    manifest
):

    print()
    print("=" * 78)
    print("COLLECTING 15 CMT1A GAIT PATIENTS")
    print("=" * 78)

    arrays = []

    patient_ids = []

    labels = []

    source = []

    total = 0

    for _, row in manifest.iterrows():

        patient_id = str(
            row["ID"]
        )

        X = load_cmt_patient_windows(
            patient_id
        )

        if len(X) == 0:

            print(
                patient_id,
                "-> no valid windows"
            )

            continue

        arrays.append(
            X
        )

        patient_ids.extend(
            [patient_id] * len(X)
        )

        labels.extend(
            [int(row["severity_label"])] * len(X)
        )

        source.extend(
            ["CMT1A"] * len(X)
        )

        total += len(X)

        print(
            patient_id,
            "->",
            len(X),
            "windows"
        )

    if len(arrays) == 0:

        return (
            np.empty(
                (0, WINDOW_SIZE, CHANNELS),
                dtype=np.float32
            ),
            [],
            [],
            []
        )

    X = np.concatenate(
        arrays,
        axis=0
    )

    print()
    print(
        "Total CMT1A windows:",
        len(X)
    )

    return (
        X,
        patient_ids,
        labels,
        source
    )


# ============================================================
# 16. LOAD EXISTING SUPPLEMENTARY DATASETS
# ============================================================

def collect_existing_supplementary():

    print()
    print("=" * 78)
    print("COLLECTING EXISTING SUPPLEMENTARY GAIT DATA")
    print("=" * 78)

    datasets = []

    # --------------------------------------------------------
    # Multi-sensor gait
    # --------------------------------------------------------

    X_multi = load_npz_windows(
        MULTI_GAIT_FILE,
        "Multi-Sensor Gait"
    )

    if len(X_multi) > 0:

        datasets.append(
            (
                "Multi-Sensor",
                X_multi
            )
        )

    # --------------------------------------------------------
    # Secondary
    # --------------------------------------------------------

    X_secondary = load_npz_windows(
        SECONDARY_GAIT_FILE,
        "Secondary Gait"
    )

    if len(X_secondary) > 0:

        datasets.append(
            (
                "Secondary",
                X_secondary
            )
        )

    # --------------------------------------------------------
    # Combined
    #
    # IMPORTANT:
    # Do NOT add combined if it already contains the
    # multi + secondary datasets.
    #
    # Otherwise we would duplicate data.
    # --------------------------------------------------------

    X_combined = load_npz_windows(
        COMBINED_GAIT_FILE,
        "Combined Gait"
    )

    if len(X_combined) > 0:

        datasets.append(
            (
                "Combined",
                X_combined
            )
        )

    return datasets


# ============================================================
# 17. PROCESS RAW EXTERNAL DATASETS
# ============================================================

def process_external_sources(
    found
):

    datasets = []

    for dataset_name, files in found.items():

        for path in files:

            # ------------------------------------------------
            # Skip the CMT project ZIPs
            # ------------------------------------------------

            lower = path.lower()

            if (
                "cmt1a" in lower
                or "biostamp" in lower
            ):

                continue

            # ------------------------------------------------
            # ZIP
            # ------------------------------------------------

            if lower.endswith(
                ".zip"
            ):

                X = process_zip_dataset(
                    path,
                    dataset_name
                )

                if len(X) > 0:

                    datasets.append(
                        (
                            dataset_name,
                            X
                        )
                    )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            elif lower.endswith(
                ".csv"
            ):

                X = process_csv_file(
                    path,
                    dataset_name
                )

                if len(X) > 0:

                    datasets.append(
                        (
                            dataset_name,
                            X
                        )
                    )

    return datasets


# ============================================================
# 18. DEDUPLICATE DATASET SOURCES
# ============================================================

def remove_duplicate_sources(
    datasets
):

    """
    If Combined already contains Multi-Sensor and Secondary,
    don't add all three again.
    """

    names = [
        name
        for name, _ in datasets
    ]

    cleaned = []

    has_combined = (
        "Combined" in names
    )

    for name, X in datasets:

        if (
            has_combined
            and name in [
                "Multi-Sensor",
                "Secondary"
            ]
        ):

            print(
                f"Skipping {name} because Combined dataset exists."
            )

            continue

        cleaned.append(
            (
                name,
                X
            )
        )

    return cleaned


# ============================================================
# 19. CREATE INTEGRATED PRETRAINING CORPUS
# ============================================================

def create_integrated_corpus(
    cmt_X,
    cmt_patients,
    cmt_labels,
    cmt_sources,
    supplementary
):

    print()
    print("=" * 78)
    print("CREATING INTEGRATED GAIT CORPUS")
    print("=" * 78)

    # --------------------------------------------------------
    # Important:
    #
    # CMT windows + supplementary windows are stored together.
    #
    # Labels:
    #
    # CMT -> true severity label
    # External -> -1 (unlabelled)
    #
    # This lets Stage 5A use all gait data for representation
    # learning without falsely labelling external patients.
    # --------------------------------------------------------

    all_X = []
    all_labels = []
    all_patient_ids = []
    all_sources = []

    # --------------------------------------------------------
    # CMT
    # --------------------------------------------------------

    if len(cmt_X) > 0:

        all_X.append(
            cmt_X
        )

        all_labels.extend(
            cmt_labels
        )

        all_patient_ids.extend(
            cmt_patients
        )

        all_sources.extend(
            cmt_sources
        )

    # --------------------------------------------------------
    # Supplementary
    # --------------------------------------------------------

    for source_name, X in supplementary:

        if len(X) == 0:
            continue

        all_X.append(
            X
        )

        # External datasets are unlabeled
        all_labels.extend(
            [-1] * len(X)
        )

        # Anonymous external subject/window ID
        all_patient_ids.extend(
            [
                f"{source_name}_sample_{i}"
                for i in range(len(X))
            ]
        )

        all_sources.extend(
            [source_name] * len(X)
        )

    if len(all_X) == 0:

        raise RuntimeError(
            "No gait datasets were successfully integrated."
        )

    X = np.concatenate(
        all_X,
        axis=0
    ).astype(
        np.float32
    )

    labels = np.asarray(
        all_labels,
        dtype=np.int64
    )

    patient_ids = np.asarray(
        all_patient_ids,
        dtype=object
    )

    sources = np.asarray(
        all_sources,
        dtype=object
    )

    print()
    print(
        "Integrated tensor:",
        X.shape
    )

    print(
        "Total windows:",
        len(X)
    )

    print()
    print(
        "Source distribution:"
    )

    print(
        pd.Series(
            sources
        ).value_counts()
    )

    # --------------------------------------------------------
    # Save master corpus
    # --------------------------------------------------------

    output_npz = os.path.join(
        INTEGRATED_DIR,
        "integrated_all_gait_windows.npz"
    )

    np.savez_compressed(
        output_npz,
        X=X,
        severity_label=labels,
        patient_id=patient_ids,
        source=sources
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_npz
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "severity_label": labels,
            "source": sources
        }
    )

    metadata_path = os.path.join(
        INTEGRATED_DIR,
        "integrated_gait_metadata.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    print(
        "Metadata saved:"
    )

    print(
        metadata_path
    )

    return X, metadata


# ============================================================
# 20. CREATE DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    manifest,
    supplementary
):

    rows = []

    # --------------------------------------------------------
    # CMT1A
    # --------------------------------------------------------

    cmt_windows = int(
        manifest[
            "gait_windows"
        ].sum()
    )

    rows.append(
        {
            "Dataset": "CMT1A BioStampNPoint2023",
            "Patients": int(
                manifest["ID"].nunique()
            ),
            "Gait Windows": cmt_windows,
            "Role": "Supervised CMT severity"
        }
    )

    # --------------------------------------------------------
    # Supplementary
    # --------------------------------------------------------

    for name, X in supplementary:

        rows.append(
            {
                "Dataset": name,
                "Patients": "External / unknown",
                "Gait Windows": len(X),
                "Role": "Gait representation pretraining"
            }
        )

    summary = pd.DataFrame(
        rows
    )

    output = os.path.join(
        INTEGRATED_DIR,
        "integrated_dataset_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        output
    )

    return summary


# ============================================================
# 21. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    clinical = load_clinical_data()

    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    manifest = build_cmt_manifest(
        clinical
    )

    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    longitudinal = build_longitudinal_table(
        clinical
    )

    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    (
        cmt_X,
        cmt_patients,
        cmt_labels,
        cmt_sources
    ) = collect_cmt_gait(
        manifest
    )

    # --------------------------------------------------------
    # Step 5
    # Existing supplementary datasets
    # --------------------------------------------------------

    supplementary = (
        collect_existing_supplementary()
    )

    # --------------------------------------------------------
    # Step 6
    # Search raw external datasets
    # --------------------------------------------------------

    found = find_external_datasets()

    # --------------------------------------------------------
    # Step 7
    # Process raw external datasets
    # --------------------------------------------------------

    external = process_external_sources(
        found
    )

    # --------------------------------------------------------
    # Step 8
    # Add external datasets
    # --------------------------------------------------------

    supplementary.extend(
        external
    )

    # --------------------------------------------------------
    # Step 9
    # Prevent accidental duplication
    # --------------------------------------------------------

    supplementary = (
        remove_duplicate_sources(
            supplementary
        )
    )

    # --------------------------------------------------------
    # Print supplementary summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("SUPPLEMENTARY DATASETS READY")
    print("=" * 78)

    for name, X in supplementary:

        print(
            f"{name:<35} {len(X):>10} windows"
        )

    # --------------------------------------------------------
    # Step 10
    # Create integrated corpus
    # --------------------------------------------------------

    X, metadata = create_integrated_corpus(
        cmt_X,
        cmt_patients,
        cmt_labels,
        cmt_sources,
        supplementary
    )

    # --------------------------------------------------------
    # Step 11
    # Summary
    # --------------------------------------------------------

    create_dataset_summary(
        manifest,
        supplementary
    )

    # --------------------------------------------------------
    # Step 12
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INTEGRATION COMPLETED")
    print("=" * 78)

    print()
    print(
        "CMT1A patients:",
        manifest["ID"].nunique()
    )

    print(
        "CMT1A gait windows:",
        len(cmt_X)
    )

    print(
        "Integrated gait windows:",
        len(X)
    )

    print()

    print(
        "Integrated tensor:"
    )

    print(
        os.path.join(
            INTEGRATED_DIR,
            "integrated_all_gait_windows.npz"
        )
    )

    print()

    print(
        "Metadata:"
    )

    print(
        os.path.join(
            INTEGRATED_DIR,
            "integrated_gait_metadata.csv"
        )
    )

    print()

    print(
        "15-patient manifest:"
    )

    print(
        os.path.join(
            SPLIT_DIR,
            "stage5c_patient_manifest.csv"
        )
    )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()