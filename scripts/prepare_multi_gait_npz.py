import os
import glob
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_DIR = r"D:\CMT_PROJECT\A multi-sensor human gait dataset\processed_data"

OUTPUT_DIR = r"D:\CMT_PROJECT\processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "multi_gait_windows.npz"
)

MANIFEST_FILE = os.path.join(
    OUTPUT_DIR,
    "multi_gait_manifest.csv"
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

WINDOW_SIZE = 256
STEP_SIZE = 128

# Keep this as metadata only.
# Change it if your dataset documentation specifies
# another sampling frequency.
SAMPLING_FREQUENCY = None


# ------------------------------------------------------------
# FIND IMU WALK FILES
# ------------------------------------------------------------

files = glob.glob(
    os.path.join(
        INPUT_DIR,
        "*",
        "*_imu_walk.csv"
    )
)

files = sorted(files)


print("=" * 70)
print("MULTI-SENSOR GAIT PREPARATION")
print("=" * 70)

print("Input directory :", INPUT_DIR)
print("Recordings found:", len(files))


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

all_windows = []

window_user = []
window_recording = []

manifest_rows = []


# ------------------------------------------------------------
# PROCESS EACH RECORDING
# ------------------------------------------------------------

for file_path in files:

    try:

        df = pd.read_csv(file_path)

        required_columns = [
            "acc x",
            "acc y",
            "acc z"
        ]

        # ----------------------------------------------------
        # CHECK COLUMNS
        # ----------------------------------------------------

        if not all(
            col in df.columns
            for col in required_columns
        ):

            print(
                "SKIPPED - missing acceleration columns:",
                file_path
            )

            continue


        # ----------------------------------------------------
        # EXTRACT ACCELERATION
        # ----------------------------------------------------

        acc = df[
            ["acc x", "acc y", "acc z"]
        ].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(
            dtype=np.float32
        )


        # ----------------------------------------------------
        # REMOVE NaN / INF
        # ----------------------------------------------------

        valid = np.isfinite(acc).all(axis=1)

        acc = acc[valid]


        # ----------------------------------------------------
        # CHECK LENGTH
        # ----------------------------------------------------

        if len(acc) < WINDOW_SIZE:

            print(
                "SKIPPED - recording shorter than window:",
                file_path
            )

            continue


        # ----------------------------------------------------
        # NORMALIZE RECORDING
        # ----------------------------------------------------

        mean = np.mean(
            acc,
            axis=0
        )

        std = np.std(
            acc,
            axis=0
        )

        std[std < 1e-6] = 1.0

        acc = (
            acc - mean
        ) / std


        # ----------------------------------------------------
        # IDENTIFY RECORDING
        # ----------------------------------------------------

        recording_name = os.path.basename(
            file_path
        )

        user_name = os.path.basename(
            os.path.dirname(file_path)
        )


        # ----------------------------------------------------
        # CREATE WINDOWS
        # ----------------------------------------------------

        recording_window_count = 0

        for start in range(
            0,
            len(acc) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            window = acc[
                start:start + WINDOW_SIZE
            ]


            if window.shape != (
                WINDOW_SIZE,
                3
            ):

                continue


            all_windows.append(
                window
            )

            window_user.append(
                user_name
            )

            window_recording.append(
                recording_name
            )

            recording_window_count += 1


        # ----------------------------------------------------
        # MANIFEST ENTRY
        # ----------------------------------------------------

        manifest_rows.append({

            "user_id": user_name,

            "recording_id": recording_name,

            "samples_after_cleaning": len(acc),

            "windows": recording_window_count,

            "window_size": WINDOW_SIZE,

            "step_size": STEP_SIZE,

            "channels": "acc x, acc y, acc z",

            "severity_label": "unknown",

            "disease_label": "unknown"

        })


        print(
            f"OK: {recording_name} -> "
            f"{recording_window_count} windows"
        )


    except Exception as e:

        print(
            "ERROR:",
            file_path
        )

        print(
            "      ",
            e
        )


# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------

if len(all_windows) == 0:

    raise RuntimeError(
        "No valid gait windows were generated."
    )


# ------------------------------------------------------------
# CONVERT TO NUMPY
# ------------------------------------------------------------

X = np.stack(
    all_windows
).astype(
    np.float32
)


window_user = np.array(
    window_user
)

window_recording = np.array(
    window_recording
)


# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SAVE NPZ
# ------------------------------------------------------------

np.savez_compressed(

    OUTPUT_FILE,

    gait=X,

    user_id=window_user,

    recording_id=window_recording,

    window_size=WINDOW_SIZE,

    step_size=STEP_SIZE,

    sampling_frequency=SAMPLING_FREQUENCY,

    channels=np.array([
        "acc_x",
        "acc_y",
        "acc_z"
    ])

)


# ------------------------------------------------------------
# SAVE MANIFEST
# ------------------------------------------------------------

manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

unique_users = sorted(
    set(
        window_user.tolist()
    )
)

unique_recordings = sorted(
    set(
        window_recording.tolist()
    )
)


print()
print("=" * 70)
print("COMPLETED")
print("=" * 70)

print(
    "Valid recordings :",
    len(unique_recordings)
)

print(
    "Users             :",
    len(unique_users)
)

print(
    "Total windows     :",
    len(X)
)

print(
    "Window shape      :",
    X.shape
)

print()
print("NPZ saved:")
print(OUTPUT_FILE)

print()
print("Manifest saved:")
print(MANIFEST_FILE)

print()
print("CMT severity labels:")
print("NOT ASSIGNED")

print()
print("Users:")
print(
    ", ".join(unique_users)
)

print("=" * 70)