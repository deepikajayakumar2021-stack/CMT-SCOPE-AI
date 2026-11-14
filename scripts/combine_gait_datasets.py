# ============================================================
# COMBINE GAIT DATASETS
#
# Sources:
#   1. Multi-Sensor Human Gait Dataset
#   2. Secondary / Triadic Gait Dataset
#
# Output:
#   combined_gait_windows.npz
#
# Shape:
#   [N, 256, 3]
#
# IMPORTANT:
#   No severity labels are created for supplementary datasets.
# ============================================================

import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\CMT_PROJECT"

MULTI_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "multi_gait_windows.npz"
)

SECONDARY_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "secondary_gait_windows.npz"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "combined_gait_windows.npz"
)


# ============================================================
# LOAD MULTI-SENSOR DATA
# ============================================================

print("=" * 70)
print("COMBINING GAIT DATASETS")
print("=" * 70)

print()
print("Loading Multi-Sensor gait dataset...")

multi = np.load(
    MULTI_FILE,
    allow_pickle=True
)

multi_X = multi["gait"]
multi_users = multi["user_id"]
multi_recordings = multi["recording_id"]

print("Multi-Sensor windows:", multi_X.shape)
print("Multi-Sensor users:", len(np.unique(multi_users)))
print(
    "Multi-Sensor recordings:",
    len(np.unique(multi_recordings))
)


# ============================================================
# LOAD SECONDARY DATASET
# ============================================================

print()
print("Loading secondary gait dataset...")

secondary = np.load(
    SECONDARY_FILE,
    allow_pickle=True
)

secondary_X = secondary["X"]
secondary_sources = secondary["source_files"]

print(
    "Secondary windows:",
    secondary_X.shape
)

print(
    "Secondary source files:",
    len(np.unique(secondary_sources))
)


# ============================================================
# VALIDATE SHAPES
# ============================================================

if multi_X.ndim != 3:
    raise ValueError(
        "Multi-Sensor gait data must be 3-dimensional."
    )

if secondary_X.ndim != 3:
    raise ValueError(
        "Secondary gait data must be 3-dimensional."
    )

if multi_X.shape[1:] != (256, 3):
    raise ValueError(
        f"Unexpected Multi-Sensor shape: {multi_X.shape}"
    )

if secondary_X.shape[1:] != (256, 3):
    raise ValueError(
        f"Unexpected secondary shape: {secondary_X.shape}"
    )


# ============================================================
# COMBINE
# ============================================================

combined_X = np.concatenate(
    [
        multi_X.astype(np.float32),
        secondary_X.astype(np.float32)
    ],
    axis=0
)


# ============================================================
# SOURCE METADATA
# ============================================================

multi_source = np.array(
    ["multi_sensor"] * len(multi_X),
    dtype=object
)

secondary_source = np.array(
    ["secondary"] * len(secondary_X),
    dtype=object
)

source_dataset = np.concatenate(
    [
        multi_source,
        secondary_source
    ]
)


# ============================================================
# RECORDING / SAMPLE IDENTIFIERS
# ============================================================

multi_ids = np.array(
    [
        f"multi_{x}"
        for x in multi_recordings
    ],
    dtype=object
)

secondary_ids = np.array(
    [
        f"secondary_{x}"
        for x in secondary_sources
    ],
    dtype=object
)

source_id = np.concatenate(
    [
        multi_ids,
        secondary_ids
    ]
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X=combined_X,
    source_dataset=source_dataset,
    source_id=source_id
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMBINATION COMPLETE")
print("=" * 70)

print()
print("Multi-Sensor windows :", len(multi_X))
print("Secondary windows    :", len(secondary_X))
print("Combined windows     :", len(combined_X))

print()
print("Combined shape:")
print(combined_X.shape)

print()
print("Dataset distribution:")

unique, counts = np.unique(
    source_dataset,
    return_counts=True
)

for name, count in zip(unique, counts):
    print(f"{name}: {count}")

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("IMPORTANT:")
print(
    "Supplementary gait data has no CMT severity labels."
)

print("=" * 70)