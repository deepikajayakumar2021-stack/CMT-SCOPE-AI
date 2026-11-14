import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)
INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)
if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nCurrent test results were not found.\n\n"
        f"Expected:\n{INPUT_FILE}\n\n"
        "Please run Stage 8 evaluation first."
    )
df = pd.read_csv(
    INPUT_FILE
)



required_columns = [
    "ID",
    "true_severity",
    "predicted_severity",
    "alpha",
    "gait_embedding_norm",
    "clinical_embedding_norm",
    "joint_representation_norm",
    "severity_representation_norm",
    "impairment_representation_norm"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if len(missing_columns) > 0:

    raise ValueError(
        "\nRequired columns are missing:\n"
        +
        "\n".join(
            missing_columns
        )
    )



print()
print("=" * 75)
print("CMT-SCOPE AI")
print("STAGE 11: DUAL-AXIS PATIENT REPRESENTATION")
print("=" * 75)

print()

print(
    "Number of test patients:",
    len(df)
)



#  CREATE DUAL-AXIS CSV


dual_axis_df = df[
    [
        "ID",
        "true_severity",
        "predicted_severity",
        "alpha",
        "gait_embedding_norm",
        "clinical_embedding_norm",
        "joint_representation_norm",
        "severity_representation_norm",
        "impairment_representation_norm"
    ]
].copy()


dual_axis_df.rename(
    columns={
        "ID":
            "Patient_ID",

        "alpha":
            "Adaptive_Alpha",

        "gait_embedding_norm":
            "Gait_Representation_Strength",

        "clinical_embedding_norm":
            "Clinical_Representation_Strength",

        "joint_representation_norm":
            "Joint_Representation_Strength",

        "severity_representation_norm":
            "Severity_Representation_Strength",

        "impairment_representation_norm":
            "Impairment_Representation_Strength"
    },
    inplace=True
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "dual_axis_patient_profile.csv"
)


dual_axis_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print()
print(
    "Saved:",
    OUTPUT_CSV
)

print()
print(
    "Creating dual-axis plot..."
)


plt.figure(
    figsize=(10, 7)
)


plt.scatter(
    dual_axis_df[
        "Gait_Representation_Strength"
    ],
    dual_axis_df[
        "Clinical_Representation_Strength"
    ],
    s=120
)


for _, row in dual_axis_df.iterrows():

    plt.annotate(
        str(
            row["Patient_ID"]
        ),
        (
            row[
                "Gait_Representation_Strength"
            ],
            row[
                "Clinical_Representation_Strength"
            ]
        ),
        xytext=(7, 7),
        textcoords="offset points"
    )



plt.xlabel(
    "Gait Representation Strength"
)

plt.ylabel(
    "Clinical Representation Strength"
)

plt.title(
    "CMT1A Dual-Axis Patient Representation"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

OUTPUT_PLOT = os.path.join(
    RESULTS_DIR,
    "dual_axis_patient_profile.png"
)


plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    OUTPUT_PLOT
)



#  REPRESENTATION SUMMARY


print()
print("=" * 75)

print(
    "REPRESENTATION SUMMARY"
)

print(
    "=" * 75
)

print()

print(
    dual_axis_df[
        [
            "Patient_ID",
            "Gait_Representation_Strength",
            "Clinical_Representation_Strength",
            "Joint_Representation_Strength",
            "Severity_Representation_Strength",
            "Impairment_Representation_Strength"
        ]
    ].to_string(
        index=False
    )
)



# 14. ADAPTIVE FUSION SUMMARY


print()
print("=" * 75)

print(
    "ADAPTIVE FUSION"
)

print(
    "=" * 75
)

print()

print(
    "Mean Adaptive Alpha:",
    round(
        dual_axis_df[
            "Adaptive_Alpha"
        ].mean(),
        6
    )
)

print(
    "Minimum Adaptive Alpha:",
    round(
        dual_axis_df[
            "Adaptive_Alpha"
        ].min(),
        6
    )
)

print(
    "Maximum Adaptive Alpha:",
    round(
        dual_axis_df[
            "Adaptive_Alpha"
        ].max(),
        6
    )
)


# ============================================================
# 15. INTERPRETATION
# ============================================================

print()
print(
    "Interpretation:"
)

print(
    "The dual-axis representation visualizes the"
)

print(
    "learned gait and clinical representation strengths."
)

print(
    "Adaptive alpha is the learned fusion coefficient"
)

print(
    "used to combine the two representations."
)

print()
print("=" * 75)

print(
    "STAGE 11 COMPLETE"
)

print("=" * 75)

print()