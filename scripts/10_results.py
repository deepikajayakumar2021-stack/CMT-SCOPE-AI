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

test_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)



if not os.path.exists(test_file):

    raise FileNotFoundError(
        "\nTest results were not found.\n\n"
        f"Expected file:\n{test_file}\n\n"
        "Please run Stage 8 evaluation first."
    )



df = pd.read_csv(
    test_file
)



required_columns = [
    "ID",
    "true_severity",
    "predicted_severity",
    "confidence",
    "prob_mild",
    "prob_moderate",
    "prob_severe",
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
        + "\n".join(missing_columns)
    )


print()
print("=" * 75)
print("CMT-SCOPE AI")
print("STAGE 10: RESULTS VISUALIZATION")
print("=" * 75)

print()

print(
    "Number of test patients:",
    len(df)
)


 

df["alpha"] = pd.to_numeric(
    df["alpha"],
    errors="coerce"
)



#  ACTUAL VS PREDICTED SEVERITY


print()
print(
    "Creating actual vs predicted severity plot..."
)

severity_order = [
    "Mild",
    "Moderate",
    "Severe"
]

severity_to_number = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

df["true_numeric"] = df[
    "true_severity"
].map(
    severity_to_number
)

df["predicted_numeric"] = df[
    "predicted_severity"
].map(
    severity_to_number
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["ID"].astype(str),
    df["true_numeric"],
    marker="o",
    linewidth=2,
    label="Actual Severity"
)

plt.plot(
    df["ID"].astype(str),
    df["predicted_numeric"],
    marker="o",
    linewidth=2,
    label="Predicted Severity"
)

plt.xlabel(
    "Patient"
)

plt.ylabel(
    "Severity Class"
)

plt.yticks(
    [0, 1, 2],
    severity_order
)

plt.title(
    "Actual vs Predicted CMT1A Severity"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


actual_predicted_file = os.path.join(
    RESULTS_DIR,
    "actual_vs_predicted_severity.png"
)

plt.savefig(
    actual_predicted_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    actual_predicted_file
)




print()
print(
    "Creating prediction confidence plot..."
)


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["ID"].astype(str),
    df["confidence"]
)

plt.xlabel(
    "Patient"
)

plt.ylabel(
    "Prediction Confidence"
)

plt.title(
    "Severity Prediction Confidence"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


confidence_file = os.path.join(
    RESULTS_DIR,
    "severity_prediction_confidence.png"
)

plt.savefig(
    confidence_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    confidence_file
)



#  ADAPTIVE FUSION


print()
print(
    "Creating adaptive fusion plot..."
)


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["ID"].astype(str),
    df["alpha"]
)

plt.axhline(
    0.5,
    linewidth=1,
    linestyle="--",
    label="Equal Fusion Reference"
)

plt.xlabel(
    "Patient"
)

plt.ylabel(
    "Adaptive Fusion Alpha"
)

plt.title(
    "Patient-Specific Adaptive Gait-Clinical Fusion"
)

plt.xticks(
    rotation=45
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


fusion_file = os.path.join(
    RESULTS_DIR,
    "adaptive_fusion.png"
)

plt.savefig(
    fusion_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    fusion_file
)



#  ADAPTIVE FUSION TABLE


fusion_df = df[
    [
        "ID",
        "alpha"
    ]
].copy()


fusion_df.rename(
    columns={
        "ID": "Patient_ID",
        "alpha": "Adaptive_Alpha"
    },
    inplace=True
)


fusion_df[
    "Gait_Weight"
] = fusion_df[
    "Adaptive_Alpha"
]

fusion_df[
    "Clinical_Weight"
] = (
    1.0
    -
    fusion_df[
        "Adaptive_Alpha"
    ]
)


fusion_table_file = os.path.join(
    RESULTS_DIR,
    "adaptive_fusion_summary.csv"
)


fusion_df.to_csv(
    fusion_table_file,
    index=False
)


print(
    "Saved:",
    fusion_table_file
)



#  DUAL-AXIS REPRESENTATION


print()
print(
    "Creating dual-axis representation plot..."
)


plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    df[
        "gait_embedding_norm"
    ],
    df[
        "clinical_embedding_norm"
    ],
    s=100
)


for _, row in df.iterrows():

    plt.annotate(
        str(row["ID"]),
        (
            row[
                "gait_embedding_norm"
            ],
            row[
                "clinical_embedding_norm"
            ]
        ),
        xytext=(6, 6),
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


dual_axis_file = os.path.join(
    RESULTS_DIR,
    "dual_axis_patient_profile.png"
)

plt.savefig(
    dual_axis_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    dual_axis_file
)



#  TEST RESULTS TABLE


print()
print("=" * 75)
print("TEST RESULTS TABLE")
print("=" * 75)
print()


display_columns = [
    "ID",
    "true_severity",
    "predicted_severity",
    "confidence",
    "prob_mild",
    "prob_moderate",
    "prob_severe",
    "alpha"
]


print(
    df[
        display_columns
    ].to_string(
        index=False
    )
)



# ACCURACY


correct_predictions = (
    df["true_severity"]
    ==
    df["predicted_severity"]
).sum()


total_predictions = len(df)


accuracy = (
    correct_predictions
    /
    total_predictions
)


print()
print(
    "Classification Accuracy:",
    round(
        accuracy,
        4
    )
)



# PREDICTION DISTRIBUTION


print()
print(
    "Predicted Severity Distribution:"
)

print()

print(
    df[
        "predicted_severity"
    ].value_counts()
    .reindex(
        severity_order,
        fill_value=0
    )
    .to_string()
)



#  TRUE SEVERITY DISTRIBUTION


print()
print(
    "True Severity Distribution:"
)

print()

print(
    df[
        "true_severity"
    ].value_counts()
    .reindex(
        severity_order,
        fill_value=0
    )
    .to_string()
)



# ADAPTIVE FUSION SUMMARY


print()
print(
    "=" * 75
)

print(
    "ADAPTIVE FUSION SUMMARY"
)

print(
    "=" * 75
)

print()

print(
    "Mean Adaptive Alpha:",
    round(
        df["alpha"].mean(),
        6
    )
)

print(
    "Minimum Adaptive Alpha:",
    round(
        df["alpha"].min(),
        6
    )
)

print(
    "Maximum Adaptive Alpha:",
    round(
        df["alpha"].max(),
        6
    )
)



#  INTERPRETATION


print()
print(
    "Adaptive Fusion Interpretation:"
)

print(
    "Alpha represents the learned coefficient used to"
)

print(
    "combine gait and clinical representations."
)

print(
    "Alpha is NOT a percentage of clinical importance."
)



# CHECK IF ALL TEST PREDICTIONS ARE MILD


if (
    df[
        "predicted_severity"
    ]
    == "Mild"
).all():

    print()
    print(
        "NOTE:"
    )

    print(
        "All current test patients were predicted as Mild."
    )

    print(
        "This describes the current held-out test set only."
    )

    print(
        "It does not mean that all CMT1A patients are Mild."
    )



print()
print("=" * 75)

print(
    "GENERATED FILES"
)

print("=" * 75)

print()

print(
    "1.",
    actual_predicted_file
)

print(
    "2.",
    confidence_file
)

print(
    "3.",
    fusion_file
)

print(
    "4.",
    fusion_table_file
)

print(
    "5.",
    dual_axis_file
)




print()
print("=" * 75)

print(
    "STAGE 10 COMPLETE"
)

print("=" * 75)

print()