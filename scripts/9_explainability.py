import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

sys.path.insert(
    0,
    SCRIPT_DIR
)


# ============================================================
# 3. START
# ============================================================

print()
print("=" * 70)
print("STAGE 9: EXPLAINABILITY")
print("CMT1A SEVERITY CLASSIFICATION")
print("=" * 70)
print()


# ============================================================
# 4. LOAD STAGE 6 MODEL
# ============================================================

print("Loading 6_model.py...")

model_path = os.path.join(
    SCRIPT_DIR,
    "6_model.py"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"\nModel file not found:\n{model_path}"
    )

model_spec = spec_from_file_location(
    "model_6",
    model_path
)

if model_spec is None:
    raise ImportError(
        "Could not create module specification "
        "for 6_model.py"
    )

model_module = module_from_spec(
    model_spec
)

model_spec.loader.exec_module(
    model_module
)

CMTSeverityModel = (
    model_module.CMTSeverityModel
)

DEVICE = model_module.DEVICE

print(
    "6_model.py loaded successfully."
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# 5. LOAD STAGE 5C DATASET
# ============================================================

print()
print("Loading 5c_patient_dataset.py...")

dataset_path = os.path.join(
    SCRIPT_DIR,
    "5c_patient_dataset.py"
)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"\nDataset file not found:\n{dataset_path}"
    )

dataset_spec = spec_from_file_location(
    "dataset_5c",
    dataset_path
)

if dataset_spec is None:
    raise ImportError(
        "Could not create module specification "
        "for 5c_patient_dataset.py"
    )

dataset_module = module_from_spec(
    dataset_spec
)

dataset_spec.loader.exec_module(
    dataset_module
)

train_dataset = (
    dataset_module.train_dataset
)

test_dataset = (
    dataset_module.test_dataset
)

print(
    "5c_patient_dataset.py loaded successfully."
)

print(
    "Training patients:",
    len(train_dataset)
)

print(
    "Test patients:",
    len(test_dataset)
)


# ============================================================
# 6. MODEL CHECKPOINT
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_cmt_severity_model.pth"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nClassification model not found:\n"
        f"{MODEL_PATH}\n\n"
        "Run Stage 7 training first."
    )

print()
print(
    "Classification checkpoint found:"
)

print(
    MODEL_PATH
)


# ============================================================
# 7. RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 8. CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Severe"
]

CLINICAL_FEATURES = [
    "Age (Years)",
    "Gender",
    "Pinprick",
    "Vibration"
]


# ============================================================
# 9. CREATE MODEL
# ============================================================

print()
print("Creating CMT severity model...")

model = CMTSeverityModel().to(
    DEVICE
)


# ============================================================
# 10. LOAD CLASSIFICATION CHECKPOINT
# ============================================================

print()
print("Loading trained classification checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if not isinstance(
    checkpoint,
    dict
):

    raise ValueError(
        "Invalid classification checkpoint."
    )


if "model_state_dict" not in checkpoint:

    raise ValueError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print(
    "Classification model loaded successfully."
)

print(
    "Best epoch:",
    checkpoint.get(
        "epoch",
        "Unknown"
    )
)

print(
    "Best validation accuracy:",
    checkpoint.get(
        "best_val_acc",
        checkpoint.get(
            "best_val_accuracy",
            "Unknown"
        )
    )
)


# ============================================================
# 11. SELECT TEST PATIENT
# ============================================================

print()
print("=" * 70)
print("SELECTING TEST PATIENT")
print("=" * 70)

if len(test_dataset) == 0:

    raise ValueError(
        "Test dataset is empty."
    )


# ------------------------------------------------------------
# Select first test patient
# ------------------------------------------------------------

patient_index = 0

patient = test_dataset[
    patient_index
]


# ------------------------------------------------------------
# Stage 5C returns:
#
# patient_id
# gait
# clinical
# severity
# ------------------------------------------------------------

patient_id = patient[0]

gait = patient[1].to(
    DEVICE
)

clinical = patient[2].to(
    DEVICE
)

true_label = patient[3].to(
    DEVICE
)



if gait.ndim != 3:

    raise ValueError(
        f"Expected gait shape "
        f"[windows,256,3], got {gait.shape}"
    )


if gait.shape[1] != 256:

    raise ValueError(
        f"Expected 256 samples per window, "
        f"got {gait.shape[1]}"
    )


if gait.shape[2] != 3:

    raise ValueError(
        f"Expected 3 gait channels, "
        f"got {gait.shape[2]}"
    )


if clinical.ndim != 1:

    raise ValueError(
        f"Expected clinical shape [4], "
        f"got {clinical.shape}"
    )


if clinical.shape[0] != 4:

    raise ValueError(
        f"Expected 4 clinical features, "
        f"got {clinical.shape[0]}"
    )


print()

print(
    "Patient:",
    patient_id
)

print(
    "Gait shape:",
    tuple(gait.shape)
)

print(
    "Clinical shape:",
    tuple(clinical.shape)
)

print(
    "Actual severity:",
    CLASS_NAMES[
        int(true_label.item())
    ]
)


# ============================================================
# PART A
# BASELINE CLASSIFICATION PREDICTION
# ============================================================

print()
print("=" * 70)
print("BASELINE CLASSIFICATION PREDICTION")
print("=" * 70)


with torch.no_grad():

    baseline_output = model(
        gait,
        clinical
    )


# ------------------------------------------------------------
# Extract logits
# ------------------------------------------------------------

baseline_logits = (
    baseline_output[
        "logits"
    ]
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)


# ------------------------------------------------------------
# Calculate probabilities
# ------------------------------------------------------------

baseline_probabilities = (
    torch.softmax(
        torch.tensor(
            baseline_logits
        ),
        dim=0
    )
    .numpy()
)


# ------------------------------------------------------------
# Predicted class
# ------------------------------------------------------------

baseline_class = int(
    np.argmax(
        baseline_probabilities
    )
)

baseline_severity = (
    CLASS_NAMES[
        baseline_class
    ]
)

baseline_confidence = float(
    baseline_probabilities[
        baseline_class
    ]
)


# ------------------------------------------------------------
# Adaptive fusion alpha
# ------------------------------------------------------------

baseline_alpha = float(
    baseline_output[
        "alpha"
    ]
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)[0]
)


print()

print(
    "Actual severity:",
    CLASS_NAMES[
        int(true_label.item())
    ]
)

print(
    "Predicted severity:",
    baseline_severity
)

print(
    f"Confidence: "
    f"{baseline_confidence:.4f}"
)

print()

print(
    "Class probabilities:"
)

for i in range(
    len(CLASS_NAMES)
):

    print(
        f"  {CLASS_NAMES[i]} = "
        f"{baseline_probabilities[i]:.4f}"
    )


print()

print(
    "Adaptive fusion alpha:",
    f"{baseline_alpha:.4f}"
)


# ============================================================
# SAVE BASELINE PREDICTION
# ============================================================

baseline_file = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_baseline_explanation.csv"
)

baseline_df = pd.DataFrame(
    {
        "Patient_ID": [
            patient_id
        ],

        "Actual_Severity": [
            CLASS_NAMES[
                int(true_label.item())
            ]
        ],

        "Predicted_Severity": [
            baseline_severity
        ],

        "Confidence": [
            baseline_confidence
        ],

        "Mild_Probability": [
            baseline_probabilities[0]
        ],

        "Moderate_Probability": [
            baseline_probabilities[1]
        ],

        "Severe_Probability": [
            baseline_probabilities[2]
        ],

        "Adaptive_Fusion_Alpha": [
            baseline_alpha
        ]
    }
)

baseline_df.to_csv(
    baseline_file,
    index=False
)

print()
print(
    "Baseline explanation saved:"
)

print(
    baseline_file
)



# SHAP CLINICAL EXPLAINABILITY


print()
print("=" * 70)
print("PART A: SHAP CLINICAL EXPLANATION")
print("=" * 70)



#  SHAP PREDICTION FUNCTION


def clinical_prediction(
    clinical_numpy
):
    """
    SHAP prediction function.

    Clinical features are changed while
    the patient's gait remains fixed.

    Returns the probability of the
    currently predicted severity class.
    """

    clinical_numpy = np.asarray(
        clinical_numpy,
        dtype=np.float32
    )

    if clinical_numpy.ndim == 1:

        clinical_numpy = (
            clinical_numpy.reshape(
                1,
                -1
            )
        )


    predictions = []


    for i in range(
        clinical_numpy.shape[0]
    ):

        current_clinical = torch.tensor(
            clinical_numpy[i],
            dtype=torch.float32,
            device=DEVICE
        )


        with torch.no_grad():

            output = model(
                gait,
                current_clinical
            )


        probabilities = (
            output[
                "probabilities"
            ]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        predicted_class_probability = (
            probabilities[
                baseline_class
            ]
        )


        predictions.append(
            predicted_class_probability
        )


    return np.asarray(
        predictions,
        dtype=np.float64
    )



#  CREATE SHAP BACKGROUND DATA


print()
print(
    "Preparing SHAP background data..."
)

background_values = []


for i in range(
    len(train_dataset)
):

    item = train_dataset[i]

    clinical_value = (
        item[2]
        .detach()
        .cpu()
        .numpy()
    )

    background_values.append(
        clinical_value
    )


background_values = np.asarray(
    background_values,
    dtype=np.float32
)


if background_values.shape[0] == 0:

    raise ValueError(
        "Training clinical background "
        "data is empty."
    )


print(
    "SHAP background shape:",
    background_values.shape
)



#  PATIENT CLINICAL VALUES


patient_clinical = (
    clinical
    .detach()
    .cpu()
    .numpy()
)


patient_clinical = (
    patient_clinical
    .reshape(
        1,
        -1
    )
)


print()

print(
    "Patient clinical values:"
)

for i in range(
    len(CLINICAL_FEATURES)
):

    print(
        f"{CLINICAL_FEATURES[i]}: "
        f"{patient_clinical[0][i]}"
    )



#  CREATE SHAP EXPLAINER


print()
print(
    "Creating SHAP KernelExplainer..."
)

explainer = shap.KernelExplainer(
    clinical_prediction,
    background_values
)



#  CALCULATE SHAP VALUES


print()
print(
    "Calculating SHAP values..."
)

shap_values = explainer.shap_values(
    patient_clinical,
    nsamples=100
)



#  HANDLE SHAP OUTPUT


if isinstance(
    shap_values,
    list
):

    shap_values = (
        shap_values[0]
    )


shap_values = np.asarray(
    shap_values
)


shap_values = (
    shap_values.reshape(-1)
)


if len(shap_values) != len(
    CLINICAL_FEATURES
):

    raise ValueError(
        "Number of SHAP values does not "
        "match the number of clinical features."
    )



# SHAP VALUES


print()
print("=" * 70)
print("CLINICAL SHAP VALUES")
print("=" * 70)

print()

print(
    "Explained class:",
    baseline_severity
)

print()

for i in range(
    len(CLINICAL_FEATURES)
):

    print(
        f"{CLINICAL_FEATURES[i]:<20}"
        f" SHAP = "
        f"{shap_values[i]:.6f}"
    )



#  SHAP CSV


shap_df = pd.DataFrame(
    {
        "Feature":
            CLINICAL_FEATURES,

        "Clinical_Value":
            patient_clinical[0],

        "SHAP_Value":
            shap_values,

        "Absolute_SHAP":
            np.abs(
                shap_values
            ),

        "Explained_Class":
            baseline_severity
    }
)


shap_file = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_clinical_shap.csv"
)


shap_df.to_csv(
    shap_file,
    index=False
)


print()
print(
    "SHAP CSV saved:"
)

print(
    shap_file
)

print()
print(
    "Creating SHAP plot..."
)

plt.figure(
    figsize=(8, 5)
)

plt.barh(
    CLINICAL_FEATURES,
    shap_values
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "SHAP Value"
)

plt.ylabel(
    "Clinical Feature"
)

plt.title(
    f"Clinical SHAP Contributions - "
    f"{patient_id} - {baseline_severity}"
)

plt.tight_layout()


shap_plot = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_clinical_shap.png"
)


plt.savefig(
    shap_plot,
    dpi=300
)

plt.close()


print(
    "SHAP plot saved:"
)

print(
    shap_plot
)


# ============================================================
# PART C
# TEMPORAL GAIT EXPLAINABILITY
# ============================================================

print()
print("=" * 70)
print("PART B: TEMPORAL GAIT EXPLANATION")
print("=" * 70)


# ============================================================
# 22. GAIT INFORMATION
# ============================================================

NUMBER_OF_WINDOWS = (
    gait.shape[0]
)

WINDOW_LENGTH = (
    gait.shape[1]
)

NUMBER_OF_CHANNELS = (
    gait.shape[2]
)


print()

print(
    "Complete patient gait shape:",
    tuple(gait.shape)
)

print(
    "Number of gait windows:",
    NUMBER_OF_WINDOWS
)

print(
    "Samples per window:",
    WINDOW_LENGTH
)

print(
    "Gait channels:",
    NUMBER_OF_CHANNELS
)


# ============================================================
# 23. TEMPORAL OCCLUSION SETTINGS
# ============================================================

NUMBER_OF_SEGMENTS = 8

SEGMENT_LENGTH = (
    WINDOW_LENGTH //
    NUMBER_OF_SEGMENTS
)


temporal_results = []


print()

print(
    "Number of temporal segments:",
    NUMBER_OF_SEGMENTS
)

print(
    "Segment length:",
    SEGMENT_LENGTH
)


# ============================================================
# 24. TEMPORAL OCCLUSION LOOP
# ============================================================

for segment in range(
    NUMBER_OF_SEGMENTS
):

    start = (
        segment *
        SEGMENT_LENGTH
    )


    if segment == (
        NUMBER_OF_SEGMENTS - 1
    ):

        end = WINDOW_LENGTH

    else:

        end = (
            start +
            SEGMENT_LENGTH
        )


    # --------------------------------------------------------
    # Copy complete gait
    # --------------------------------------------------------

    modified_gait = gait.clone()


    # --------------------------------------------------------
    # Mask this temporal region
    # ACROSS ALL GAIT WINDOWS
    #
    # This is more appropriate than
    # masking only the first window.
    # --------------------------------------------------------

    modified_gait[
        :,
        start:end,
        :
    ] = 0.0


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            modified_gait,
            clinical
        )


    modified_probabilities = (
        output[
            "probabilities"
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


    modified_class_probability = float(
        modified_probabilities[
            baseline_class
        ]
    )


    modified_class = int(
        np.argmax(
            modified_probabilities
        )
    )


    modified_severity = (
        CLASS_NAMES[
            modified_class
        ]
    )


    # --------------------------------------------------------
    # Change in predicted-class probability
    # --------------------------------------------------------

    probability_change = (
        baseline_confidence -
        modified_class_probability
    )


    absolute_change = abs(
        probability_change
    )


    temporal_results.append(
        {

            "Segment":
                segment + 1,

            "Start_Sample":
                start,

            "End_Sample":
                end,

            "Baseline_Severity":
                baseline_severity,

            "Baseline_Class_Probability":
                baseline_confidence,

            "Masked_Class_Probability":
                modified_class_probability,

            "Probability_Change":
                probability_change,

            "Absolute_Probability_Change":
                absolute_change,

            "Masked_Predicted_Severity":
                modified_severity
        }
    )


    print(
        f"Segment {segment + 1}: "
        f"{start}-{end} | "
        f"Baseline probability = "
        f"{baseline_confidence:.4f} | "
        f"Masked probability = "
        f"{modified_class_probability:.4f} | "
        f"Change = "
        f"{probability_change:.6f}"
    )


# ============================================================
# 25. TEMPORAL DATAFRAME
# ============================================================

temporal_df = pd.DataFrame(
    temporal_results
)


# ============================================================
# 26. SAVE TEMPORAL CSV
# ============================================================

temporal_file = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_gait_temporal_explanation.csv"
)


temporal_df.to_csv(
    temporal_file,
    index=False
)


print()
print(
    "Temporal explanation CSV saved:"
)

print(
    temporal_file
)


# ============================================================
# 27. TEMPORAL PLOT
# ============================================================

print()
print(
    "Creating temporal explanation plot..."
)

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    temporal_df["Segment"],
    temporal_df[
        "Probability_Change"
    ]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Temporal Segment"
)

plt.ylabel(
    "Change in Predicted-Class Probability"
)

plt.title(
    f"Gait Temporal Occlusion Explanation - "
    f"{patient_id}"
)

plt.xticks(
    temporal_df["Segment"]
)

plt.tight_layout()


temporal_plot = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_gait_temporal_explanation.png"
)


plt.savefig(
    temporal_plot,
    dpi=300
)

plt.close()


print(
    "Temporal plot saved:"
)

print(
    temporal_plot
)


# ============================================================
# 28. FIND MOST INFLUENTIAL TEMPORAL SEGMENT
# ============================================================

largest_change_index = (
    temporal_df[
        "Absolute_Probability_Change"
    ]
    .idxmax()
)


most_influential_segment = (
    temporal_df.loc[
        largest_change_index,
        "Segment"
    ]
)


largest_change = (
    temporal_df.loc[
        largest_change_index,
        "Absolute_Probability_Change"
    ]
)


most_influential_start = (
    temporal_df.loc[
        largest_change_index,
        "Start_Sample"
    ]
)


most_influential_end = (
    temporal_df.loc[
        largest_change_index,
        "End_Sample"
    ]
)


print()
print("=" * 70)
print("MOST INFLUENTIAL TEMPORAL REGION")
print("=" * 70)

print()

print(
    "Segment:",
    int(most_influential_segment)
)

print(
    "Sample range:",
    int(most_influential_start),
    "-",
    int(most_influential_end)
)

print(
    "Absolute probability change:",
    f"{largest_change:.6f}"
)


# ============================================================
# PART D
# ADAPTIVE FUSION INFORMATION
# ============================================================

print()
print("=" * 70)
print("ADAPTIVE FUSION INFORMATION")
print("=" * 70)

print()

print(
    f"Adaptive fusion alpha: "
    f"{baseline_alpha:.4f}"
)

print()

print(
    "Alpha represents the learned fusion "
    "weight between the gait and clinical "
    "representations."
)

print(
    "It should not be interpreted as "
    "clinical feature importance."
)


# ============================================================
# 29. SAVE FUSION INFORMATION
# ============================================================

fusion_file = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_adaptive_fusion.csv"
)


fusion_df = pd.DataFrame(
    {
        "Patient_ID": [
            patient_id
        ],

        "Predicted_Severity": [
            baseline_severity
        ],

        "Adaptive_Fusion_Alpha": [
            baseline_alpha
        ],

        "Gait_Fusion_Weight": [
            baseline_alpha
        ],

        "Clinical_Fusion_Weight": [
            1.0 - baseline_alpha
        ]
    }
)


fusion_df.to_csv(
    fusion_file,
    index=False
)


print(
    "Adaptive fusion file saved:"
)

print(
    fusion_file
)


# ============================================================
# PART E
# EXPLAINABILITY SUMMARY
# ============================================================

summary_file = os.path.join(
    RESULTS_DIR,
    f"{patient_id}_explainability_summary.txt"
)


with open(
    summary_file,
    "w"
) as file:

    file.write(
        "CMT1A SEVERITY EXPLAINABILITY SUMMARY\n"
    )

    file.write(
        "=" * 60 +
        "\n\n"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    file.write(
        f"Patient ID: {patient_id}\n"
    )

    file.write(
        f"Actual Severity: "
        f"{CLASS_NAMES[int(true_label.item())]}\n"
    )

    file.write(
        f"Predicted Severity: "
        f"{baseline_severity}\n"
    )

    file.write(
        f"Prediction Confidence: "
        f"{baseline_confidence:.6f}\n"
    )

    file.write(
        f"Mild Probability: "
        f"{baseline_probabilities[0]:.6f}\n"
    )

    file.write(
        f"Moderate Probability: "
        f"{baseline_probabilities[1]:.6f}\n"
    )

    file.write(
        f"Severe Probability: "
        f"{baseline_probabilities[2]:.6f}\n"
    )

    file.write(
        f"Adaptive Fusion Alpha: "
        f"{baseline_alpha:.6f}\n\n"
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    file.write(
        "CLINICAL EXPLAINABILITY\n"
    )

    file.write(
        "-" * 40 +
        "\n"
    )

    file.write(
        f"Explained class: "
        f"{baseline_severity}\n\n"
    )

    file.write(
        "SHAP estimates the contribution "
        "of each clinical feature to the "
        "predicted-class probability while "
        "keeping the patient's gait fixed.\n\n"
    )


    for i in range(
        len(CLINICAL_FEATURES)
    ):

        file.write(
            f"{CLINICAL_FEATURES[i]}: "
            f"SHAP = "
            f"{shap_values[i]:.6f}\n"
        )


    file.write(
        "\n"
    )


    # --------------------------------------------------------
    # Gait
    # --------------------------------------------------------

    file.write(
        "GAIT EXPLAINABILITY\n"
    )

    file.write(
        "-" * 40 +
        "\n"
    )

    file.write(
        "Temporal occlusion was used to "
        "examine how masking different "
        "temporal regions changes the "
        "predicted-class probability.\n\n"
    )

    file.write(
        "The analysis masks the selected "
        "temporal region across all gait "
        "windows.\n\n"
    )

    file.write(
        "This is a local model sensitivity "
        "analysis and does not establish "
        "clinical causation.\n\n"
    )


    file.write(
        f"Number of gait windows: "
        f"{NUMBER_OF_WINDOWS}\n"
    )

    file.write(
        f"Samples per window: "
        f"{WINDOW_LENGTH}\n"
    )

    file.write(
        f"Number of channels: "
        f"{NUMBER_OF_CHANNELS}\n\n"
    )


    file.write(
        f"Most influential temporal segment: "
        f"{int(most_influential_segment)}\n"
    )

    file.write(
        f"Sample range: "
        f"{int(most_influential_start)}-"
        f"{int(most_influential_end)}\n"
    )

    file.write(
        f"Largest absolute probability change: "
        f"{largest_change:.6f}\n\n"
    )


    # --------------------------------------------------------
    # Limitation
    # --------------------------------------------------------

    file.write(
        "INTERPRETATION LIMITATIONS\n"
    )

    file.write(
        "-" * 40 +
        "\n"
    )

    file.write(
        "SHAP values describe local model "
        "feature contributions and should "
        "not be interpreted as causal effects.\n\n"
    )

    file.write(
        "Temporal occlusion measures local "
        "prediction sensitivity to masked "
        "gait regions and does not identify "
        "a clinical biomarker.\n\n"
    )

    file.write(
        "Adaptive fusion alpha represents "
        "a learned model fusion weight and "
        "is not a measure of clinical importance.\n"
    )


# ============================================================
# 30. DISPLAY OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("STAGE 9 OUTPUT FILES")
print("=" * 70)

print()

print(
    "1. Baseline prediction:"
)

print(
    baseline_file
)

print()

print(
    "2. Clinical SHAP CSV:"
)

print(
    shap_file
)

print()

print(
    "3. Clinical SHAP plot:"
)

print(
    shap_plot
)

print()

print(
    "4. Gait temporal CSV:"
)

print(
    temporal_file
)

print()

print(
    "5. Gait temporal plot:"
)

print(
    temporal_plot
)

print()

print(
    "6. Adaptive fusion CSV:"
)

print(
    fusion_file
)

print()

print(
    "7. Explainability summary:"
)

print(
    summary_file
)


# ============================================================
# 31. COMPLETE
# ============================================================

print()
print("=" * 70)
print("STAGE 9 COMPLETE")
print("=" * 70)

print()

print(
    "Classification prediction explanation : COMPLETE"
)

print(
    "Clinical SHAP explanation             : COMPLETE"
)

print(
    "Temporal gait explanation             : COMPLETE"
)

print(
    "Adaptive fusion analysis              : COMPLETE"
)

print(
    "Explainability files generated        : COMPLETE"
)

print()