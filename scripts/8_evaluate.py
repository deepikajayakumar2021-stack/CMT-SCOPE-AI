
import os
import sys
import json
import importlib.util

import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = r"D:\CMT_PROJECT"

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "6_model.py"
)

DATASET_FILE = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "5C_patient_dataset.py"
)

CHECKPOINT_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_cmt_severity_model.pth"
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "stage5c_patient_manifest.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Severe"
]

NUM_CLASSES = 3


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STAGE 8: TEST EVALUATION")
print("CMT1A SEVERITY CLASSIFICATION")
print("=" * 70)

print()
print("Device:", device)



def load_module(
    file_path,
    module_name
):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            file_path
        )
    )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module



#  LOAD STAGE 6 MODEL


print()
print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model_module = load_module(
    MODEL_FILE,
    "stage6_model"
)


# ------------------------------------------------------------
# Find CMTSeverityModel
# ------------------------------------------------------------

if not hasattr(
    model_module,
    "CMTSeverityModel"
):

    raise AttributeError(
        "CMTSeverityModel was not found "
        "in 6_model.py"
    )


ModelClass = (
    model_module.CMTSeverityModel
)


print(
    "Model class found:",
    ModelClass.__name__
)


# ============================================================
# 6. CREATE MODEL
#
# IMPORTANT:
# Your actual Stage 6 constructor is:
#
#     CMTSeverityModel()
#
# It does NOT accept d_model, clinical_dim,
# fusion_dim, or num_classes arguments.
# ============================================================

model = ModelClass().to(
    device
)

print()
print("=" * 70)
print("LOADING BEST TRAINED MODEL")
print("=" * 70)


if not os.path.exists(
    CHECKPOINT_FILE
):

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_FILE}"
    )


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)


# ------------------------------------------------------------
# Determine checkpoint state dictionary
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# 8. REMOVE DataParallel PREFIX IF PRESENT
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        clean_key = key[
            len("module.") :
        ]

    else:

        clean_key = key


    clean_state_dict[
        clean_key
    ] = value


# ============================================================
# 9. LOAD MODEL WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

except RuntimeError as error:

    print()
    print(
        "ERROR: Checkpoint does not match "
        "the current Stage 6 model."
    )

    print()
    print(error)

    raise


model.eval()


print(
    "Best model loaded successfully."
)


# ============================================================
# 10. CHECKPOINT INFORMATION
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint["epoch"]
        )

    if "best_val_acc" in checkpoint:

        print(
            "Best validation accuracy:",
            checkpoint["best_val_acc"]
        )

    elif "best_val_accuracy" in checkpoint:

        print(
            "Best validation accuracy:",
            checkpoint["best_val_accuracy"]
        )


# ============================================================
# 11. LOAD STAGE 5C DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)


dataset_module = load_module(
    DATASET_FILE,
    "stage5c_dataset"
)


if not hasattr(
    dataset_module,
    "CMT1APatientDataset"
):

    raise AttributeError(
        "CMT1APatientDataset was not found "
        "in 5C_patient_dataset.py"
    )


DatasetClass = (
    dataset_module.CMT1APatientDataset
)


print(
    "Dataset class found:",
    DatasetClass.__name__
)


# ============================================================
# 12. LOAD STAGE 5C MANIFEST
# ============================================================

if not os.path.exists(
    MANIFEST_FILE
):

    raise FileNotFoundError(
        f"Manifest not found:\n"
        f"{MANIFEST_FILE}"
    )


manifest = pd.read_csv(
    MANIFEST_FILE
)


print(
    "Patients in Stage 5C manifest:",
    len(manifest)
)


# ============================================================
# 13. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "ID",
    "severity_label",
    "severity_name",
    "split"
]


missing_columns = [
    column
    for column in required_columns
    if column not in manifest.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# 14. SELECT TEST PATIENTS
# ============================================================

test_df = manifest[
    manifest["split"]
    .astype(str)
    .str.lower()
    == "test"
].copy()


test_df = test_df.reset_index(
    drop=True
)


print(
    "Test patients found:",
    len(test_df)
)


if len(test_df) == 0:

    raise ValueError(
        "No test patients found "
        "in Stage 5C manifest."
    )


# ============================================================
# 15. CREATE TEST DATASET
#
# EXACT Stage 5C constructor:
#
#     CMT1APatientDataset(dataframe)
# ============================================================

test_dataset = DatasetClass(
    test_df
)


print(
    "Test dataset created successfully.",
    "Test patients:",
    len(test_dataset)
)


# ============================================================
# 16. RUN MODEL
#
# IMPORTANT:
#
# Stage 6 was tested successfully with:
#
#     gait     = [258, 256, 3]
#     clinical = [4]
#
# Therefore:
#
#     NO unsqueeze()
#
# ============================================================

def run_model(
    gait,
    clinical
):

    # --------------------------------------------------------
    # Gait shape
    # --------------------------------------------------------

    if gait.ndim != 3:

        raise ValueError(
            "Expected gait tensor with shape "
            "[num_windows, 256, 3]. "
            f"Got {gait.shape}"
        )


    # --------------------------------------------------------
    # Gait channels
    # --------------------------------------------------------

    if gait.shape[-1] != 3:

        raise ValueError(
            "Expected 3 gait channels "
            "(Accel X, Accel Y, Accel Z). "
            f"Got {gait.shape}"
        )


    # --------------------------------------------------------
    # Clinical shape
    #
    # Stage 6 expects:
    #
    #     [4]
    #
    # NOT:
    #
    #     [1,4]
    # --------------------------------------------------------

    if clinical.ndim != 1:

        raise ValueError(
            "Expected clinical tensor "
            "with shape [4]. "
            f"Got {clinical.shape}"
        )


    if clinical.shape[0] != 4:

        raise ValueError(
            "Expected exactly 4 clinical "
            f"features. Got {clinical.shape}"
        )


    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    gait = gait.to(
        device
    )

    clinical = clinical.to(
        device
    )


    # --------------------------------------------------------
    # Model forward
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            gait,
            clinical
        )


    return output


# ============================================================
# 17. EXTRACT LOGITS AND PROBABILITIES
# ============================================================

def extract_prediction(
    output
):

    if not isinstance(
        output,
        dict
    ):

        raise TypeError(
            "Stage 6 model should return "
            "a dictionary."
        )


    if "logits" not in output:

        raise KeyError(
            "Stage 6 output does not contain "
            "'logits'."
        )


    logits = output[
        "logits"
    ]


    if not torch.is_tensor(
        logits
    ):

        logits = torch.tensor(
            logits,
            dtype=torch.float32,
            device=device
        )


    # --------------------------------------------------------
    # Stage 6 single-patient output:
    #
    # logits = [3]
    # --------------------------------------------------------

    if logits.ndim == 1:

        if logits.shape[0] != NUM_CLASSES:

            raise ValueError(
                "Expected 3 logits. "
                f"Got {logits.shape}"
            )


        probabilities = torch.softmax(
            logits,
            dim=-1
        )


        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )


        logits_numpy = (
            logits
            .detach()
            .cpu()
            .numpy()
        )


        probabilities_numpy = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )


        return (
            logits_numpy,
            probabilities_numpy,
            predicted_class
        )


    # --------------------------------------------------------
    # Support [1,3] if encountered
    # --------------------------------------------------------

    elif logits.ndim == 2:

        if logits.shape != (
            1,
            NUM_CLASSES
        ):

            raise ValueError(
                "Unexpected logits shape: "
                f"{logits.shape}"
            )


        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]


        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )


        logits_numpy = (
            logits[0]
            .detach()
            .cpu()
            .numpy()
        )


        probabilities_numpy = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )


        return (
            logits_numpy,
            probabilities_numpy,
            predicted_class
        )


    else:

        raise ValueError(
            "Unexpected logits dimensions: "
            f"{logits.shape}"
        )


# ============================================================
# 18. EXTRACT ALPHA
# ============================================================

def extract_alpha(
    output
):

    if "alpha" not in output:

        return np.nan


    alpha = output[
        "alpha"
    ]


    if not torch.is_tensor(
        alpha
    ):

        alpha = np.asarray(
            alpha
        )

    else:

        alpha = (
            alpha
            .detach()
            .cpu()
            .numpy()
        )


    if alpha.size == 0:

        return np.nan


    return float(
        alpha.reshape(-1)[0]
    )


# ============================================================
# 19. EXTRACT REPRESENTATION NORM
# ============================================================

def extract_norm(
    output,
    key
):

    if key not in output:

        return np.nan


    value = output[
        key
    ]


    if not torch.is_tensor(
        value
    ):

        return np.nan


    return float(
        torch.norm(
            value.detach()
        )
        .cpu()
        .item()
    )


# ============================================================
# 20. TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING TEST PREDICTIONS")
print("=" * 70)


results = []


for index in range(
    len(test_dataset)
):

    (
        patient_id,
        gait,
        clinical,
        severity
    ) = test_dataset[
        index
    ]


    # --------------------------------------------------------
    # Display original Stage 5C shapes
    # --------------------------------------------------------

    print()
    print(
        f"Evaluating {patient_id}..."
    )

    print(
        "  Gait:",
        tuple(gait.shape)
    )

    print(
        "  Clinical:",
        tuple(clinical.shape)
    )


    # --------------------------------------------------------
    # True class
    # --------------------------------------------------------

    if torch.is_tensor(
        severity
    ):

        true_class = int(
            severity.item()
        )

    else:

        true_class = int(
            severity
        )


    if not (
        0 <= true_class < NUM_CLASSES
    ):

        raise ValueError(
            f"Invalid severity label "
            f"{true_class} for {patient_id}"
        )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    output = run_model(
        gait,
        clinical
    )


    # --------------------------------------------------------
    # Extract prediction
    # --------------------------------------------------------

    (
        logits,
        probabilities,
        predicted_class
    ) = extract_prediction(
        output
    )


    # --------------------------------------------------------
    # Names
    # --------------------------------------------------------

    true_name = (
        CLASS_NAMES[
            true_class
        ]
    )

    predicted_name = (
        CLASS_NAMES[
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # Adaptive fusion
    # --------------------------------------------------------

    alpha = extract_alpha(
        output
    )


    # --------------------------------------------------------
    # Latent representation norms
    # --------------------------------------------------------

    gait_norm = extract_norm(
        output,
        "gait_embedding"
    )

    clinical_norm = extract_norm(
        output,
        "clinical_embedding"
    )

    joint_norm = extract_norm(
        output,
        "joint_representation"
    )

    severity_norm = extract_norm(
        output,
        "severity_representation"
    )

    impairment_norm = extract_norm(
        output,
        "impairment_representation"
    )


    # --------------------------------------------------------
    # Save patient result
    # --------------------------------------------------------

    result = {

        "ID":
            patient_id,

        "true_label":
            true_class,

        "true_severity":
            true_name,

        "predicted_label":
            predicted_class,

        "predicted_severity":
            predicted_name,

        "confidence":
            confidence,

        "prob_mild":
            float(
                probabilities[0]
            ),

        "prob_moderate":
            float(
                probabilities[1]
            ),

        "prob_severe":
            float(
                probabilities[2]
            ),

        "logit_mild":
            float(
                logits[0]
            ),

        "logit_moderate":
            float(
                logits[1]
            ),

        "logit_severe":
            float(
                logits[2]
            ),

        "alpha":
            alpha,

        "gait_embedding_norm":
            gait_norm,

        "clinical_embedding_norm":
            clinical_norm,

        "joint_representation_norm":
            joint_norm,

        "severity_representation_norm":
            severity_norm,

        "impairment_representation_norm":
            impairment_norm
    }


    results.append(
        result
    )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "  True:",
        true_name
    )

    print(
        "  Predicted:",
        predicted_name
    )

    print(
        "  Confidence:",
        f"{confidence:.4f}"
    )

    print(
        "  Probabilities:",
        f"Mild={probabilities[0]:.4f},",
        f"Moderate={probabilities[1]:.4f},",
        f"Severe={probabilities[2]:.4f}"
    )

    if not np.isnan(
        alpha
    ):

        print(
            "  Adaptive fusion alpha:",
            f"{alpha:.4f}"
        )


# ============================================================
# 21. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

predictions_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)


results_df.to_csv(
    predictions_file,
    index=False
)


print()
print(
    "Saved:",
    predictions_file
)


# ============================================================
# 23. TRUE AND PREDICTED LABELS
# ============================================================

y_true = (
    results_df[
        "true_label"
    ]
    .astype(int)
    .to_numpy()
)

y_pred = (
    results_df[
        "predicted_label"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 24. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred
    )
)


print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=[
        0,
        1,
        2
    ],
    target_names=CLASS_NAMES,
    zero_division=0
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


report_file = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CMT1A Severity Classification\n"
    )

    f.write(
        "=============================\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}\n\n"
    )

    f.write(
        report
    )


print(
    "Saved:",
    report_file
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1,
        2
    ]
)


cm_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)


cm_dataframe = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_dataframe.to_csv(
    cm_file
)


print(
    "Saved:",
    cm_file
)


# ============================================================
# 27. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "CMT1A Severity Classification\n"
    "Test Confusion Matrix"
)


plt.colorbar()


plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)


plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)


plt.xlabel(
    "Predicted Severity"
)


plt.ylabel(
    "True Severity"
)


for i in range(
    NUM_CLASSES
):

    for j in range(
        NUM_CLASSES
    ):

        plt.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_plot_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    cm_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    cm_plot_file
)


# ============================================================
# 28. CONFIDENCE PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.bar(
    results_df["ID"],
    results_df["confidence"]
)


plt.ylim(
    0,
    1
)


plt.xlabel(
    "Patient ID"
)


plt.ylabel(
    "Prediction Confidence"
)


plt.title(
    "Prediction Confidence by Test Patient"
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


confidence_file = os.path.join(
    RESULTS_DIR,
    "test_prediction_confidence.png"
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


# ============================================================
# 29. ADAPTIVE FUSION ALPHA
# ============================================================

if results_df[
    "alpha"
].notna().any():

    plt.figure(
        figsize=(8, 5)
    )


    plt.bar(
        results_df["ID"],
        results_df["alpha"]
    )


    plt.xlabel(
        "Patient ID"
    )


    plt.ylabel(
        "Adaptive Fusion Alpha"
    )


    plt.title(
        "Patient-Specific Adaptive Fusion"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()


    alpha_file = os.path.join(
        RESULTS_DIR,
        "adaptive_fusion_alpha.png"
    )


    plt.savefig(
        alpha_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        alpha_file
    )


# ============================================================
# 30. GAIT VS CLINICAL REPRESENTATION
# ============================================================

if (
    results_df[
        "gait_embedding_norm"
    ].notna().any()
    and
    results_df[
        "clinical_embedding_norm"
    ].notna().any()
):

    x = np.arange(
        len(results_df)
    )


    width = 0.35


    plt.figure(
        figsize=(9, 5)
    )


    plt.bar(
        x - width / 2,
        results_df[
            "gait_embedding_norm"
        ],
        width,
        label="Gait Representation"
    )


    plt.bar(
        x + width / 2,
        results_df[
            "clinical_embedding_norm"
        ],
        width,
        label="Clinical Representation"
    )


    plt.xticks(
        x,
        results_df["ID"],
        rotation=45
    )


    plt.xlabel(
        "Patient ID"
    )


    plt.ylabel(
        "Latent Representation Norm"
    )


    plt.title(
        "Gait and Clinical Latent Representations"
    )


    plt.legend()


    plt.tight_layout()


    representation_file = os.path.join(
        RESULTS_DIR,
        "gait_clinical_representation.png"
    )


    plt.savefig(
        representation_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        representation_file
    )


# ============================================================
# 31. SAVE SUMMARY JSON
# ============================================================

summary = {

    "task":
        "CMT1A severity classification",

    "classes":
        CLASS_NAMES,

    "accuracy":
        float(accuracy),

    "balanced_accuracy":
        float(
            balanced_accuracy
        ),

    "num_test_patients":
        int(
            len(results_df)
        ),

    "test_class_distribution": {

        class_name:
            int(
                np.sum(
                    y_true
                    == class_index
                )
            )

        for class_index, class_name
        in enumerate(
            CLASS_NAMES
        )
    },

    "test_predictions": {

        class_name:
            int(
                np.sum(
                    y_pred
                    == class_index
                )
            )

        for class_index, class_name
        in enumerate(
            CLASS_NAMES
        )
    }
}


summary_file = os.path.join(
    RESULTS_DIR,
    "stage8_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print(
    "Saved:",
    summary_file
)


# ============================================================
# 32. DATASET LIMITATION
# ============================================================

print()
print("=" * 70)
print("IMPORTANT DATASET LIMITATION")
print("=" * 70)


severe_test_count = int(
    np.sum(
        y_true == 2
    )
)


if severe_test_count == 0:

    print(
        "The independent TEST set contains "
        "no Severe patient."
    )

    print(
        "Therefore, independent Severe-class "
        "test performance cannot be estimated."
    )


# ============================================================
# 33. FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 8 COMPLETE")
print("=" * 70)

print()
print(
    "Test predictions:",
    predictions_file
)

print(
    "Classification report:",
    report_file
)

print(
    "Confusion matrix:",
    cm_plot_file
)

print()
print(
    "The classification model was evaluated "
    "using the Stage 5C test split."
)

print("=" * 70)