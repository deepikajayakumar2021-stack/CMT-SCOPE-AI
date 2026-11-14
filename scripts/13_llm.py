import requests
import pandas as pd
import numpy as np

from pathlib import Path

from importlib.util import (
    spec_from_file_location,
    module_from_spec
)

import sys


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\CMT_PROJECT"
)

SCRIPT_DIR = (
    PROJECT_ROOT /
    "scripts"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

sys.path.insert(
    0,
    str(SCRIPT_DIR)
)


# ============================================================
# 2. PATIENT
# ============================================================
#
# ROC-016 is a current held-out test patient.
#
# Current test patients:
# ROC-016
# ROC-034
# ROC-029
# ROC-041
#
# ROC-016 also has Stage 9 explainability files.
# ============================================================

PATIENT_ID = "ROC-016"


# ============================================================
# 3. LOAD RAG MODULE
# ============================================================

rag_spec = spec_from_file_location(
    "rag_module",
    SCRIPT_DIR / "12_rag.py"
)

rag_module = module_from_spec(
    rag_spec
)

rag_spec.loader.exec_module(
    rag_module
)


# ============================================================
# 4. OLLAMA SETTINGS
# ============================================================

OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

MODEL_NAME = "llama3.2"


# ============================================================
# 5. FILE PATHS
# ============================================================

test_predictions_file = (
    RESULTS_DIR /
    "test_predictions.csv"
)

shap_file = (
    RESULTS_DIR /
    f"{PATIENT_ID}_clinical_shap.csv"
)

gait_file = (
    RESULTS_DIR /
    f"{PATIENT_ID}_gait_temporal_explanation.csv"
)

output_file = (
    RESULTS_DIR /
    f"{PATIENT_ID}_llm_explanation.txt"
)


# ============================================================
# 6. CHECK REQUIRED FILES
# ============================================================

required_files = [

    test_predictions_file,

    shap_file,

    gait_file

]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(

            f"\nRequired file not found:\n"
            f"{file_path}\n"

        )


# ============================================================
# 7. LOAD CURRENT CLASSIFICATION RESULTS
# ============================================================

test_df = pd.read_csv(
    test_predictions_file
)


# ------------------------------------------------------------
# Current classification column check
# ------------------------------------------------------------

required_prediction_columns = [

    "ID",

    "true_severity",

    "predicted_severity",

    "confidence",

    "prob_mild",

    "prob_moderate",

    "prob_severe",

    "alpha"

]


missing_prediction_columns = [

    column

    for column in required_prediction_columns

    if column not in test_df.columns

]


if missing_prediction_columns:

    raise ValueError(

        "\nCurrent classification columns are missing:\n"

        +
        "\n".join(
            missing_prediction_columns
        )

    )


# ============================================================
# 8. FIND PATIENT
# ============================================================

patient_rows = test_df[
    test_df["ID"].astype(str) ==
    PATIENT_ID
]


if patient_rows.empty:

    raise ValueError(

        f"\n{PATIENT_ID} was not found in "
        "the current test_predictions.csv.\n\n"

        "Available test patients:\n"

        +
        "\n".join(
            test_df["ID"].astype(str).tolist()
        )

    )


patient_row = patient_rows.iloc[0]


# ============================================================
# 9. CURRENT MODEL OUTPUT
# ============================================================

true_severity = str(
    patient_row["true_severity"]
)

predicted_severity = str(
    patient_row["predicted_severity"]
)

confidence = float(
    patient_row["confidence"]
)

prob_mild = float(
    patient_row["prob_mild"]
)

prob_moderate = float(
    patient_row["prob_moderate"]
)

prob_severe = float(
    patient_row["prob_severe"]
)

alpha = float(
    patient_row["alpha"]
)


# ============================================================
# 10. LOAD SHAP RESULTS
# ============================================================

shap_df = pd.read_csv(
    shap_file
)


required_shap_columns = [

    "Feature",

    "SHAP_Value"

]


missing_shap_columns = [

    column

    for column in required_shap_columns

    if column not in shap_df.columns

]


if missing_shap_columns:

    raise ValueError(

        "\nSHAP columns are missing:\n"

        +
        "\n".join(
            missing_shap_columns
        )

    )


shap_results = []


for _, row in shap_df.iterrows():

    shap_results.append({

        "Feature":
            str(
                row["Feature"]
            ),

        "SHAP_Value":
            float(
                row["SHAP_Value"]
            )

    })


# ============================================================
# 11. LOAD GAIT TEMPORAL RESULTS
# ============================================================

gait_df = pd.read_csv(
    gait_file
)


if "Segment" not in gait_df.columns:

    raise ValueError(

        "\nGait temporal explanation file "
        "does not contain 'Segment'."

    )


# ------------------------------------------------------------
# Find the numerical prediction-change column.
#
# This makes the code compatible with the current Stage 9
# explanation file even if the exact column name differs.
# ------------------------------------------------------------

gait_change_candidates = [

    "Prediction_Change",

    "Prediction_Change_Probability",

    "Prediction_Change_CMT_FOM",

    "Absolute_Change",

    "Probability_Change"

]


gait_change_column = None


for column in gait_change_candidates:

    if column in gait_df.columns:

        gait_change_column = column

        break


# ------------------------------------------------------------
# If no known name exists, find the first numerical column
# other than Segment.
# ------------------------------------------------------------

if gait_change_column is None:

    numerical_columns = [

        column

        for column in gait_df.columns

        if column != "Segment"

        and pd.api.types.is_numeric_dtype(
            gait_df[column]
        )

    ]

    if numerical_columns:

        gait_change_column = numerical_columns[0]


if gait_change_column is None:

    raise ValueError(

        "\nCould not identify the temporal "
        "prediction-change column in:\n"

        f"{gait_file}"

    )


gait_results = []


for _, row in gait_df.iterrows():

    gait_results.append({

        "Segment":
            int(
                row["Segment"]
            ),

        "Prediction_Change":
            float(
                row[gait_change_column]
            )

    })


# ============================================================
# 12. IDENTIFY MOST INFLUENTIAL SHAP FEATURE
# ============================================================

if shap_results:

    largest_shap = max(

        shap_results,

        key=lambda item:
            abs(
                item["SHAP_Value"]
            )

    )

else:

    largest_shap = None


# ============================================================
# 13. IDENTIFY MOST INFLUENTIAL GAIT SEGMENT
# ============================================================

if gait_results:

    largest_gait = max(

        gait_results,

        key=lambda item:
            abs(
                item["Prediction_Change"]
            )

    )

else:

    largest_gait = None


# ============================================================
# 14. RETRIEVE LITERATURE
# ============================================================

query = (

    "Charcot-Marie-Tooth disease CMT1A "

    "gait pattern disease severity "

    "clinical assessment "

    "CMTES CMT-FOM "

    "muscle weakness sensory impairment "

    "walking function"

)


retrieved_evidence = rag_module.retrieve(

    query,

    top_k=5

)


# ============================================================
# 15. BUILD SHAP TEXT
# ============================================================

shap_text = ""


for item in shap_results:

    shap_text += (

        f"{item['Feature']}: "

        f"SHAP={item['SHAP_Value']:.6f}\n"

    )


# ============================================================
# 16. BUILD GAIT TEXT
# ============================================================

gait_text = ""


for item in gait_results:

    gait_text += (

        f"Segment {item['Segment']}: "

        f"Prediction_Change="
        f"{item['Prediction_Change']:.6f}\n"

    )


# ============================================================
# 17. BUILD LITERATURE TEXT
# ============================================================

evidence_text = ""


for item in retrieved_evidence:

    evidence_text += (

        "\nSOURCE FILE: "

        +
        str(
            item["source"]
        )

        +

        "\nSIMILARITY: "

        +
        f"{item['score']:.4f}"

        +

        "\nTEXT:\n"

        +
        item["text"]

        +

        "\n"

    )


# ============================================================
# 18. PREPARE MODEL INTERPRETATION
# ============================================================

if largest_shap is not None:

    largest_shap_text = (

        f"Feature: {largest_shap['Feature']}; "

        f"SHAP value: "
        f"{largest_shap['SHAP_Value']:.6f}"

    )

else:

    largest_shap_text = (
        "No SHAP result available."
    )


if largest_gait is not None:

    largest_gait_text = (

        f"Segment: {largest_gait['Segment']}; "

        f"Prediction change: "
        f"{largest_gait['Prediction_Change']:.6f}"

    )

else:

    largest_gait_text = (
        "No gait temporal result available."
    )


# ============================================================
# 19. STRICT CLASSIFICATION LLM PROMPT
# ============================================================

prompt = f"""
You are an evidence-grounded explanation assistant
for a research prototype involving Charcot-Marie-Tooth
disease type 1A (CMT1A).

The system performs three-class severity classification
using gait and clinical representations.

IMPORTANT:

The LLM is NOT the classifier.

The LLM must only explain the supplied model output,
supplied explainability results, and retrieved literature.

============================================================
PATIENT
============================================================

Patient ID:
{PATIENT_ID}

============================================================
MODEL PREDICTION
============================================================

True severity label in the evaluation dataset:
{true_severity}

Predicted severity class:
{predicted_severity}

Prediction confidence:
{confidence:.6f}

Class probabilities:

Mild:
{prob_mild:.6f}

Moderate:
{prob_moderate:.6f}

Severe:
{prob_severe:.6f}

Adaptive fusion alpha:
{alpha:.6f}

IMPORTANT INTERPRETATION:

Adaptive alpha is a learned model fusion coefficient.

It is NOT:

- clinical importance,
- disease severity,
- probability of disease,
- percentage of gait contribution,
- percentage of clinical contribution,
- or a causal measure.

============================================================
STRICT CLASSIFICATION RULES
============================================================

1. Report the supplied predicted class exactly.

2. Report the supplied confidence only as model confidence.

3. Do not convert confidence into clinical probability.

4. Do not claim that model confidence represents
   probability of disease.

5. Do not invent severity thresholds.

6. Do not invent a CMT-FOM numerical score.

7. Do not convert the classification into a continuous
   clinical score.

8. Do not diagnose the patient.

9. Do not claim that the model is clinically validated.

10. Do not provide treatment recommendations.

============================================================
EXPLAINABILITY RULES
============================================================

The supplied SHAP results describe local model behavior
for the clinical features.

SHAP values must NOT be described as:

- causal effects,
- biomarkers,
- clinical importance,
- physiological mechanisms.

The supplied temporal occlusion results show how the model
prediction changed when portions of the gait representation
were occluded.

Temporal occlusion must NOT be described as:

- a clinical biomarker,
- a specific anatomical function,
- a causal physiological finding.

Largest supplied SHAP result:

{largest_shap_text}

Largest tested gait temporal change:

{largest_gait_text}

============================================================
LITERATURE RULES
============================================================

Use ONLY the supplied retrieved literature.

Every literature-based statement must identify the
SOURCE FILE from which it came.

Do not invent:

- authors,
- publication years,
- study results,
- clinical thresholds,
- treatment recommendations,
- citations.

Clearly separate:

MODEL-DERIVED INFORMATION

from

LITERATURE-DERIVED INFORMATION.

============================================================
RETRIEVED LITERATURE
============================================================

{evidence_text}

============================================================
SHAP RESULTS
============================================================

{shap_text}

============================================================
GAIT TEMPORAL OCCLUSION
============================================================

{gait_text}

============================================================
WRITE EXACTLY THESE SECTIONS
============================================================

1. Model Prediction

State the patient ID, predicted severity class,
and model confidence.

Explain that this is the output of the research model.

Do not reinterpret the confidence as a clinical probability.

2. Model Explanation

Explain the supplied SHAP results and gait temporal
occlusion results as local model-level explanations.

Mention the largest supplied SHAP contribution if useful.

Mention the largest tested gait temporal change if useful.

Do not claim causality.

3. Relevant Literature Evidence

Summarize only literature-supported information.

For every literature-based statement, identify
the source filename.

4. Evidence-Based Clinical Support

Explain how model-derived findings and literature
evidence can be considered together in a research setting.

Do not diagnose.

Do not recommend treatment.

Do not claim clinical validation.

5. Limitations

Mention:

- the small held-out test set,
- the class imbalance in the available labelled dataset,
- the absence of a severe case in the held-out test set,
- the research-prototype nature of the system,
- the local nature of the explainability analysis,
- the limited temporal occlusion analysis,
- the fact that literature retrieval is based only on
  the supplied documents,
- and the need for further external validation before
  any clinical use.

End with exactly:

This system is a research decision-support prototype and
is not a medical diagnosis.

============================================================
FINAL RULE
============================================================

Do not invent numerical values.

Use only the supplied model results.

Do not turn model outputs into medical conclusions.
"""


# ============================================================
# 20. DISPLAY STAGE INFORMATION
# ============================================================

print()

print("=" * 70)

print(
    "STAGE 13: EVIDENCE-GROUNDED LLM"
)

print("=" * 70)

print()

print(
    "Patient:",
    PATIENT_ID
)

print(
    "True severity:",
    true_severity
)

print(
    "Predicted severity:",
    predicted_severity
)

print(
    "Model confidence:",
    round(
        confidence,
        6
    )
)

print(
    "Adaptive Alpha:",
    round(
        alpha,
        6
    )
)

print()

print(
    "Retrieved literature:",
    len(
        retrieved_evidence
    ),
    "chunks"
)

print()

print(
    "Connecting to Ollama..."
)


# ============================================================
# 21. SEND TO OLLAMA
# ============================================================

try:

    response = requests.post(

        OLLAMA_URL,

        json={

            "model":
                MODEL_NAME,

            "prompt":
                prompt,

            "stream":
                False,

            "options": {

                "temperature":
                    0.1

            }

        },

        timeout=300

    )

    response.raise_for_status()


except requests.exceptions.ConnectionError:

    raise RuntimeError(

        "\nOllama is not running or "
        "cannot be reached.\n\n"

        "Check with:\n"

        "ollama list\n"

        "\n"

        "Do NOT start another ollama serve "
        "if Ollama is already running."

    )


except requests.exceptions.Timeout:

    raise RuntimeError(

        "\nOllama took too long to respond.\n\n"

        "The model may require more time on "
        "the available hardware.\n"

    )


# ============================================================
# 22. EXTRACT RESPONSE
# ============================================================

response_json = response.json()


if "response" not in response_json:

    raise RuntimeError(

        "Ollama response did not contain "
        "the expected 'response' field."

    )


explanation = response_json[
    "response"
]


# ============================================================
# 23. BASIC SAFETY CHECKS
# ============================================================

if predicted_severity not in explanation:

    print()

    print(
        "WARNING: LLM did not reproduce "
        "the predicted severity class."
    )


required_sections = [

    "1. Model Prediction",

    "2. Model Explanation",

    "3. Relevant Literature Evidence",

    "4. Evidence-Based Clinical Support",

    "5. Limitations"

]


missing_sections = [

    section

    for section in required_sections

    if section not in explanation

]


if missing_sections:

    print()

    print(
        "WARNING: Some expected sections "
        "were not generated:"
    )

    for section in missing_sections:

        print(
            "-",
            section
        )


# ============================================================
# 24. SAVE EXPLANATION
# ============================================================

with open(

    output_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        explanation
    )


# ============================================================
# 25. DISPLAY
# ============================================================

print()

print("=" * 70)

print(
    "GENERATED EVIDENCE-GROUNDED EXPLANATION"
)

print("=" * 70)

print()

print(
    explanation
)

print()

print("=" * 70)

print(
    "STAGE 13 COMPLETE"
)

print("=" * 70)

print()

print(
    "Saved:",
    output_file
)