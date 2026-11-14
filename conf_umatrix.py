
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\CMT_PROJECT")

INPUT_FILE = PROJECT_DIR / "results" / "test_predictions.csv"

OUTPUT_DIR = PROJECT_DIR / "results" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Mild", "Moderate", "Severe"]

CLASS_IDS = {
    0: "Mild",
    1: "Moderate",
    2: "Severe"
}

# ============================================================
# 2. LOAD ACTUAL TEST PREDICTIONS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("CMT-SCOPE AI - ACTUAL MODEL EVALUATION")
print("=" * 60)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# ============================================================
# 3. IDENTIFY ACTUAL AND PREDICTED COLUMNS
# ============================================================

possible_columns = [
    ("actual_severity", "predicted_severity"),
    ("actual_class", "predicted_class")
]

actual_col = None
predicted_col = None

for actual_name, predicted_name in possible_columns:

    if actual_name in df.columns and predicted_name in df.columns:

        actual_col = actual_name
        predicted_col = predicted_name
        break

if actual_col is None:

    raise ValueError(
        "Could not identify actual and predicted columns.\n"
        "Expected actual_severity and predicted_severity, "
        "or actual_class and predicted_class."
    )

df = df.dropna(
    subset=[actual_col, predicted_col]
).copy()

if df.empty:
    raise ValueError("No valid prediction records found.")

# ============================================================
# 4. NORMALIZE CLASS LABELS
# ============================================================

def normalize_label(value):

    if pd.isna(value):
        raise ValueError("Missing class label detected.")

    label = str(value).strip()

    try:
        number = float(label)

        if number.is_integer():

            class_id = int(number)

            if class_id in CLASS_IDS:
                return CLASS_IDS[class_id]

    except ValueError:
        pass

    label = label.title()

    if label in CLASS_NAMES:
        return label

    raise ValueError(
        f"Unknown severity label: {value}"
    )


actual = df[actual_col].map(normalize_label)

predicted = df[predicted_col].map(normalize_label)

# ============================================================
# 5. CHECK EVALUATION DATA
# ============================================================

total = len(actual)

if total == 0:
    raise ValueError("No evaluation records available.")

print("\nNumber of evaluation records:", total)

print("\nActual class distribution:")

print(
    actual.value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
)

print("\nPredicted class distribution:")

print(
    predicted.value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
)

# If the CSV contains patient identifiers, check duplicates.
# Multiple rows per patient may indicate window-level predictions.

patient_columns = [
    "patient_id",
    "patient",
    "subject_id",
    "subject",
    "patient_code"
]

for column in patient_columns:

    if column in df.columns:

        print(f"\nPatient identifier column detected: {column}")

        if df[column].duplicated().any():

            raise ValueError(
                "Multiple prediction rows exist for at least "
                "one patient. This may be window-level data. "
                "Generate one prediction per patient before "
                "reporting patient-level classification metrics."
            )

        break

# ============================================================
# 6. CALCULATE EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted
) * 100

balanced_accuracy = balanced_accuracy_score(
    actual,
    predicted
) * 100

macro_precision = precision_score(
    actual,
    predicted,
    labels=CLASS_NAMES,
    average="macro",
    zero_division=0
) * 100

macro_recall = recall_score(
    actual,
    predicted,
    labels=CLASS_NAMES,
    average="macro",
    zero_division=0
) * 100

macro_f1 = f1_score(
    actual,
    predicted,
    labels=CLASS_NAMES,
    average="macro",
    zero_division=0
) * 100

correct = int((actual == predicted).sum())

incorrect = total - correct

# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(f"Total records          : {total}")
print(f"Correct predictions    : {correct}")
print(f"Incorrect predictions  : {incorrect}")

print(f"Accuracy               : {accuracy:.2f}%")
print(f"Balanced Accuracy      : {balanced_accuracy:.2f}%")
print(f"Macro Precision        : {macro_precision:.2f}%")
print(f"Macro Recall           : {macro_recall:.2f}%")
print(f"Macro F1-score         : {macro_f1:.2f}%")

print("\nCLASSIFICATION REPORT")

report = classification_report(
    actual,
    predicted,
    labels=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

print(
    classification_report(
        actual,
        predicted,
        labels=CLASS_NAMES,
        zero_division=0
    )
)

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual,
    predicted,
    labels=CLASS_NAMES
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in CLASS_NAMES],
    columns=[f"Predicted {c}" for c in CLASS_NAMES]
)

print("\nCONFUSION MATRIX")
print(cm_df)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

# ============================================================
# 9. SAVE EVALUATED PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "Actual": actual,
    "Predicted": predicted,
    "Correct": actual == predicted

})

predictions_df.to_csv(
    OUTPUT_DIR / "evaluated_predictions.csv",
    index=False
)

# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "Total_Records": total,
    "Correct_Predictions": correct,
    "Incorrect_Predictions": incorrect,

    "Accuracy_Percent": round(accuracy, 2),

    "Balanced_Accuracy_Percent": round(
        balanced_accuracy, 2
    ),

    "Macro_Precision_Percent": round(
        macro_precision, 2
    ),

    "Macro_Recall_Percent": round(
        macro_recall, 2
    ),

    "Macro_F1_Percent": round(
        macro_f1, 2
    )

}])

metrics_df.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

# ============================================================
# 11. PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(cm, cmap="Blues")

fig.colorbar(
    image,
    ax=ax,
    label="Number of patients"
)

ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))

ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)

ax.set_xlabel("Predicted Severity")
ax.set_ylabel("Actual Severity")

ax.set_title(
    "CMT-SCOPE AI - Confusion Matrix"
)

threshold = cm.max() / 2 if cm.max() > 0 else 0

for i in range(len(CLASS_NAMES)):

    for j in range(len(CLASS_NAMES)):

        value = cm[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=15,
            color="white" if value > threshold else "black"
        )

fig.text(
    0.5,
    0.025,
    f"Accuracy: {accuracy:.2f}% | "
    f"Balanced Accuracy: {balanced_accuracy:.2f}%",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# 12. PLOT EVALUATION METRICS
# ============================================================

metric_names = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

metric_values = [
    accuracy,
    balanced_accuracy,
    macro_precision,
    macro_recall,
    macro_f1
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    metric_names,
    metric_values
)

ax.set_ylabel("Score (%)")

ax.set_title(
    "CMT-SCOPE AI - Evaluation Metrics"
)

ax.set_ylim(
    0,
    max(100, max(metric_values) + 12)
)

ax.tick_params(
    axis="x",
    rotation=15
)

for bar, value in zip(bars, metric_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "evaluation_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

# ============================================================
# 13. SAVE CLASSIFICATION REPORT
# ============================================================

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. confusion_matrix.csv")
print("2. confusion_matrix.png")
print("3. evaluated_predictions.csv")
print("4. evaluation_metrics.csv")
print("5. evaluation_metrics.png")
print("6. classification_report.csv")
