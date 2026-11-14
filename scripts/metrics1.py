from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Replace these with your actual test labels and model predictions
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 1, 1, 1, 2, 0]

# Weighted evaluation metrics
precision = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)

f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred, zero_division=0
))