from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\CMT_PROJECT")
OUTPUT_DIR = PROJECT_ROOT / "results" / "simulated_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Mild", "Moderate", "Severe"]


matrix = np.array([
    [6, 1, 0],
    [1, 6, 0],
    [0, 0, 1]
])


metrics = {
    "Accuracy": 86.67,
    "Balanced Accuracy": 66.63,
    "Macro Precision": 45.00,
    "Macro Recall": 13.33,
    "Macro F1-score": 42.90
}


pd.DataFrame(
    matrix,
    index=[f"Actual {c}" for c in CLASSES],
    columns=[f"Predicted {c}" for c in CLASSES]
).to_csv(OUTPUT_DIR / "simulated_confusion_matrix.csv")

pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value (%)"]
).to_csv(OUTPUT_DIR / "simulated_metrics.csv", index=False)

# --------------------------------------------------
# 5. Plot confusion matrix
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax, label="Number of samples")

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES)
ax.set_yticklabels(CLASSES)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("CMT-SCOPE AI")

threshold = matrix.max() / 2

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(
            j,
            i,
            str(matrix[i, j]),
            ha="center",
            va="center",
            color="white" if matrix[i, j] > threshold else "black",
            fontsize=12
        )


fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(
    OUTPUT_DIR / "simulated_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


fig, ax = plt.subplots(figsize=(9, 5))

names = list(metrics.keys())
values = list(metrics.values())

bars = ax.bar(names, values)

ax.set_title("CMT-SCOPE AI ")
ax.set_ylabel("Value (%)")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=20)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )


fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(
    OUTPUT_DIR / "simulated_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

# --------------------------------------------------
# 7. Print results and show graphs
# --------------------------------------------------
print("\nCMT-SCOPE AI ")
print("-" * 45)

for name, value in metrics.items():
    print(f"{name}: {value:.2f}%")

print("\nFiles saved to:")
print(OUTPUT_DIR)


plt.show()
