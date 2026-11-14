import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from importlib.util import spec_from_file_location, module_from_spec


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("\n" + "=" * 70)
print("CMT-SCOPE AI - STAGE 7 TRAINING")
print("=" * 70)



print("\nLoading model...")

model_path = os.path.join(SCRIPT_DIR, "6_model.py")

model_spec = spec_from_file_location("model_6", model_path)

if model_spec is None or model_spec.loader is None:
    raise ImportError(f"Could not load model: {model_path}")

model_module = module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)

CMTSeverityModel = model_module.CMTSeverityModel
DEVICE = model_module.DEVICE
CLASS_NAMES = list(model_module.CLASS_NAMES)
NUM_CLASSES = int(model_module.NUM_CLASSES)

if NUM_CLASSES != 3:
    raise ValueError(
        f"Expected 3 severity classes, but found {NUM_CLASSES}."
    )

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError("CLASS_NAMES does not match NUM_CLASSES.")

print("Classes:", CLASS_NAMES)
print("Device:", DEVICE)



print("\nLoading patient dataset...")

dataset_path = os.path.join(
    SCRIPT_DIR,
    "5c_patient_dataset.py"
)

dataset_spec = spec_from_file_location(
    "dataset_5c",
    dataset_path
)

if dataset_spec is None or dataset_spec.loader is None:
    raise ImportError(f"Could not load dataset: {dataset_path}")

dataset_module = module_from_spec(dataset_spec)
dataset_spec.loader.exec_module(dataset_module)

train_dataset = dataset_module.train_dataset
val_dataset = dataset_module.val_dataset

if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty.")

if len(val_dataset) == 0:
    raise ValueError("Validation dataset is empty.")

print("Training patients:", len(train_dataset))
print("Validation patients:", len(val_dataset))




MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PRETRAINED_PATH = os.path.join(
    PROJECT_ROOT,
    "processed",
    "supplementary_gait",
    "best_gait_encoder.pth"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_cmt_severity_model.pth"
)



#  TRAINING SETTINGS


EPOCHS = 50

# Fine-tune pretrained encoder slowly.
ENCODER_LR = 0.00001

# Train fusion and classification layers at a higher rate.
HEAD_LR = 0.0001

WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
PATIENCE = 10

# Batch size 1 is used because patients have different
# numbers of gait windows.
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)



#  CHECK TRAINING LABELS


train_labels = []

for i in range(len(train_dataset)):
    sample = train_dataset[i]

    # Expected dataset format:
    # sample[0] = patient ID
    # sample[1] = gait windows
    # sample[2] = clinical features
    # sample[3] = encoded severity label

    label = int(torch.as_tensor(sample[3]).item())
    train_labels.append(label)

train_labels = np.asarray(train_labels, dtype=np.int64)

if not np.isin(train_labels, np.arange(NUM_CLASSES)).all():
    raise ValueError(
        "Labels must be encoded as 0, 1, and 2. "
        f"Found: {np.unique(train_labels).tolist()}"
    )

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_counts[i]}")

if np.any(class_counts == 0):
    print(
        "\nWARNING: At least one class is absent from training. "
        "The model cannot learn that class reliably."
    )




print("\nCreating severity model...")

model = CMTSeverityModel().to(DEVICE)

if not hasattr(model, "gait_encoder"):
    raise AttributeError(
        "6_model.py must expose model.gait_encoder "
        "to load the pretrained encoder."
    )



# LOAD SELF-SUPERVISED PRETRAINED ENCODER


print("\nLoading pretrained gait encoder...")

if not os.path.isfile(PRETRAINED_PATH):
    raise FileNotFoundError(
        "\nPretrained encoder not found:\n"
        f"{PRETRAINED_PATH}\n\n"
        "Run 5a_pretrain_gait_encoder.py first."
    )

try:
    pretrained_checkpoint = torch.load(
        PRETRAINED_PATH,
        map_location=DEVICE,
        weights_only=False
    )
except TypeError:
    # Compatibility with older PyTorch versions.
    pretrained_checkpoint = torch.load(
        PRETRAINED_PATH,
        map_location=DEVICE
    )

if (
    isinstance(pretrained_checkpoint, dict)
    and "model_state_dict" in pretrained_checkpoint
):
    encoder_state = pretrained_checkpoint["model_state_dict"]
elif isinstance(pretrained_checkpoint, dict):
    # Supports checkpoints containing the state dict directly.
    encoder_state = pretrained_checkpoint
else:
    raise ValueError("Unrecognized pretrained checkpoint format.")

try:
    load_result = model.gait_encoder.load_state_dict(
        encoder_state,
        strict=True
    )
except RuntimeError as exc:
    raise RuntimeError(
        "\nThe pretrained encoder architecture does not match "
        "model.gait_encoder in 6_model.py.\n"
        "Do not bypass this with strict=False. Check the encoder "
        "definition and tensor dimensions.\n\n"
        f"Original error:\n{exc}"
    ) from exc

print("Pretrained encoder loaded successfully.")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)



# CLASS WEIGHTS


# Calculated only from training patients.
# Severe has only one training patient, so its weight will be high.

total_patients = float(len(train_labels))

class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

for i in range(NUM_CLASSES):
    if class_counts[i] > 0:
        class_weights[i] = (
            total_patients /
            (NUM_CLASSES * float(class_counts[i]))
        )

print("\nClass weights:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {class_weights[i]:.4f}")

weight_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)



#  LOSS FUNCTION


# Use reduction='none' because a batch contains one patient.
# Multiplying each patient's loss by its class weight ensures
# the class weight is not cancelled by weighted-mean reduction.

criterion = nn.CrossEntropyLoss(reduction="none")


def calculate_loss(logits, target):
    """
    logits: [1, NUM_CLASSES]
    target: scalar class index
    """
    per_patient_loss = criterion(
        logits,
        target.reshape(1)
    )

    patient_weight = weight_tensor[target.reshape(())]

    return (per_patient_loss * patient_weight).mean()



# OPTIMIZER WITH SEPARATE LEARNING RATES


encoder_parameters = list(model.gait_encoder.parameters())

encoder_parameter_ids = {
    id(parameter) for parameter in encoder_parameters
}

head_parameters = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in encoder_parameter_ids
]

if not encoder_parameters:
    raise ValueError("The gait encoder has no parameters.")

if not head_parameters:
    raise ValueError(
        "No non-encoder parameters found for the classifier."
    )

optimizer = torch.optim.AdamW(
    [
        {
            "params": encoder_parameters,
            "lr": ENCODER_LR
        },
        {
            "params": head_parameters,
            "lr": HEAD_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

print("\nOptimizer configured.")
print("Encoder learning rate:", ENCODER_LR)
print("Other layers learning rate:", HEAD_LR)


# PREPARE MODEL INPUTS AND LOGITS 

def prepare_batch(batch):
    gait = batch[1].to(
        DEVICE,
        dtype=torch.float32
    )

    clinical = batch[2].to(
        DEVICE,
        dtype=torch.float32
    )

    target = torch.as_tensor(
        batch[3],
        dtype=torch.long,
        device=DEVICE
    ).reshape(-1)

    # Remove only the outer DataLoader batch dimension.
    # Do not squeeze the gait-window dimension.
    if gait.ndim >= 2 and gait.shape[0] == 1:
        gait = gait.squeeze(0)

    if clinical.ndim >= 2 and clinical.shape[0] == 1:
        clinical = clinical.squeeze(0)

    if target.numel() != 1:
        raise ValueError(
            f"Expected one patient per batch, got {target.numel()} labels."
        )

    target = target.reshape(())

    return gait, clinical, target


def get_logits(output):
    if not isinstance(output, dict) or "logits" not in output:
        raise ValueError(
            'Model output must be a dictionary containing "logits".'
        )

    logits = output["logits"]

    if not torch.is_tensor(logits):
        raise TypeError("Model output['logits'] must be a tensor.")

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.shape != (1, NUM_CLASSES):
        raise ValueError(
            "Unexpected logits shape: "
            f"{tuple(logits.shape)}; expected (1, {NUM_CLASSES})."
        )

    return logits



# . TRAIN ONE EPOCH


def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        gait, clinical, target = prepare_batch(batch)

        optimizer.zero_grad(set_to_none=True)

        output = model(gait, clinical)
        logits = get_logits(output)

        loss = calculate_loss(logits, target)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss encountered."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        total_loss += loss.item()

        prediction = int(logits.argmax(dim=1).item())
        actual = int(target.item())

        correct += int(prediction == actual)
        total += 1

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


# ============================================================
# 14. VALIDATION
# ============================================================

def validate():
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():
        for batch in val_loader:
            gait, clinical, target = prepare_batch(batch)

            output = model(gait, clinical)
            logits = get_logits(output)

            loss = calculate_loss(logits, target)
            total_loss += loss.item()

            probability = torch.softmax(logits, dim=1)
            prediction = int(probability.argmax(dim=1).item())

            predictions.append(prediction)
            targets.append(int(target.item()))
            probabilities.append(
                probability.squeeze(0).cpu().numpy()
            )

    if len(targets) == 0:
        return 0.0, 0.0, [], [], []

    accuracy = float(
        np.mean(np.asarray(predictions) == np.asarray(targets))
    )

    average_loss = total_loss / len(targets)

    return (
        average_loss,
        accuracy,
        predictions,
        targets,
        probabilities
    )


# ============================================================
# 15. TRAINING LOOP AND SAFE BEST-CHECKPOINT SAVING
# ============================================================

best_val_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = train_one_epoch()

    (
        val_loss,
        val_accuracy,
        predictions,
        targets,
        probabilities
    ) = validate()

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Copy weights so later updates cannot modify the saved best state.
    if np.isfinite(val_loss) and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "class_weights": class_weights.tolist(),
                "pretrained_encoder": PRETRAINED_PATH,
                "seed": SEED
            },
            BEST_MODEL_PATH
        )

        print("  --> Best validation-loss checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

if best_state is None or not os.path.isfile(BEST_MODEL_PATH):
    raise RuntimeError(
        "No valid checkpoint was saved. Check the loss and dataset."
    )


# ============================================================
# 16. RESTORE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("RESTORING BEST MODEL")
print("=" * 70)

try:
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

print("Best epoch:", checkpoint["epoch"])
print("Best validation loss:", f"{checkpoint['val_loss']:.6f}")
print("Best validation accuracy:", f"{checkpoint['val_accuracy']:.4f}")
print("Saved to:", BEST_MODEL_PATH)


# ============================================================
# 17. FINAL VALIDATION SUMMARY
# ============================================================

(
    final_val_loss,
    final_val_accuracy,
    final_predictions,
    final_targets,
    final_probabilities
) = validate()

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

for actual, predicted in zip(final_targets, final_predictions):
    confusion_matrix[actual, predicted] += 1

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(f"Validation loss: {final_val_loss:.6f}")
print(f"Validation accuracy: {final_val_accuracy:.4f}")
print(f"Correct patients: {sum(p == t for p, t in zip(final_predictions, final_targets))}/{len(final_targets)}")

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("             " + " ".join(f"{name:>10}" for name in CLASS_NAMES))

for i, name in enumerate(CLASS_NAMES):
    row = " ".join(
        f"{int(value):>10}"
        for value in confusion_matrix[i]
    )
    print(f"{name:<12}{row}")

print("\nPer-class recall:")

for i, name in enumerate(CLASS_NAMES):
    actual_count = int(confusion_matrix[i].sum())

    if actual_count > 0:
        recall = confusion_matrix[i, i] / actual_count
        print(f"{name}: {recall:.4f} ({confusion_matrix[i, i]}/{actual_count})")
    else:
        print(f"{name}: Not present in validation set")

print("\nValidation predictions:")

for index, (predicted, actual) in enumerate(
    zip(final_predictions, final_targets),
    start=1
):
    print(
        f"Patient {index}: "
        f"Actual={CLASS_NAMES[actual]} | "
        f"Predicted={CLASS_NAMES[predicted]}"
    )

print("\n" + "=" * 70)
print("STAGE 7 COMPLETE")
print("=" * 70)
print("Best model:", BEST_MODEL_PATH)
print("Class mapping used by the model:")

for index, name in enumerate(CLASS_NAMES):
    print(f"{index} = {name}")

print(
    "\nReminder: validation results are based on only "
    f"{len(val_dataset)} patients. They are not test-set results."
)
