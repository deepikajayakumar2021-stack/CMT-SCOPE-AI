import os
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.optim import Adam

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = r"D:\CMT_PROJECT"

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

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

EPOCHS = 150

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

D_MODEL = 32

DROPOUT = 0.20

NUM_CLASSES = 3

PATIENCE = 20

GRAD_CLIP = 1.0


# ------------------------------------------------------------
# Temporal Transformer settings
#
# These match the architecture used by the main model.
# ------------------------------------------------------------

NUM_HEADS = 4

NUM_TRANSFORMER_LAYERS = 2

TRANSFORMER_FF_DIM = 128

MAX_WINDOWS = 512


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 70)

print("CMT-SCOPE AI")

print("GAIT-ONLY CLASSIFICATION ABLATION")

print("=" * 70)

print()

print("Device:", DEVICE)

print()

print("Architecture:")

print("Gait")

print("  ↓")

print("InceptionTime")

print("  ↓")

print("Temporal Transformer")

print("  ↓")

print("Window Attention")

print("  ↓")

print("Gait Representation")

print("  ↓")

print("Classification Head")

print()


# ============================================================
# 5. SEVERITY LABELS
# ============================================================

SEVERITY_TO_LABEL = {

    "mild": 0,

    "moderate": 1,

    "severe": 2

}


LABEL_TO_SEVERITY = {

    0: "Mild",

    1: "Moderate",

    2: "Severe"

}


# ============================================================
# 6. INCEPTION BLOCK
#
# IMPORTANT:
#   Uses BatchNorm1d to match the main model.
# ============================================================

class InceptionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        num_filters=32
    ):

        super().__init__()

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = nn.Conv1d(
            in_channels,
            num_filters,
            kernel_size=1,
            bias=False
        )

        # ----------------------------------------------------
        # Multi-scale convolution branches
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size=9,
            padding=4,
            bias=False
        )

        self.conv2 = nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size=19,
            padding=9,
            bias=False
        )

        self.conv3 = nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size=39,
            padding=19,
            bias=False
        )

        # ----------------------------------------------------
        # Max-pooling branch
        # ----------------------------------------------------

        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool_conv = nn.Conv1d(
            in_channels,
            num_filters,
            kernel_size=1,
            bias=False
        )

        # ----------------------------------------------------
        # Four branches
        #
        # 32 × 4 = 128 channels
        # ----------------------------------------------------

        output_channels = num_filters * 4

        self.bn = nn.BatchNorm1d(
            output_channels
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        self.residual = nn.Conv1d(
            in_channels,
            output_channels,
            kernel_size=1,
            bias=False
        )

        self.residual_bn = nn.BatchNorm1d(
            output_channels
        )

        self.relu = nn.ReLU()


    def forward(self, x):

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        bottleneck = self.bottleneck(
            x
        )

        # ----------------------------------------------------
        # Multi-scale branches
        # ----------------------------------------------------

        branch1 = self.conv1(
            bottleneck
        )

        branch2 = self.conv2(
            bottleneck
        )

        branch3 = self.conv3(
            bottleneck
        )

        branch4 = self.pool_conv(
            self.maxpool(x)
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        output = torch.cat(
            [
                branch1,
                branch2,
                branch3,
                branch4
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        output = self.bn(
            output
        )

        # ----------------------------------------------------
        # Residual
        # ----------------------------------------------------

        residual = self.residual(
            x
        )

        residual = self.residual_bn(
            residual
        )

        # ----------------------------------------------------
        # Residual addition
        # ----------------------------------------------------

        output = output + residual

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------

        output = self.relu(
            output
        )

        return output


# ============================================================
# 7. INCEPTION TIME
# ============================================================

class InceptionTime(nn.Module):

    def __init__(
        self,
        input_channels=3,
        num_filters=32,
        embedding_size=32
    ):

        super().__init__()

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        self.block1 = InceptionBlock(
            input_channels,
            num_filters
        )

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        self.block2 = InceptionBlock(
            num_filters * 4,
            num_filters
        )

        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        self.block3 = InceptionBlock(
            num_filters * 4,
            num_filters
        )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool1d(
            1
        )

        # ----------------------------------------------------
        # Window embedding
        # ----------------------------------------------------

        self.embedding = nn.Linear(
            num_filters * 4,
            embedding_size
        )


    def forward(self, x):

        # x:
        # batch/windows × 3 × 256

        x = self.block1(
            x
        )

        x = self.block2(
            x
        )

        x = self.block3(
            x
        )

        x = self.global_pool(
            x
        )

        x = x.squeeze(
            -1
        )

        x = self.embedding(
            x
        )

        return x


# ============================================================
# 8. TEMPORAL TRANSFORMER
#
# Same conceptual temporal module as the main model.
#
# Input:
#   [batch, windows, D_MODEL]
#
# Output:
#   [batch, windows, D_MODEL]
# ============================================================

class TemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_TRANSFORMER_LAYERS,
        ff_dim=TRANSFORMER_FF_DIM,
        max_windows=MAX_WINDOWS,
        dropout=DROPOUT
    ):

        super().__init__()

        # ----------------------------------------------------
        # Learnable positional embedding
        # ----------------------------------------------------

        self.positional_embedding = nn.Parameter(

            torch.zeros(
                1,
                max_windows,
                d_model
            )

        )

        # ----------------------------------------------------
        # Transformer encoder layer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=num_heads,

            dim_feedforward=ff_dim,

            dropout=dropout,

            batch_first=True,

            activation="gelu"

        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        # ----------------------------------------------------
        # x:
        # [batch, windows, D_MODEL]
        # ----------------------------------------------------

        batch_size = x.size(0)

        num_windows = x.size(1)

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if num_windows > MAX_WINDOWS:

            # Use the first MAX_WINDOWS windows.
            #
            # The main model also has a maximum window limit.

            x = x[
                :,
                :MAX_WINDOWS,
                :
            ]

            num_windows = MAX_WINDOWS

        # ----------------------------------------------------
        # Positional information
        # ----------------------------------------------------

        position = self.positional_embedding[
            :,
            :num_windows,
            :
        ]

        x = x + position

        x = self.dropout(
            x
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.transformer(
            x
        )

        return x


# ============================================================
# 9. PATIENT GAIT ENCODER
#
# IMPORTANT:
#   This now matches the main gait flow:
#
#   InceptionTime
#       ↓
#   Temporal Transformer
#       ↓
#   Window Attention
#       ↓
#   Patient gait representation
# ============================================================

class PatientGaitEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # InceptionTime
        # ----------------------------------------------------

        self.inception = InceptionTime(

            input_channels=3,

            num_filters=32,

            embedding_size=D_MODEL

        )

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        self.temporal_transformer = TemporalTransformer(

            d_model=D_MODEL,

            num_heads=NUM_HEADS,

            num_layers=NUM_TRANSFORMER_LAYERS,

            ff_dim=TRANSFORMER_FF_DIM,

            max_windows=MAX_WINDOWS,

            dropout=DROPOUT

        )

        # ----------------------------------------------------
        # Window Attention
        # ----------------------------------------------------

        self.window_attention = nn.Sequential(

            nn.Linear(
                D_MODEL,
                16
            ),

            nn.Tanh(),

            nn.Linear(
                16,
                1
            )

        )


    def forward(self, gait):

        # ----------------------------------------------------
        # Input
        #
        # windows × 256 × 3
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Convert:
        #
        # windows × 256 × 3
        #
        # to:
        #
        # windows × 3 × 256
        # ----------------------------------------------------

        gait = gait.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------------
        # InceptionTime
        #
        # Each gait window becomes a D_MODEL embedding.
        # ----------------------------------------------------

        window_embeddings = self.inception(
            gait
        )

        # ----------------------------------------------------
        # Add batch dimension for Transformer
        #
        # windows × D_MODEL
        #
        # →
        #
        # 1 × windows × D_MODEL
        # ----------------------------------------------------

        sequence = window_embeddings.unsqueeze(
            0
        )

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        sequence = self.temporal_transformer(
            sequence
        )

        # ----------------------------------------------------
        # Remove batch dimension
        #
        # 1 × windows × D_MODEL
        #
        # →
        #
        # windows × D_MODEL
        # ----------------------------------------------------

        window_embeddings = sequence.squeeze(
            0
        )

        # ----------------------------------------------------
        # Window attention
        # ----------------------------------------------------

        scores = self.window_attention(
            window_embeddings
        ).squeeze(
            -1
        )

        # ----------------------------------------------------
        # Normalize attention across windows
        # ----------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=0
        )

        # ----------------------------------------------------
        # Weighted patient-level representation
        # ----------------------------------------------------

        patient_embedding = torch.sum(

            window_embeddings *

            weights.unsqueeze(
                -1
            ),

            dim=0

        )

        return patient_embedding


# ============================================================
# 10. GAIT-ONLY CLASSIFICATION MODEL
# ============================================================

class GaitOnlyClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Gait encoder
        # ----------------------------------------------------

        self.gait_encoder = PatientGaitEncoder()

        # ----------------------------------------------------
        # Classification head
        #
        # No clinical branch.
        # No AdaptiveClinicalFusion.
        # No Dual-Axis Representation.
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                D_MODEL,
                16
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                16,
                NUM_CLASSES
            )

        )


    def forward(self, gait):

        # ----------------------------------------------------
        # Gait representation
        # ----------------------------------------------------

        gait_embedding = self.gait_encoder(
            gait
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            gait_embedding
        )

        return {

            "gait_embedding":
                gait_embedding,

            "logits":
                logits

        }


# ============================================================
# 11. LOAD MANIFEST
# ============================================================

if not os.path.exists(
    MANIFEST_FILE
):

    raise FileNotFoundError(

        "\nStage 5C manifest was not found:\n"

        f"{MANIFEST_FILE}\n\n"

        "Run Stage 5C first."

    )


manifest = pd.read_csv(
    MANIFEST_FILE
)


# ============================================================
# 12. REQUIRED MANIFEST COLUMNS
# ============================================================

required_manifest_columns = [

    "ID",

    "gait_file",

    "split",

    "severity_label"

]


missing_columns = [

    column

    for column in required_manifest_columns

    if column not in manifest.columns

]


if missing_columns:

    raise ValueError(

        "\nRequired manifest columns are missing:\n"

        + "\n".join(
            missing_columns
        )

    )


# ============================================================
# 13. LOAD PATIENT
# ============================================================

def load_patient(row):

    patient_id = str(
        row["ID"]
    )

    gait_file = str(
        row["gait_file"]
    )

    # --------------------------------------------------------
    # Check gait file
    # --------------------------------------------------------

    if not os.path.exists(
        gait_file
    ):

        raise FileNotFoundError(

            "\nGait file not found:\n"

            f"{gait_file}"

        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = np.load(

        gait_file,

        allow_pickle=False

    )

    # --------------------------------------------------------
    # Check X
    # --------------------------------------------------------

    if "X" not in data:

        raise ValueError(

            f"\nGait file does not contain 'X': "

            f"{gait_file}"

        )

    gait = data["X"].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if gait.ndim != 3:

        raise ValueError(

            f"\nUnexpected gait shape "

            f"for {patient_id}: "

            f"{gait.shape}"

        )

    # --------------------------------------------------------
    # Expected:
    #
    # windows × 256 × 3
    # --------------------------------------------------------

    if gait.shape[1] != 256:

        raise ValueError(

            f"\nExpected 256 samples per gait window "

            f"for {patient_id}. "

            f"Got: {gait.shape}"

        )

    if gait.shape[2] != 3:

        raise ValueError(

            f"\nExpected 3 gait channels "

            f"for {patient_id}. "

            f"Got: {gait.shape}"

        )

    # --------------------------------------------------------
    # Check NaN / Inf
    # --------------------------------------------------------

    if not np.isfinite(
        gait
    ).all():

        raise ValueError(

            f"\nNaN or Inf found in gait data "

            f"for {patient_id}"

        )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_value = row[
        "severity_label"
    ]

    try:

        label = int(
            float(
                severity_value
            )
        )

    except (
        ValueError,
        TypeError
    ):

        severity_text = str(
            severity_value
        ).strip().lower()

        if severity_text in SEVERITY_TO_LABEL:

            label = SEVERITY_TO_LABEL[
                severity_text
            ]

        else:

            raise ValueError(

                f"\nUnknown severity label "

                f"for {patient_id}: "

                f"{severity_value}"

            )

    # --------------------------------------------------------
    # Validate label
    # --------------------------------------------------------

    if label not in [0, 1, 2]:

        raise ValueError(

            f"\nInvalid severity label "

            f"for {patient_id}: "

            f"{label}"

        )

    severity = LABEL_TO_SEVERITY[
        label
    ]

    return (

        patient_id,

        torch.tensor(
            gait,
            dtype=torch.float32
        ),

        int(label),

        severity

    )


# ============================================================
# 14. CREATE EXISTING PATIENT SPLITS
# ============================================================

train_data = []

val_data = []

test_data = []


for _, row in manifest.iterrows():

    sample = load_patient(
        row
    )

    split = str(
        row["split"]
    ).strip().lower()

    if split == "train":

        train_data.append(
            sample
        )

    elif split == "val":

        val_data.append(
            sample
        )

    elif split == "test":

        test_data.append(
            sample
        )

    else:

        raise ValueError(

            f"Invalid split: {split}"

        )


# ============================================================
# 15. BASIC DATA CHECK
# ============================================================

if len(train_data) == 0:

    raise ValueError(
        "Training dataset is empty."
    )

if len(val_data) == 0:

    raise ValueError(
        "Validation dataset is empty."
    )

if len(test_data) == 0:

    raise ValueError(
        "Testing dataset is empty."
    )


# ============================================================
# 16. MODEL
# ============================================================

model = GaitOnlyClassifier().to(
    DEVICE
)


# ============================================================
# 17. CLASS WEIGHTS
# ============================================================

train_labels = np.asarray(

    [

        sample[2]

        for sample in train_data

    ],

    dtype=np.int64

)


class_counts = np.bincount(

    train_labels,

    minlength=NUM_CLASSES

)


class_weights = np.zeros(

    NUM_CLASSES,

    dtype=np.float32

)


total_train = len(
    train_labels
)


for class_index in range(
    NUM_CLASSES
):

    if class_counts[class_index] > 0:

        class_weights[class_index] = (

            total_train

            /

            (

                NUM_CLASSES

                *

                class_counts[class_index]

            )

        )

    else:

        class_weights[class_index] = 0.0


class_weights_tensor = torch.tensor(

    class_weights,

    dtype=torch.float32,

    device=DEVICE

)


# ============================================================
# 18. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights_tensor

)


optimizer = Adam(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# 19. EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    model,
    dataset
):

    model.eval()

    true_labels = []

    predicted_labels = []

    losses = []

    probabilities_list = []


    with torch.no_grad():

        for (

            patient_id,

            gait,

            label,

            severity

        ) in dataset:

            gait = gait.to(
                DEVICE
            )

            target = torch.tensor(

                [label],

                dtype=torch.long,

                device=DEVICE

            )

            output = model(
                gait
            )

            logits = output[
                "logits"
            ]

            loss = criterion(

                logits.unsqueeze(
                    0
                ),

                target

            )

            probabilities = torch.softmax(

                logits,

                dim=0

            )

            prediction = int(

                torch.argmax(
                    probabilities
                ).item()

            )

            true_labels.append(
                label
            )

            predicted_labels.append(
                prediction
            )

            probabilities_list.append(

                probabilities.cpu().numpy()

            )

            losses.append(
                float(
                    loss.item()
                )
            )


    if len(true_labels) == 0:

        return {

            "loss": np.inf,

            "accuracy": np.nan,

            "balanced_accuracy": np.nan,

            "macro_f1": np.nan,

            "probabilities": None

        }


    accuracy = accuracy_score(

        true_labels,

        predicted_labels

    )


    balanced_accuracy = balanced_accuracy_score(

        true_labels,

        predicted_labels

    )


    macro_f1 = f1_score(

        true_labels,

        predicted_labels,

        average="macro",

        zero_division=0

    )


    return {

        "loss":
            float(
                np.mean(losses)
            ),

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "probabilities":
            probabilities_list

    }


# ============================================================
# 20. TRAINING
# ============================================================

best_val_macro_f1 = -np.inf

best_val_loss = np.inf

best_state = None

best_epoch = 0

epochs_without_improvement = 0


# ------------------------------------------------------------
# Training history
# ------------------------------------------------------------

history = []


print(
    "Training gait-only classification model..."
)

print()


for epoch in range(

    1,

    EPOCHS + 1

):

    model.train()

    train_losses = []


    # --------------------------------------------------------
    # Shuffle patient order
    # --------------------------------------------------------

    random.shuffle(
        train_data
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for (

        patient_id,

        gait,

        label,

        severity

    ) in train_data:

        gait = gait.to(
            DEVICE
        )

        target = torch.tensor(

            [label],

            dtype=torch.long,

            device=DEVICE

        )

        optimizer.zero_grad()


        output = model(
            gait
        )


        logits = output[
            "logits"
        ]


        loss = criterion(

            logits.unsqueeze(
                0
            ),

            target

        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            GRAD_CLIP

        )


        optimizer.step()


        train_losses.append(

            float(
                loss.item()
            )

        )


    train_loss = float(

        np.mean(
            train_losses
        )

    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics = evaluate_dataset(

        model,

        val_data

    )


    val_loss = val_metrics[
        "loss"
    ]

    val_accuracy = val_metrics[
        "accuracy"
    ]

    val_macro_f1 = val_metrics[
        "macro_f1"
    ]


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_balanced_accuracy":
            val_metrics[
                "balanced_accuracy"
            ],

        "val_macro_f1":
            val_macro_f1

    })


    # --------------------------------------------------------
    # Best model selection
    #
    # Primary:
    #   Validation Macro F1
    #
    # Tie breaker:
    #   Validation Loss
    # --------------------------------------------------------

    improved = False


    if val_macro_f1 > best_val_macro_f1:

        improved = True

    elif (

        val_macro_f1 == best_val_macro_f1

        and

        val_loss < best_val_loss

    ):

        improved = True


    if improved:

        best_val_macro_f1 = val_macro_f1

        best_val_loss = val_loss

        best_state = copy.deepcopy(

            model.state_dict()

        )

        best_epoch = epoch

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    if (

        epoch == 1

        or

        epoch % 10 == 0

    ):

        print(

            f"Epoch {epoch:3d} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_accuracy:.4f} | "

            f"Val Macro F1: {val_macro_f1:.4f}"

        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (

        epochs_without_improvement

        >=

        PATIENCE

    ):

        print()

        print(
            "Early stopping."
        )

        break


# ============================================================
# 21. RESTORE BEST MODEL
# ============================================================

if best_state is None:

    raise RuntimeError(
        "No best model state was saved."
    )


model.load_state_dict(
    best_state
)


# ============================================================
# 22. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_file = os.path.join(

    RESULTS_DIR,

    "gait_only_training_history.csv"

)


history_df.to_csv(

    history_file,

    index=False

)


# ============================================================
# 23. TEST EVALUATION
# ============================================================

model.eval()


prediction_rows = []

true_labels = []

predicted_labels = []


with torch.no_grad():

    for (

        patient_id,

        gait,

        label,

        severity

    ) in test_data:

        gait = gait.to(
            DEVICE
        )


        output = model(
            gait
        )


        logits = output[
            "logits"
        ]


        probabilities = torch.softmax(

            logits,

            dim=0

        )


        predicted_label = int(

            torch.argmax(
                probabilities
            ).item()

        )


        confidence = float(

            probabilities[
                predicted_label
            ].item()

        )


        true_labels.append(
            label
        )

        predicted_labels.append(
            predicted_label
        )


        prediction_rows.append({

            "ID":
                patient_id,

            "true_label":
                int(label),

            "true_severity":
                LABEL_TO_SEVERITY[
                    label
                ],

            "predicted_label":
                predicted_label,

            "predicted_severity":
                LABEL_TO_SEVERITY[
                    predicted_label
                ],

            "confidence":
                confidence,

            "prob_mild":
                float(
                    probabilities[0].item()
                ),

            "prob_moderate":
                float(
                    probabilities[1].item()
                ),

            "prob_severe":
                float(
                    probabilities[2].item()
                )

        })


# ============================================================
# 24. TEST METRICS
# ============================================================

accuracy = accuracy_score(

    true_labels,

    predicted_labels

)


balanced_accuracy = balanced_accuracy_score(

    true_labels,

    predicted_labels

)


macro_f1 = f1_score(

    true_labels,

    predicted_labels,

    average="macro",

    zero_division=0

)


weighted_f1 = f1_score(

    true_labels,

    predicted_labels,

    average="weighted",

    zero_division=0

)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(

    true_labels,

    predicted_labels,

    labels=[
        0,
        1,
        2
    ]

)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_labels,

    predicted_labels,

    labels=[
        0,
        1,
        2
    ],

    target_names=[

        "Mild",

        "Moderate",

        "Severe"

    ],

    zero_division=0,

    output_dict=True

)


# ============================================================
# 27. SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_file = os.path.join(

    RESULTS_DIR,

    "gait_only_classification_predictions.csv"

)


predictions_df.to_csv(

    predictions_file,

    index=False

)


# ============================================================
# 28. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "Model":
        "Gait-Only Classification",

    "Architecture":
        "InceptionTime + Temporal Transformer + Window Attention",

    "Epochs_Trained":
        len(history),

    "Best_Epoch":
        best_epoch,

    "Best_Val_Macro_F1":
        best_val_macro_f1,

    "Best_Val_Loss":
        best_val_loss,

    "Accuracy":
        accuracy,

    "Balanced_Accuracy":
        balanced_accuracy,

    "Macro_F1":
        macro_f1,

    "Weighted_F1":
        weighted_f1,

    "Mild_Precision":
        report["Mild"]["precision"],

    "Mild_Recall":
        report["Mild"]["recall"],

    "Mild_F1":
        report["Mild"]["f1-score"],

    "Moderate_Precision":
        report["Moderate"]["precision"],

    "Moderate_Recall":
        report["Moderate"]["recall"],

    "Moderate_F1":
        report["Moderate"]["f1-score"],

    "Severe_Precision":
        report["Severe"]["precision"],

    "Severe_Recall":
        report["Severe"]["recall"],

    "Severe_F1":
        report["Severe"]["f1-score"]

}])


metrics_file = os.path.join(

    RESULTS_DIR,

    "gait_only_classification_metrics.csv"

)


metrics_df.to_csv(

    metrics_file,

    index=False

)


# ============================================================
# 29. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(

    confusion,

    index=[

        "True_Mild",

        "True_Moderate",

        "True_Severe"

    ],

    columns=[

        "Pred_Mild",

        "Pred_Moderate",

        "Pred_Severe"

    ]

)


confusion_file = os.path.join(

    RESULTS_DIR,

    "gait_only_confusion_matrix.csv"

)


confusion_df.to_csv(

    confusion_file

)


# ============================================================
# 30. SAVE MODEL
# ============================================================

model_file = os.path.join(

    MODELS_DIR,

    "gait_only_classification_best.pth"

)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "num_classes":
            NUM_CLASSES,

        "d_model":
            D_MODEL,

        "num_heads":
            NUM_HEADS,

        "num_transformer_layers":
            NUM_TRANSFORMER_LAYERS,

        "transformer_ff_dim":
            TRANSFORMER_FF_DIM,

        "max_windows":
            MAX_WINDOWS,

        "class_mapping":
            LABEL_TO_SEVERITY,

        "best_epoch":
            best_epoch,

        "best_val_macro_f1":
            best_val_macro_f1,

        "best_val_loss":
            best_val_loss

    },

    model_file

)


# ============================================================
# 31. FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "GAIT-ONLY CLASSIFICATION ABLATION RESULTS"
)

print("=" * 70)

print()

print(
    f"Best Epoch:          {best_epoch}"
)

print(
    f"Best Val Macro F1:   {best_val_macro_f1:.4f}"
)

print(
    f"Accuracy:            {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:   {balanced_accuracy:.4f}"
)

print(
    f"Macro F1:            {macro_f1:.4f}"
)

print(
    f"Weighted F1:         {weighted_f1:.4f}"
)

print()

print(
    "Results saved:"
)

print()

print(
    "Metrics:",
    metrics_file
)

print(
    "Predictions:",
    predictions_file
)

print(
    "Confusion Matrix:",
    confusion_file
)

print(
    "Training History:",
    history_file
)

print(
    "Model:",
    model_file
)

print()

print("=" * 70)

print(
    "GAIT-ONLY CLASSIFICATION ABLATION COMPLETE"
)

print("=" * 70)

print()