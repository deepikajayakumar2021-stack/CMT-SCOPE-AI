import copy
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(r"D:\CMT_PROJECT")
DATA_ROOT = ROOT / "processed" / "supplementary_gait"

WINDOW_DIR = DATA_ROOT / "windows"
MANIFEST_FILE = DATA_ROOT / "supplementary_gait_manifest.csv"
SPLIT_FILE = DATA_ROOT / "supplementary_gait_split.csv"

BEST_FILE = DATA_ROOT / "best_gait_encoder.pth"
FINAL_FILE = DATA_ROOT / "final_gait_encoder.pth"
HISTORY_FILE = DATA_ROOT / "pretraining_history.csv"

# ============================================================
# 2. SETTINGS (OPTIMIZED FOR CPU: ~15 MIN TOTAL)
# ============================================================

SEED = 42
INPUT_CHANNELS = 3
WINDOW_LENGTH = 256
NUM_FILTERS = 32
EMBEDDING_SIZE = 32

EPOCHS = 15              # was 40 - enough for 30% subset
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.50
PATIENCE = 6             # was 15
PRINT_EVERY = 10
NUM_WORKERS = 0

# NEW: Use only 30% of training data per epoch for speed
TRAIN_SUBSAMPLE_FRAC = 0.30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 75, flush=True)
print("CMT-SCOPE AI - GAIT PRETRAINING (CPU-OPTIMIZED)", flush=True)
print("=" * 75, flush=True)
print("Device:", DEVICE, flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    print("CPU training enabled.", flush=True)

# ============================================================
# 3. CHECK FILES
# ============================================================

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_FILE}")

if not WINDOW_DIR.is_dir():
    raise FileNotFoundError(f"Window directory not found:\n{WINDOW_DIR}")

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. LOAD MANIFEST AND SPLITS
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)

required = {"subject_id", "window_file"}
missing = required - set(manifest.columns)

if missing:
    raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

if manifest.empty:
    raise ValueError("The supplementary manifest is empty.")

manifest["subject_id"] = manifest["subject_id"].astype(str).str.strip()
manifest["window_file"] = manifest["window_file"].astype(str).str.strip()

if manifest[["subject_id", "window_file"]].isna().any().any():
    raise ValueError("Missing subject IDs or window filenames.")

if "split" not in manifest.columns:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found:\n{SPLIT_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)

    if not {"subject_id", "split"}.issubset(split_df.columns):
        raise ValueError("Split file must contain subject_id and split columns.")

    split_df["subject_id"] = split_df["subject_id"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()

    if split_df["subject_id"].duplicated().any():
        raise ValueError("The split file contains multiple entries per subject.")

    manifest = manifest.merge(
        split_df[["subject_id", "split"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
else:
    manifest["split"] = manifest["split"].astype(str).str.lower().str.strip()

if manifest["split"].isna().any():
    raise ValueError("Some windows have no split assignment.")

valid_splits = {"train", "val", "test"}

if not set(manifest["split"].unique()).issubset(valid_splits):
    raise ValueError(f"Invalid split values: {manifest['split'].unique()}")

subject_split_count = manifest.groupby("subject_id")["split"].nunique()

if (subject_split_count != 1).any():
    bad = subject_split_count[subject_split_count != 1].index.tolist()
    raise RuntimeError("Subjects appear in multiple splits: " + ", ".join(bad[:20]))

train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

train_subjects = set(train_df["subject_id"])
val_subjects = set(val_df["subject_id"])
test_subjects = set(test_df["subject_id"])

assert not (train_subjects & val_subjects)
assert not (train_subjects & test_subjects)
assert not (val_subjects & test_subjects)

print("\nDATASET SUMMARY", flush=True)
print("Total windows:", len(manifest), flush=True)
print("Unique subjects:", manifest["subject_id"].nunique(), flush=True)
print("Train subjects/windows:", len(train_subjects), len(train_df), flush=True)
print("Validation subjects/windows:", len(val_subjects), len(val_df), flush=True)
print("Test subjects/windows:", len(test_subjects), len(test_df), flush=True)
print("Subject-level split check: PASSED", flush=True)

if len(train_df) < 2 or len(val_df) < 2:
    raise ValueError("Train and validation each need at least two windows.")

# ============================================================
# 5. WINDOW LOADER
# ============================================================

def load_window(filename):
    path = WINDOW_DIR / str(filename)

    if not path.exists():
        raise FileNotFoundError(f"Window not found:\n{path}")

    with np.load(path, allow_pickle=False) as data:
        if "signal" in data.files:
            x = data["signal"]
        elif "X" in data.files:
            x = data["X"]
        elif "gait" in data.files:
            x = data["gait"]
        elif "windows" in data.files:
            x = data["windows"]
        elif len(data.files) == 1:
            x = data[data.files[0]]
        else:
            raise ValueError(f"Cannot identify signal array in {path}. Arrays: {data.files}")

        x = np.asarray(x, dtype=np.float32)

    if x.shape == (256, 3):
        x = x.T

    if x.shape != (3, 256):
        raise ValueError(f"Unexpected shape {x.shape} in {path}; expected (3,256).")

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values found in {path}")

    return np.ascontiguousarray(x, dtype=np.float32)


# ============================================================
# 6. AUGMENTATION (WEAKENED FOR STABILITY)
# ============================================================

def augment_gait(signal, rng):
    x = signal.copy()

    if rng.random() < 0.8:
        scale = rng.uniform(0.90, 1.10, size=(3, 1))
        x *= scale.astype(np.float32)

    if rng.random() < 0.8:
        noise = rng.normal(0.0, 0.02, size=x.shape)
        x += noise.astype(np.float32)

    # WEAKENED: shift reduced from ±8 to ±3
    if rng.random() < 0.3:
        shift = int(rng.integers(-3, 4))
        if shift:
            x = np.roll(x, shift=shift, axis=1)

    # WEAKENED: mask reduced from 5-20 to 3-10
    if rng.random() < 0.2:
        length = int(rng.integers(3, 11))
        start = int(rng.integers(0, 257 - length))
        x[:, start:start + length] = 0.0

    return np.ascontiguousarray(x, dtype=np.float32)


# ============================================================
# 7. DATASET
# ============================================================

class GaitContrastiveDataset(Dataset):
    def __init__(self, dataframe, training=True, seed=42):
        self.df = dataframe.reset_index(drop=True)
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        signal = load_window(row["window_file"])

        if self.training:
            seed1 = int(np.random.randint(0, 2**31 - 1))
            seed2 = int(np.random.randint(0, 2**31 - 1))
            rng1 = np.random.default_rng(seed1)
            rng2 = np.random.default_rng(seed2)
        else:
            rng1 = np.random.default_rng(self.seed + index * 2)
            rng2 = np.random.default_rng(self.seed + index * 2 + 1)

        view1 = augment_gait(signal, rng1)
        view2 = augment_gait(signal, rng2)

        return torch.from_numpy(view1), torch.from_numpy(view2)


# ============================================================
# SUBSAMPLE TRAINING DATA FOR CPU SPEED
# ============================================================

train_df_sampled = train_df.sample(
    frac=TRAIN_SUBSAMPLE_FRAC, random_state=SEED
).reset_index(drop=True)

print(f"\n[CPU OPTIMIZATION]")
print(f"Full training set: {len(train_df)} windows")
print(f"Sampled training set: {len(train_df_sampled)} windows")
print(f"Subsample fraction: {TRAIN_SUBSAMPLE_FRAC}")

train_dataset = GaitContrastiveDataset(
    train_df_sampled, training=True, seed=SEED
)

val_dataset = GaitContrastiveDataset(
    val_df, training=False, seed=SEED + 10000
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

print("\nTraining batches per epoch:", len(train_loader), flush=True)
print("Validation batches per epoch:", len(val_loader), flush=True)

# ============================================================
# 8. INCEPTION BLOCK
# ============================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, num_filters=32):
        super().__init__()

        self.bottleneck = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=19, padding=9)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=39, padding=19)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.pool_conv = nn.Conv1d(in_channels, num_filters, kernel_size=1)

        self.bn = nn.BatchNorm1d(num_filters * 4)

        self.residual = nn.Conv1d(in_channels, num_filters * 4, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(num_filters * 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        b1 = self.conv1(bottleneck)
        b2 = self.conv2(bottleneck)
        b3 = self.conv3(bottleneck)
        b4 = self.pool_conv(self.maxpool(x))

        output = torch.cat([b1, b2, b3, b4], dim=1)
        output = self.bn(output)

        residual = self.residual_bn(self.residual(x))

        return self.relu(output + residual)


# ============================================================
# 9. ENCODER
# ============================================================

class InceptionTime(nn.Module):
    def __init__(self, input_channels=3, num_filters=32, embedding_size=32):
        super().__init__()

        self.block1 = InceptionBlock(input_channels, num_filters)
        self.block2 = InceptionBlock(num_filters * 4, num_filters)
        self.block3 = InceptionBlock(num_filters * 4, num_filters)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Linear(num_filters * 4, embedding_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.embedding(x)


# ============================================================
# 10. PROJECTION HEAD
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.network(x), p=2, dim=1)


class GaitContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = InceptionTime(INPUT_CHANNELS, NUM_FILTERS, EMBEDDING_SIZE)
        self.projector = ProjectionHead(EMBEDDING_SIZE, 64, 32)

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projector(embedding)
        return embedding, projection


# ============================================================
# 11. NT-XENT LOSS
# ============================================================

def nt_xent_loss(z1, z2, temperature=0.50):
    batch_size = z1.shape[0]

    if batch_size < 2:
        raise ValueError("Contrastive loss needs at least two windows.")

    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature

    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(self_mask, -1e9)

    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    return F.cross_entropy(similarity, positives)


# ============================================================
# 12. MODEL AND OPTIMIZER
# ============================================================

model = GaitContrastiveModel().to(DEVICE)

print("\nEncoder parameters:", f"{sum(p.numel() for p in model.encoder.parameters()):,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)


# ============================================================
# 13. TRAINING
# ============================================================

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    total_batches = len(train_loader)
    start = time.time()

    print(f"\nEpoch {epoch:03d}: training started", flush=True)

    for batch_idx, (view1, view2) in enumerate(train_loader, start=1):
        if view1.shape[0] < 2:
            continue

        view1 = view1.to(DEVICE, non_blocking=True)
        view2 = view2.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, z1 = model(view1)
        _, z2 = model(view2)

        loss = nt_xent_loss(z1, z2, TEMPERATURE)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

        if batch_idx == 1 or batch_idx % PRINT_EVERY == 0 or batch_idx == total_batches:
            print(
                f"Batch {batch_idx}/{total_batches} "
                f"| Loss {loss.item():.5f} "
                f"| Mean {total_loss / valid_batches:.5f} "
                f"| Elapsed {time.time() - start:.1f}s",
                flush=True,
            )

    if valid_batches == 0:
        raise RuntimeError("No valid training batches were processed.")

    return total_loss / valid_batches


# ============================================================
# 14. VALIDATION
# ============================================================

@torch.no_grad()
def validate(epoch):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    total_batches = len(val_loader)

    print(f"Epoch {epoch:03d}: validation started", flush=True)

    for batch_idx, (view1, view2) in enumerate(val_loader, start=1):
        if view1.shape[0] < 2:
            continue

        view1 = view1.to(DEVICE, non_blocking=True)
        view2 = view2.to(DEVICE, non_blocking=True)

        _, z1 = model(view1)
        _, z2 = model(view2)

        loss = nt_xent_loss(z1, z2, TEMPERATURE)

        total_loss += loss.item()
        valid_batches += 1

        if batch_idx == 1 or batch_idx % PRINT_EVERY == 0 or batch_idx == total_batches:
            print(
                f"Validation {batch_idx}/{total_batches} "
                f"| Mean loss {total_loss / valid_batches:.5f}",
                flush=True,
            )

    if valid_batches == 0:
        raise RuntimeError("No valid validation batches were processed.")

    return total_loss / valid_batches


# ============================================================
# 15. CHECKPOINT SAVING
# ============================================================

def cpu_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_encoder(path, epoch, val_loss):
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model.encoder),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "embedding_size": EMBEDDING_SIZE,
            "input_channels": INPUT_CHANNELS,
            "window_length": WINDOW_LENGTH,
            "num_filters": NUM_FILTERS,
            "architecture": "CMT-SCOPE AI InceptionTime",
            "pretraining": "self-supervised contrastive gait representation",
            "dataset_role": "supplementary general gait representation pretraining",
            "seed": SEED,
            "train_subject_count": len(train_subjects),
            "validation_subject_count": len(val_subjects),
            "train_window_count": len(train_df),
            "validation_window_count": len(val_df),
            "train_subsample_fraction": TRAIN_SUBSAMPLE_FRAC,
            "loss": "NT-Xent",
        },
        path,
    )


# ============================================================
# 16. MAIN PRETRAINING LOOP
# ============================================================

print("\n" + "=" * 75, flush=True)
print("STARTING SELF-SUPERVISED PRETRAINING", flush=True)
print("=" * 75, flush=True)
print("No Mild/Moderate/Severe labels are used.", flush=True)
print(f"Learning rate: {LEARNING_RATE}", flush=True)
print(f"Temperature: {TEMPERATURE}", flush=True)
print(f"Batch size: {BATCH_SIZE}", flush=True)
print(f"Epochs: {EPOCHS}", flush=True)
print(f"Patience: {PATIENCE}", flush=True)

best_val_loss = float("inf")
best_epoch = 0
best_state = None
patience_count = 0
history = []
overall_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    train_loss = train_one_epoch(epoch)
    val_loss = validate(epoch)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_time_seconds": time.time() - epoch_start,
    })

    pd.DataFrame(history).to_csv(HISTORY_FILE, index=False)
    scheduler.step(val_loss)

    print(
        f"\nEpoch {epoch}/{EPOCHS} "
        f"| Train {train_loss:.6f} "
        f"| Validation {val_loss:.6f}",
        flush=True,
    )

    if np.isfinite(val_loss) and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_count = 0

        best_state = copy.deepcopy(cpu_state_dict(model.encoder))
        save_encoder(BEST_FILE, epoch, val_loss)

        print("BEST ENCODER SAVED:", BEST_FILE, flush=True)
    else:
        patience_count += 1
        print(f"No improvement: {patience_count}/{PATIENCE}", flush=True)

    if patience_count >= PATIENCE:
        print("Early stopping triggered.", flush=True)
        break

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

# ============================================================
# 17. RESTORE BEST MODEL AND SAVE FINAL ENCODER
# ============================================================

model.encoder.load_state_dict(best_state, strict=True)
model.encoder.eval()

save_encoder(FINAL_FILE, best_epoch, best_val_loss)

checkpoint = torch.load(FINAL_FILE, map_location="cpu", weights_only=False)

verification_encoder = InceptionTime(INPUT_CHANNELS, NUM_FILTERS, EMBEDDING_SIZE)
verification_encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
verification_encoder.eval()

with torch.no_grad():
    dummy = torch.zeros(2, 3, 256)
    output = verification_encoder(dummy)

if output.shape != (2, 32):
    raise RuntimeError(f"Unexpected embedding shape: {tuple(output.shape)}")

# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75, flush=True)
print("PRETRAINING COMPLETED", flush=True)
print("=" * 75, flush=True)

print("Best epoch:", best_epoch, flush=True)
print("Best validation contrastive loss:", best_val_loss, flush=True)
print("Total subjects:", manifest["subject_id"].nunique(), flush=True)
print("Total windows:", len(manifest), flush=True)
print("Train subjects:", len(train_subjects), flush=True)
print("Validation subjects:", len(val_subjects), flush=True)
print("Test subjects:", len(test_subjects), flush=True)
print("Training time (minutes):", round((time.time() - overall_start) / 60, 2), flush=True)

print("\nBest encoder:", BEST_FILE, flush=True)
print("Final encoder:", FINAL_FILE, flush=True)
print("History:", HISTORY_FILE, flush=True)
print("Checkpoint load verification: PASSED", flush=True)

print("\nThe test split was not used for gradient updates or checkpoint selection.", flush=True)
print("Pretraining alone does not establish improved CMT1A severity accuracy.", flush=True)
print("Next: transfer the encoder into the three-class CMT1A model and evaluate with patient-level splits.", flush=True)
print("=" * 75, flush=True)