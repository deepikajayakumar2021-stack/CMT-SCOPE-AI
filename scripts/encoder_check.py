import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score

ROOT = Path(r"D:\CMT_PROJECT")
DATA_ROOT = ROOT / "processed" / "supplementary_gait"
WINDOW_DIR = DATA_ROOT / "windows"
MANIFEST_FILE = DATA_ROOT / "supplementary_gait_manifest.csv"
SPLIT_FILE = DATA_ROOT / "supplementary_gait_split.csv"
CKPT = DATA_ROOT / "best_gait_encoder.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Encoder architecture (must match pretraining) ----------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, num_filters=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=19, padding=9)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=39, padding=19)
        self.maxpool = nn.MaxPool1d(3, 1, 1)
        self.pool_conv = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_filters * 4)
        self.residual = nn.Conv1d(in_channels, num_filters * 4, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(num_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([self.conv1(b), self.conv2(b), self.conv3(b),
                         self.pool_conv(self.maxpool(x))], dim=1)
        out = self.bn(out) + self.residual_bn(self.residual(x))
        return self.relu(out)

class InceptionTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = InceptionBlock(3, 32)
        self.block2 = InceptionBlock(128, 32)
        self.block3 = InceptionBlock(128, 32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(128, 32)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.embedding(x)


# ---------- Load checkpoint ----------
print("Loading checkpoint...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
encoder = InceptionTime()
encoder.load_state_dict(ckpt["model_state_dict"])
encoder.to(DEVICE)
encoder.eval()
print(f"Checkpoint epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f}")

# ---------- Load manifest + split ----------
print("\nLoading manifest and split...")
manifest = pd.read_csv(MANIFEST_FILE)
manifest["subject_id"] = manifest["subject_id"].astype(str).str.strip()
manifest["window_file"] = manifest["window_file"].astype(str).str.strip()

# FIX: Merge split file if 'split' column isn't present
if "split" not in manifest.columns:
    print("Merging split file into manifest...")
    split_df = pd.read_csv(SPLIT_FILE)
    split_df["subject_id"] = split_df["subject_id"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    manifest = manifest.merge(
        split_df[["subject_id", "split"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

print("Split distribution in manifest:")
print(manifest["split"].value_counts(dropna=False))

# ---------- Load test windows ----------
test = manifest[manifest["split"] == "test"].head(300).reset_index(drop=True)
print(f"\nTest windows to evaluate: {len(test)}")

def load_window(fn):
    path = WINDOW_DIR / str(fn)
    if not path.exists():
        raise FileNotFoundError(f"Window not found: {path}")
    with np.load(path, allow_pickle=False) as d:
        for k in ["signal", "X", "gait", "windows"]:
            if k in d.files:
                x = d[k]; break
        else:
            x = d[d.files[0]]
    x = np.asarray(x, dtype=np.float32)
    if x.shape == (256, 3):
        x = x.T
    return x

print(f"Loading {len(test)} windows...")
signals = []
valid_subjects = []
for _, row in test.iterrows():
    try:
        s = load_window(row["window_file"])
        signals.append(s)
        valid_subjects.append(row["subject_id"])
    except Exception as e:
        print(f"  Skip {row['window_file']}: {e}")

signals = np.stack(signals)
signals_t = torch.from_numpy(signals)
labels = np.array(valid_subjects)
print(f"Loaded {len(signals)} windows from {len(np.unique(labels))} unique subjects")

# ---------- Extract embeddings ----------
print("\nExtracting embeddings...")
with torch.no_grad():
    all_emb = []
    for i in range(0, len(signals_t), 64):
        batch = signals_t[i:i+64].to(DEVICE)
        emb = encoder(batch).cpu().numpy()
        all_emb.append(emb)
embeddings = np.concatenate(all_emb, axis=0)
print(f"Embeddings shape: {embeddings.shape}")

# ---------- Statistics ----------
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
mean_vec = embeddings.mean(axis=0)
std_vec = embeddings.std(axis=0)
print(f"  Mean abs of mean vector:  {np.abs(mean_vec).mean():.4f}")
print(f"  Std across dims (mean):   {std_vec.mean():.4f}")
print(f"  Std across dims (min):    {std_vec.min():.4f}")
print(f"  Std across dims (max):    {std_vec.max():.4f}")
print(f"  # dims with std < 0.01:    {(std_vec < 0.01).sum()}/{len(std_vec)}")
print(f"  # dims with std < 0.10:    {(std_vec < 0.10).sum()}/{len(std_vec)}")

# ---------- Pairwise similarity ----------
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings[:100])
np.fill_diagonal(sim, 0)
print(f"\n  Pairwise cosine similarity (random 100 windows):")
print(f"    Mean:  {sim.mean():.4f}")
print(f"    Max:   {sim.max():.4f}")
print(f"    Min:   {sim.min():.4f}")
print("    (>0.95 = collapse, 0.3-0.7 = healthy diversity)")

# ---------- Subject clustering ----------
unique_subjects = np.unique(labels)
if len(unique_subjects) >= 2:
    try:
        sil = silhouette_score(embeddings, labels)
        print(f"\n  Silhouette score (subject clustering): {sil:.4f}")
        print("    (<0.0 = no clustering, 0.0-0.1 = weak, 0.1-0.25 = mild, >0.25 = good)")
    except Exception as e:
        print(f"  Silhouette failed: {e}")

# ---------- Verdict ----------
print("\n" + "="*60)
print("VERDICT")
print("="*60)

mean_std = std_vec.mean()
collapsed_dims = (std_vec < 0.01).sum()
total_dims = len(std_vec)
mean_sim = sim.mean()

if mean_std < 0.1 or collapsed_dims > total_dims * 0.5:
    print("✗ ENCODER COLLAPSED")
    print(f"  Mean std = {mean_std:.4f} (target > 0.5)")
    print(f"  Collapsed dims = {collapsed_dims}/{total_dims}")
    print("  The encoder produces nearly identical embeddings for all inputs.")
    print("  >> DO NOT run 6_model.py — it will give 60% accuracy again.")
    print("  >> The pretraining loss (NT-Xent) has collapsed.")
    print("  >> Need to switch to VICReg or Barlow Twins loss and retrain.")

elif mean_std < 0.5 or mean_sim > 0.90:
    print("⚠ ENCODER WEAK")
    print(f"  Mean std = {mean_std:.4f} (target > 0.5)")
    print(f"  Mean pairwise similarity = {mean_sim:.4f} (target < 0.7)")
    print("  Encoder learned something but with limited variance.")
    print("  >> 6_model.py will give ~70-80% accuracy.")
    print("  >> Acceptable if you're short on time.")

else:
    print("✓ ENCODER HEALTHY")
    print(f"  Mean std = {mean_std:.4f}")
    print(f"  Mean pairwise similarity = {mean_sim:.4f}")
    print("  >> Run 6_model.py — expected 85-93% accuracy.")