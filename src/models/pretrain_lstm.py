import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model_lstm import GaitSequenceLSTM, GAIT_LANDMARK_INDICES

# ───────────────────────────── Paths ─────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SESSION_DIR  = PROJECT_ROOT / "data" / "sessions"
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────── Config ─────────────────────────────
CLIP_FRAMES    = 150
CONTEXT_FRAMES = 100
PREDICT_FRAMES = 50


# ───────────────────────────── Preprocessing ─────────────────────────────^
"""
Without the hip,
model may learn position in frame camera bias Instead of actual walking pattern
"""
def hip_center_keypoints(kp: np.ndarray) -> np.ndarray:
    mid_hip_x = (kp[:, 23*2] + kp[:, 24*2]) / 2
    mid_hip_y = (kp[:, 23*2+1] + kp[:, 24*2+1]) / 2

    out = np.zeros((kp.shape[0], len(GAIT_LANDMARK_INDICES)*2), dtype=np.float32)

    for i, lm in enumerate(GAIT_LANDMARK_INDICES):
        out[:, i*2]   = kp[:, lm*2]   - mid_hip_x
        out[:, i*2+1] = kp[:, lm*2+1] - mid_hip_y

    max_abs = np.max(np.abs(out))
    if max_abs > 1e-6:
        out /= max_abs

    return out


# ───────────────────────────── Dataset ─────────────────────────────
class GaitDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for folder in sorted(root.iterdir()):
            if "_clip_" not in folder.name:
                continue

            path = folder / "keypoints.npy"
            if not path.exists():
                continue

            kp = np.load(path)
            kp = np.nan_to_num(kp)

            # ── FORCE FIX LENGTH (CRITICAL FIX) ──
            if kp.shape[0] < CLIP_FRAMES:
                pad = np.zeros((CLIP_FRAMES - kp.shape[0], kp.shape[1]))
                kp = np.concatenate([kp, pad], axis=0)
            else:
                kp = kp[:CLIP_FRAMES]

            # now guaranteed shape (150, 66)
            kp = hip_center_keypoints(kp)
            kp = torch.tensor(kp, dtype=torch.float32)

            # sanity check (optional but useful)
            assert kp.shape[0] == CLIP_FRAMES, kp.shape

            self.samples.append(kp)

        print(f"Loaded {len(self.samples)} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip = self.samples[idx]

        context = clip[:CONTEXT_FRAMES]     # (100, 26)
        target  = clip[CONTEXT_FRAMES:]     # (50, 26)

        # HARD ASSERT (prevents silent bugs)
        assert context.shape == (CONTEXT_FRAMES, 26)
        assert target.shape == (PREDICT_FRAMES, 26)

        return context, target


# ───────────────────────────── Model ─────────────────────────────
class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = GaitSequenceLSTM()

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, PREDICT_FRAMES * 26)
        )

    def forward(self, x):
        B = x.size(0)
        lengths = torch.full((B,), CONTEXT_FRAMES, dtype=torch.long)

        h = self.encoder(x, lengths)     # (B, 64)
        out = self.decoder(h)

        return out.view(B, PREDICT_FRAMES, 26)


# ───────────────────────────── Train ─────────────────────────────
def train(epochs, lr, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GaitDataset(SESSION_DIR)

    # 90/10 split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = PretrainModel().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):

        # ───── TRAIN ─────
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ───── VALIDATION ─────
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                val_loss += loss_fn(pred, y).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        print(f"Epoch {epoch:03d} | Train {train_loss:.5f} | Val {val_loss:.5f}")

        # ───── SAVE BEST ─────
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
            "lstm": model.encoder.lstm.state_dict(),
            "proj": model.encoder.proj.state_dict(),}

    torch.save(best_state, MODELS_DIR / "lstm_pretrained.pth")
    print("Best model saved !")


# ───────────────────────────── Main ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size)