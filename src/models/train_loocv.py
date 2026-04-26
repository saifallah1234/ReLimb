import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_absolute_error,
    classification_report, confusion_matrix
)
from collections import Counter

from model_mlp import GaitMetricsMLP
from model_lstm import GaitSequenceLSTM

# ==========================================
# DECISION: Drop "Unknown / Other"
# ==========================================
# This class represents 65% of data and has no clinical meaning.
# A model trained on it learns to say "I don't know" 65% of the time.
# We train on the 8 real gait patterns only.
EXCLUDE_CLASSES = {"Unknown / Other"}
GAIT_LANDMARK_INDICES = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def hip_center_keypoints(kp: np.ndarray) -> np.ndarray:
    mid_hip_x = (kp[:, 23*2]   + kp[:, 24*2])   / 2
    mid_hip_y = (kp[:, 23*2+1] + kp[:, 24*2+1]) / 2
    out = np.zeros((kp.shape[0], len(GAIT_LANDMARK_INDICES)*2), dtype=np.float32)
    for i, lm in enumerate(GAIT_LANDMARK_INDICES):
        out[:, i*2]   = kp[:, lm*2]   - mid_hip_x
        out[:, i*2+1] = kp[:, lm*2+1] - mid_hip_y
    return out

# ==========================================
# 1. DATASET
# ==========================================

class ProGaitDataset(Dataset):
    def __init__(self, session_folders, labels_dict):
        self.session_folders = session_folders
        self.labels_dict = labels_dict

        # Store raw (unscaled) metrics — scaling is done per-fold externally
        self.raw_metrics = []
        self.masks = []
        self.keypoints_list = []
        self.targets_class = []
        self.targets_ccc = []

        self._load_all_data()

    def _load_all_data(self):
        for folder in self.session_folders:
            session_id = os.path.basename(folder)

            kp_path      = os.path.join(folder, 'keypoints.npy')
            metrics_path = os.path.join(folder, 'gait_metrics.json')

            kp_raw  = np.nan_to_num(np.load(kp_path), nan=0.0)
            kp_norm = hip_center_keypoints(kp_raw)          # [T, 26]
            kp_tensor = torch.tensor(kp_norm, dtype=torch.float32)
            self.keypoints_list.append(kp_tensor)

            with open(metrics_path, 'r') as f:
                m = json.load(f)

            feats = [
                m['symmetry'].get('stride_asymmetry',      np.nan),
                m['symmetry'].get('stance_asymmetry',      np.nan),
                m['symmetry'].get('swing_asymmetry',       np.nan),
                m['symmetry'].get('step_length_asymmetry', np.nan),
                m['symmetry'].get('cadence_bpm',           np.nan),
            ]
            mask  = [1.0 if not (v is None or np.isnan(v)) else 0.0 for v in feats]
            feats = [v  if  not (v is None or np.isnan(v)) else 0.0 for v in feats]

            self.raw_metrics.append(feats)
            self.masks.append(mask)

            label_info = self.labels_dict[session_id]
            self.targets_class.append(label_info['class_id'])
            self.targets_ccc.append(label_info['ccc_score'])

    def __len__(self):
        return len(self.session_folders)

    def __getitem__(self, idx):
        # Return raw (unscaled) metrics — caller applies per-fold scaler
        return {
            'metrics':  torch.tensor(self.raw_metrics[idx], dtype=torch.float32),
            'mask':     torch.tensor(self.masks[idx],       dtype=torch.float32),
            'keypoints': self.keypoints_list[idx],
            'length':   len(self.keypoints_list[idx]),
            'class_id': torch.tensor(self.targets_class[idx], dtype=torch.long),
            'ccc':      torch.tensor(self.targets_ccc[idx],   dtype=torch.float32),
        }


def collate_fn(batch):
    metrics   = torch.stack([b['metrics']  for b in batch])
    masks     = torch.stack([b['mask']     for b in batch])
    class_ids = torch.stack([b['class_id'] for b in batch])
    cccs      = torch.stack([b['ccc']      for b in batch])
    lengths   = torch.tensor([b['length']  for b in batch])
    keypoints = torch.nn.utils.rnn.pad_sequence(
        [b['keypoints'] for b in batch], batch_first=True, padding_value=0.0
    )
    return metrics, masks, keypoints, lengths, class_ids, cccs


# ==========================================
# 2. FUSION MODEL
# ==========================================

class ProGaitFusion(nn.Module):
    def __init__(self, num_classes, mode="fusion"):
        super().__init__()
        self.mode = mode

        self.mlp  = GaitMetricsMLP(n_metrics=5, hidden_dim=32, output_dim=64)
        self.lstm = GaitSequenceLSTM()   # defaults: input=26, hidden=32, output=64
        fusion_dim = {"fusion": 128, "mlp_only": 64, "lstm_only": 64}[mode]

        # Two-layer classifier head — gives the fusion vector more capacity
        # to separate similar gait classes before the final linear projection
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self.ccc_scorer = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, metrics, masks, keypoints, lengths):
        mlp_input = torch.cat([metrics, masks], dim=1)   # (B, 10)
        out_mlp   = self.mlp(mlp_input)                  # (B, 64)
        out_lstm  = self.lstm(keypoints, lengths)         # (B, 64)

        if self.mode == "fusion":
            fused = torch.cat([out_mlp, out_lstm], dim=1)
        elif self.mode == "mlp_only":
            fused = out_mlp
        else:
            fused = out_lstm

        return self.classifier(fused), self.ccc_scorer(fused).squeeze(1)


# ==========================================
# 3. CLASS WEIGHTS
# ==========================================

def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weighting so rare classes (10 samples) get the
    same total gradient signal as the majority class (30 samples).
    """
    counts = Counter(targets)
    total  = len(targets)
    weights = torch.ones(num_classes)   # start at 1.0, overwrite for known classes
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    # Clamp so no single class dominates the loss by more than 10×
    weights = torch.clamp(weights, max=10.0)
    return weights


# ==========================================
# 4. SCALER  (applied ONLY to train split,
#             transform applied to val split
#             using train statistics)
# ==========================================

def apply_scaler(raw_metrics: list, train_idx, val_idx):
    """
    Fit StandardScaler on train split only.
    Returns scaled arrays for train and val — does NOT modify dataset in place.
    """
    train_arr = np.array([raw_metrics[i] for i in train_idx])
    val_arr   = np.array([raw_metrics[i] for i in val_idx])

    scaler = StandardScaler().fit(train_arr)

    scaled_train = scaler.transform(train_arr)
    scaled_val   = scaler.transform(val_arr)

    return scaled_train, scaled_val, scaler


# ==========================================
# 5. TRAINING LOOP
# ==========================================

def train_and_evaluate_kfold(
    dataset,
    num_classes: int,
    mode: str = "fusion",
    epochs: int = 50,
    n_splits: int = 5,
):
    print(f"\n--- Stratified {n_splits}-Fold | mode={mode.upper()} | epochs={epochs} ---")

    # StratifiedKFold preserves class proportions in each fold
    # (important when some classes have only 10 samples)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    targets_array = np.array(dataset.targets_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}\n")

    all_true_classes, all_pred_classes = [], []
    all_true_cccs,    all_pred_cccs    = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(targets_array, targets_array)):
        print(f"── Fold {fold+1}/{n_splits}  "
              f"(train={len(train_idx)}, val={len(val_idx)}) ──")

        # ── Scaler: fit on train, transform both ──────────────────────────
        scaled_train, scaled_val, _ = apply_scaler(
            dataset.raw_metrics, train_idx, val_idx
        )

        # Temporarily patch the dataset with scaled values for this fold
        # We keep a backup so we can restore after each fold
        original_metrics = [row[:] for row in dataset.raw_metrics]
        for i, ti in enumerate(train_idx):
            dataset.raw_metrics[ti] = scaled_train[i].tolist()
        for i, vi in enumerate(val_idx):
            dataset.raw_metrics[vi] = scaled_val[i].tolist()

        # ── Class weights for this fold's train split ─────────────────────
        train_targets = [dataset.targets_class[i] for i in train_idx]
        class_weights = compute_class_weights(train_targets, num_classes).to(device)

        # ── DataLoaders ───────────────────────────────────────────────────
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=8, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=1, shuffle=False, collate_fn=collate_fn,
        )

        # ── Model ─────────────────────────────────────────────────────────
        model = ProGaitFusion(num_classes=num_classes, mode=mode).to(device)

        criterion_class = nn.CrossEntropyLoss(weight=class_weights)
        criterion_ccc   = nn.MSELoss()
        optimizer       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for metrics, masks, keypoints, lengths, class_ids, cccs in train_loader:
                metrics, masks     = metrics.to(device),   masks.to(device)
                keypoints          = keypoints.to(device)
                class_ids, cccs    = class_ids.to(device), cccs.to(device)

                optimizer.zero_grad()
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)

                loss_cls = criterion_class(logits, class_ids)
                loss_ccc = criterion_ccc(ccc_preds, cccs)
                loss = loss_cls + 0.1 * loss_ccc


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"   epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/len(train_loader):.4f}")

        # ── Evaluation ────────────────────────────────────────────────────
        model.eval()
        fold_true, fold_pred = [], []
        with torch.no_grad():
            for metrics, masks, keypoints, lengths, class_ids, cccs in val_loader:
                metrics, masks = metrics.to(device), masks.to(device)
                keypoints      = keypoints.to(device)
                logits, ccc_preds = model(metrics, masks, keypoints, lengths)

                true_cls  = class_ids.item()
                pred_cls  = torch.argmax(logits, dim=1).item()
                fold_true.append(true_cls)
                fold_pred.append(pred_cls)
                all_true_classes.append(true_cls)
                all_pred_classes.append(pred_cls)
                all_true_cccs.append(cccs.item())
                all_pred_cccs.append(ccc_preds.item())

        fold_acc = accuracy_score(fold_true, fold_pred) * 100
        print(f"   Fold {fold+1} accuracy: {fold_acc:.1f}%\n")

        # Restore original (unscaled) metrics for next fold
        dataset.raw_metrics = original_metrics

    # ── Final results ──────────────────────────────────────────────────────
    final_acc = accuracy_score(all_true_classes, all_pred_classes) * 100
    final_mae = mean_absolute_error(all_true_cccs, all_pred_cccs)

    print("=" * 60)
    print(f"OVERALL  Acc={final_acc:.1f}%  |  CCC MAE={final_mae:.3f}")
    print("=" * 60)

    # Per-class breakdown — the number that actually matters
    print("\nPer-class report:")
    print(classification_report(
        all_true_classes, all_pred_classes, zero_division=0
    ))

    # Save final fold's model weights
    save_path = f"relimb_{mode}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved → {save_path}")

    return final_acc, final_mae


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    from pathlib import Path

    print("--- INITIALIZING RELIMB TRAINING ---")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
    INDEX_FILE   = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
    MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"

    with open(MAPPING_FILE, 'r') as f:
        class_mapping = json.load(f)

    with open(INDEX_FILE, 'r') as f:
        metadata = json.load(f)

    session_folders = []
    labels_dict     = {}

    skipped_unknown = 0
    skipped_missing = 0

    for item in metadata:
        # ── Skip "Unknown / Other" entirely ───────────────────────────────
        issue_text = item.get("clean_primary_issue") or "Unknown / Other"
        if issue_text in EXCLUDE_CLASSES:
            skipped_unknown += 1
            continue

        raw_id   = item["ID"].replace(".mp4", "").replace(".avi", "")
        for prefix in ("inside_", "outside_"):
            folder_key = f"{prefix}{raw_id}"
            kp_path    = SESSIONS_DIR / folder_key / "keypoints.npy"
            if kp_path.exists():
                raw_ccc = item.get("ccc_score")
                if raw_ccc is None or str(raw_ccc).strip().lower() == 'nan':
                    ccc_score = 0.0
                else:
                    ccc_score = float(raw_ccc)
                    if math.isnan(ccc_score):
                        ccc_score = 0.0

                class_id = class_mapping.get(issue_text, 0)

                session_folders.append(str(SESSIONS_DIR / folder_key))
                labels_dict[folder_key] = {
                    "class_id": class_id,
                    "ccc_score": ccc_score,
                }
                break
        else:
            skipped_missing += 1

    print(f"✅ Training samples : {len(session_folders)}")
    print(f"   Skipped Unknown  : {skipped_unknown}")
    print(f"   Skipped missing  : {skipped_missing}")

    # Re-index class IDs to be contiguous (0,1,2,...) after dropping Unknown/Other.
    # Original mapping may have gaps (e.g. 0-6, 8) because the removed class
    # sat in the middle. We sort by original ID and reassign sequentially.
    real_classes_sorted = sorted(
        [(k, v) for k, v in class_mapping.items() if k not in EXCLUDE_CLASSES],
        key=lambda x: x[1]   # sort by original ID
    )
    # old_id → new contiguous id
    remap = {old_id: new_id for new_id, (_, old_id) in enumerate(real_classes_sorted)}
    num_classes = len(real_classes_sorted)

    # Apply remap to labels_dict
    for key in labels_dict:
        old_id = labels_dict[key]["class_id"]
        labels_dict[key]["class_id"] = remap[old_id]

    print(f"🧠 Classes          : {num_classes}")
    for new_id, (name, _) in enumerate(real_classes_sorted):
        print(f"   {new_id}: {name}")

    dataset = ProGaitDataset(session_folders, labels_dict)

    train_and_evaluate_kfold(
        dataset,
        num_classes=num_classes,
        mode="fusion",
        epochs=50,
    )