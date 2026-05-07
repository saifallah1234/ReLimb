"""
train_lstm_only.py
==================
Trains GaitSequenceLSTM + classifier head from scratch on labeled clips only
(excludes Unknown / Other).

Distribution-aware fixes applied:
- Excludes Unknown/Other from training (noisy labels, 19% of data)
- WeightedRandomSampler ensures each class seen equally per epoch
- CrossEntropyLoss with inverse-freq weights as backup signal
- Label smoothing 0.1 prevents overconfidence on Knee Issue (46%)
- StratifiedKFold so tiny Normal Gait class (29 clips) always appears in val
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import mlflow

from src.models.dataset import ProGaitDataset, pad_collate_fn
from src.models.model_lstm import GaitSequenceLSTM
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"Unknown / Other"}


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total  = len(labels)
    w = torch.ones(num_classes)
    for cls, cnt in counts.items():
        w[cls] = (total / cnt) ** 0.7   # softer than pure inverse-freq
    return torch.clamp(w, max=10.0)


def make_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_training(epochs, lr, batch_size, n_splits, seed, label_smoothing, head_dim, dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device.type.upper()}")

    # ── Dataset: labeled clips only ───────────────────────────────────────
    full_ds = ProGaitDataset()
    num_classes = len(full_ds.class_mapping)

    labeled_idx = [
        i for i, s in enumerate(full_ds.valid_sessions)
        if s["issue_text"] not in EXCLUDE
    ]
    labels_arr = np.array([
        full_ds.class_mapping[full_ds.valid_sessions[i]["issue_text"]]
        for i in labeled_idx
    ])

    print(f"Labeled clips   : {len(labeled_idx)}  (excluded Unknown: {len(full_ds) - len(labeled_idx)})")
    print(f"Classes         : {num_classes}")
    for name, idx in sorted(full_ds.class_mapping.items(), key=lambda x: x[1]):
        if name in EXCLUDE: continue
        cnt = (labels_arr == idx).sum()
        print(f"  [{idx}] {name:<25} {cnt} clips")
    print()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred = [], []

    for fold, (tr_rel, va_rel) in enumerate(skf.split(labeled_idx, labels_arr)):
        tr_idx = [labeled_idx[i] for i in tr_rel]
        va_idx = [labeled_idx[i] for i in va_rel]
        tr_labels = labels_arr[tr_rel].tolist()

        print(f"── Fold {fold+1}/{n_splits}  train={len(tr_idx)}  val={len(va_idx)} ──")

        cw      = compute_class_weights(tr_labels, num_classes).to(device)
        sampler = make_sampler(tr_labels)

        train_loader = DataLoader(
            Subset(full_ds, tr_idx), batch_size=batch_size,
            sampler=sampler, collate_fn=pad_collate_fn,
        )
        val_loader = DataLoader(
            Subset(full_ds, va_idx), batch_size=batch_size,
            shuffle=False, collate_fn=pad_collate_fn,
        )

        encoder    = GaitSequenceLSTM().to(device)
        classifier = nn.Sequential(
            nn.Linear(64, head_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 2)

        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

        best_f1, best_state = 0.0, None

        for epoch in range(1, epochs + 1):
            # ── train ────────────────────────────────────────────────────
            encoder.train(); classifier.train()
            tr_loss = correct = total = 0
            for kp, _, _, issue, lengths in train_loader:
                kp, issue, lengths = kp.to(device), issue.to(device), lengths.to(device)
                optimizer.zero_grad()
                logits = classifier(encoder(kp, lengths))
                loss   = criterion(logits, issue)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                tr_loss += loss.item()
                correct += (logits.argmax(1) == issue).sum().item()
                total   += issue.numel()
            scheduler.step()

            # ── val ──────────────────────────────────────────────────────
            encoder.eval(); classifier.eval()
            fold_true, fold_pred = [], []
            with torch.no_grad():
                for kp, _, _, issue, lengths in val_loader:
                    kp, issue, lengths = kp.to(device), issue.to(device), lengths.to(device)
                    logits = classifier(encoder(kp, lengths))
                    fold_true.extend(issue.tolist())
                    fold_pred.extend(logits.argmax(1).tolist())

            val_acc = sum(t == p for t, p in zip(fold_true, fold_pred)) / len(fold_true) * 100
            val_f1  = f1_score(fold_true, fold_pred, average="macro", zero_division=0)

            if epoch % 10 == 0:
                print(f"   ep {epoch:3d}  loss={tr_loss/len(train_loader):.4f}  "
                      f"acc={val_acc:.1f}%  f1={val_f1:.3f}")

            if mlflow.active_run():
                mlflow.log_metrics({
                    "train_loss": tr_loss / len(train_loader),
                    "val_acc": val_acc, "val_f1_macro": val_f1,
                }, step=(fold * epochs) + epoch)

            if val_f1 > best_f1:
                best_f1   = val_f1
                best_state = {
                    "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                    "classifier": {k: v.cpu() for k, v in classifier.state_dict().items()},
                }
                all_true_fold, all_pred_fold = fold_true[:], fold_pred[:]

        all_true.extend(all_true_fold)
        all_pred.extend(all_pred_fold)
        print(f"   Fold {fold+1} best F1={best_f1:.3f}\n")

    # ── Final report ──────────────────────────────────────────────────────
    class_names = [n for n, _ in sorted(full_ds.class_mapping.items(), key=lambda x: x[1])]
    print("=" * 55)
    print("LSTM-ONLY  OVERALL RESULTS")
    print("=" * 55)
    print(classification_report(all_true, all_pred, target_names=class_names, zero_division=0))

    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    save_path  = MODELS_DIR / "lstm_only.pth"
    if best_state:
        torch.save(best_state, save_path)
        print(f"💾 Saved → {save_path}")

    return {"val_f1_macro": overall_f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--n_splits",        type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--head_dim",        type=int,   default=64)
    parser.add_argument("--dropout",         type=float, default=0.3)
    args = parser.parse_args()
    run_training(
        args.epochs, args.lr, args.batch_size, args.n_splits,
        args.seed, args.label_smoothing, args.head_dim, args.dropout,
    )