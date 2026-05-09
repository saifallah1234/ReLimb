import argparse
from pathlib import Path
from collections import Counter
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import confusion_matrix, classification_report

from src.models.dataset import ProGaitDataset, pad_collate_fn
from src.models.model_stgcn import GaitSTGCN
from torch.utils.data import WeightedRandomSampler

def split_indices(n_items: int, val_split: float, seed: int):
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(n_items, generator=generator).tolist()

    val_size = max(1, int(n_items * val_split))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return train_idx, val_idx


def _extract_group_id(folder_name: str) -> str:
    """Group id for leakage-safe split.

    We group all clips that come from the same source video.
    Example: inside_1_1_1_f_clip_000 -> inside_1_1_1_f
             outside_2_4_7_s_1_clip_006 -> outside_2_4_7_s_1
    """
    # keep inside_/outside_ prefix: it's part of the original source identity
    return folder_name.rsplit("_clip_", 1)[0]


def split_indices_by_group(dataset: ProGaitDataset, val_split: float, seed: int):
    """Split indices such that no group (source video) appears in both splits."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(dataset.valid_sessions):
        gid = _extract_group_id(s["folder"].name)
        groups.setdefault(gid, []).append(i)

    g_list = sorted(groups.keys())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(g_list), generator=generator).tolist()
    g_list = [g_list[i] for i in order]

    target_val = max(1, int(len(dataset) * val_split))

    val_idx: list[int] = []
    train_idx: list[int] = []
    for gid in g_list:
        idxs = groups[gid]
        if len(val_idx) < target_val:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    # Safety: if we overshot too much (rare with huge groups), move last group back.
    if not train_idx:
        # Ensure non-empty train set.
        moved = val_idx[-len(groups[g_list[-1]]):]
        val_idx = val_idx[:-len(moved)]
        train_idx = moved

    return train_idx, val_idx, groups


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)

    total = len(labels)

    weights = torch.ones(num_classes)

    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)

    return torch.clamp(weights, max=10.0)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    max_batches=None
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):

        keypoints, _, _, issues, lengths = batch

        keypoints = keypoints.to(device)
        issues = issues.to(device)

        optimizer.zero_grad()

        logits = model(keypoints, lengths=lengths)

        loss = criterion(logits, issues)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        correct += (preds == issues).sum().item()
        total += issues.numel()

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(1, len(loader))
    accuracy = (correct / total) * 100 if total else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    max_batches=None
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):

        keypoints, _, _, issues, lengths = batch

        keypoints = keypoints.to(device)
        issues = issues.to(device)

        logits = model(keypoints, lengths=lengths)

        loss = criterion(logits, issues)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        correct += (preds == issues).sum().item()
        total += issues.numel()

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(1, len(loader))
    accuracy = (correct / total) * 100 if total else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def evaluate_predictions(
    model,
    loader,
    device,
    max_batches=None,
):
    """Return y_true, y_pred for a loader."""
    model.eval()
    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(loader):
        keypoints, _, _, issues, lengths = batch
        keypoints = keypoints.to(device)
        issues = issues.to(device)

        logits = model(keypoints, lengths=lengths)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(issues.tolist())
        y_pred.extend(preds.tolist())

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return y_true, y_pred


def run_training(
    epochs,
    lr,
    batch_size,
    val_split,
    seed,
    max_batches
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # ---------------------------------------------------
    # Dataset
    # ---------------------------------------------------

    dataset = ProGaitDataset()

    num_classes = len(dataset.class_mapping)

    train_idx, val_idx, groups = split_indices_by_group(
        dataset=dataset,
        val_split=val_split,
        seed=seed,
    )

    print(
        f"Group split: {len(groups)} source videos | "
        f"train clips={len(train_idx)} | val clips={len(val_idx)}"
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn
    )

    # ---------------------------------------------------
    # Class weights
    # ---------------------------------------------------

    train_labels = [
        dataset.class_mapping[
            dataset.valid_sessions[i]["issue_text"]
        ]
        for i in train_idx
    ]

    class_weights = compute_class_weights(
        train_labels,
        num_classes
    ).to(device)

    print("\nClass Weights:")
    print(class_weights)

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------

    model = GaitSTGCN(
        num_joints=13,
        num_classes=num_classes
    ).to(device)

    # ---------------------------------------------------
    # Loss + Optimizer
    # ---------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    # ---------------------------------------------------
    # Save path
    # ---------------------------------------------------

    models_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "models"
    )

    models_dir.mkdir(parents=True, exist_ok=True)

    save_path = models_dir / "stgcn_best.pth"

    best_val_acc = 0.0

    # ---------------------------------------------------
    # Training loop
    # ---------------------------------------------------

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
            max_batches
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.1f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.1f}%"
        )

        # -----------------------------------------
        # Evaluate predictions
        # -----------------------------------------

        y_true, y_pred = evaluate_predictions(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=max_batches,
        )

        labels = list(range(num_classes))

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        class_names = {
            v: k for k, v in dataset.class_mapping.items()
        }

        target_names = [
            class_names.get(i, str(i))
            for i in labels
        ]

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            digits=3,
        )

        # -----------------------------------------
        # Save best model
        # -----------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "val_confusion_matrix": cm,
                    "val_report": report,
                },
                save_path
            )

            print(f"✅ Saved best model -> {save_path}")

            # Save reports
            reports_dir = models_dir / "stgcn_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_path = (
                reports_dir /
                f"best_epoch_{epoch:03d}_report.txt"
            )

            cm_path = (
                reports_dir /
                f"best_epoch_{epoch:03d}_cm.csv"
            )

            report_path.write_text(
                report,
                encoding="utf-8"
            )

            import numpy as np

            np.savetxt(
                cm_path,
                cm,
                delimiter=",",
                fmt="%d"
            )

            print("Validation classification report:")
            print(report)

    print("\nTraining Complete")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0)

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        max_batches=args.max_batches or None
    )