import torch

from src.models.dataset import ProGaitDataset, pad_collate_fn
from src.models.model_stgcn import GaitSTGCN


def test_stgcn_forward_shape():
    ds = ProGaitDataset()
    x, _, _, _ = ds[0]
    model = GaitSTGCN(num_joints=x.shape[1] // 2, num_classes=len(ds.class_mapping))
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
    assert logits.shape == (1, len(ds.class_mapping))


def test_stgcn_can_overfit_tiny_batch():
    # This is a quick learning sanity check: the model should easily overfit 8 samples.
    ds = ProGaitDataset()
    model = GaitSTGCN(num_joints=13, num_classes=len(ds.class_mapping))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(min(8, len(ds))))),
        batch_size=4,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(20):
        for keypoints, _, _, issues, lengths in loader:
            opt.zero_grad()
            logits = model(keypoints, lengths=lengths)
            loss = criterion(logits, issues)
            loss.backward()
            opt.step()

    # Evaluate on same data
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for keypoints, _, _, issues, lengths in loader:
            preds = model(keypoints, lengths=lengths).argmax(1)
            correct += (preds == issues).sum().item()
            total += issues.numel()

    assert total > 0
    assert correct / total > 0.6, f"Expected to overfit tiny batch, got acc={correct/total:.2f}"
