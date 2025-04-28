"""Utility helpers for Nucleus‑Shape Detective.

Includes:
* NucleusDataset – torchvision‑style dataset that reads images from disk.
* evaluate(model, dl, device) – quick accuracy evaluator.
* train_loop(...) – minimal training routine with progress printing.
"""

import pathlib
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image


class NucleusDataset(torch.utils.data.Dataset):
    """Image dataset for nucleus images classified as 'normal' or 'bleb'."""

    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = pathlib.Path(img_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # Map string labels to int 0/1
        self.labels = self.df["label"].map({"normal": 0, "bleb": 1}).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.df.iloc[idx, 0]
        img = read_image(str(img_path)).float() / 255.0  # [0,1] tensor
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: str = "cpu") -> float:
    """Return accuracy of *model* on the images in *dl*."""
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def train_loop(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """Minimal supervised training loop.

    Parameters
    ----------
    model : torch.nn.Module
        The network to train (its final layer should already match the number of
        classes).
    train_dl, val_dl : DataLoader
        Training and validation dataloaders.
    epochs : int
        Total number of passes over the training set.
    lr : float
        Learning rate for Adam.
    device : str
        "cpu" or "cuda".
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)

        train_acc = evaluate(model, train_dl, device)
        val_acc = evaluate(model, val_dl, device)
        avg_loss = running_loss / len(train_dl.dataset)

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {avg_loss:.4f} | "
            f"train acc {train_acc:.3f} | val acc {val_acc:.3f}",
            flush=True,
        )

