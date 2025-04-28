"""Utility helpers for Nucleus‑Shape Detective – **v3**.

**Fixes**: smarter fallback when the *data* folder lives one level *above* the
current working directory (the common "I executed from `src/`" slip‑up).  Now
`NucleusDataset` tries, in order:

1. `img_dir / filename` – what you specified.
2. `img_dir.parent / filename` – one level up.
3. `csv_basedir / filename` – sibling of the directory that holds
   `annotations/labels.csv` (i.e. project‑root `/data`).

If the image is still missing, you finally get a clear error message that
lists every path it tried.
"""

from __future__ import annotations

import pathlib
import time
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image

__all__ = ["NucleusDataset", "evaluate", "train_loop"]


class NucleusDataset(torch.utils.data.Dataset):
    """Torch dataset that reads microscope images listed in a CSV.

    Designed to be **location‑robust** – you can run scripts from repo root,
    `src/`, notebooks, wherever.
    """

    def __init__(
        self,
        img_dir: str | pathlib.Path,
        csv_path: str | pathlib.Path,
        transform=None,
        *,
        min_quality: int = 0,
        drop_flags: Sequence[str] | None = ("foreign", "part"),
    ) -> None:
        self.img_dir = pathlib.Path(img_dir).expanduser().resolve()
        self.csv_path = pathlib.Path(csv_path).expanduser().resolve()
        self.csv_basedir = self.csv_path.parent.parent.resolve()  # .../data

        df = pd.read_csv(self.csv_path)

        if min_quality > 0 and "quality" in df.columns:
            df = df[df["quality"] >= min_quality]

        if drop_flags and "flags" in df.columns:
            blocked = {f.lower() for f in drop_flags}
            df = df[df["flags"].apply(lambda cell: pd.isna(cell) or blocked.isdisjoint({t.strip().lower() for t in str(cell).split(",")}))]

        if df.empty:
            raise ValueError("No samples left after filtering – check your criteria!")

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = self.df["label"].astype(int).values

    # ----------------------------- helpers ------------------------------
    def _resolve_path(self, filename: str) -> pathlib.Path:
        """Return first existing path of the candidate locations."""
        filename = filename.strip().lstrip("/")
        trials = [
            self.img_dir / filename,
            self.img_dir.parent / filename,
            self.csv_basedir / filename,
        ]
        for p in trials:
            if p.is_file():
                return p
        # If we get here, report all tried paths for easier debugging
        tried = "\n  - " + "\n  - ".join(map(str, trials))
        raise FileNotFoundError(f"Image '{filename}' not found. Tried:{tried}")

    # --------------------------- dataset API ---------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row["filename"]))
        img = read_image(str(img_path)).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


# --------------------------- eval + train ----------------------------
@torch.inference_mode()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def train_loop(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)

        train_acc = evaluate(model, train_dl, device)
        val_acc = evaluate(model, val_dl, device)
        print(
            f"[{time.strftime('%H:%M:%S')}] Ep {ep:02d}/{epochs} | "
            f"loss {run_loss/len(train_dl.dataset):.4f} | "
            f"train {train_acc:.3f} | val {val_acc:.3f}",
            flush=True,
        )

