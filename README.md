# -----------------------------------------------------------------------------
# Project: Nucleus‑Shape Detective – repo skeleton
# -----------------------------------------------------------------------------
# Directory layout (keep this comment at the top of the repo root so GitHub
# shows it first – delete lines you don’t need later):
#
# nucleus_shape_detective/
# ├── README.md
# ├── requirements.txt
# ├── data/
# │   ├── raw/            # original images downloaded from source
# │   ├── processed/      # resized / cleaned images
# │   └── annotations/    # CSV or COCO JSON with filename,label
# ├── notebooks/          # Jupyter experiments
# │   └── 01_exploration.ipynb
# ├── src/
# │   ├── __init__.py
# │   ├── train.py        # model training entry‑point
# │   ├── inference.py    # single‑image predict function
# │   └── utils.py        # helpers (data loaders, metrics)
# ├── app/
# │   ├── app.py          # Streamlit web app
# │   └── requirements.txt (optional separate for Streamlit Cloud)
# ├── tests/              # pytest unit tests
# │   └── test_utils.py
# ├── Dockerfile          # reproducible environment
# └── .github/workflows/
#     └── ci.yml          # GitHub Actions – lint + unit tests on push
# -----------------------------------------------------------------------------
# Below are minimal starter files with TODO markers so you can run
# `python src/train.py` and get an initial model within an hour.
# -----------------------------------------------------------------------------

# ============================ README.md ============================
"""
# Nucleus‑Shape Detective

Detects abnormal nuclear blebs in fluorescence‑microscopy images using a
lightweight convolutional neural network.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py  # train baseline model
streamlit run app/app.py  # launch demo
```

## Data
Place raw TIFF/PNG microscopy images in `data/raw/`. Add a file
`data/annotations/labels.csv` with two columns:
```
filename,label
img_0001.png,normal
img_0002.png,bleb
```

## Model
Baseline: ResNet‑18 (torchvision) fine‑tuned for 2 classes. Images are resized
to 224 × 224 and histogram‑equalised.

## TODO
* Improve augmentation (elastic deform, CLAHE)
* Try EfficientNet‑B0
* Build live‑camera inference pipeline
```
"""
# ========================= requirements.txt =========================
# torch>=2.2
# torchvision
# pandas
# scikit-learn
# matplotlib
# albumentations
# streamlit
# pillow
# pytest
# -------------------------------------------------------------------

# ============================= src/train.py ========================
"""Train a simple CNN to classify nuclei as normal or blebbed."""
import pathlib, argparse
from sklearn.model_selection import train_test_split
import torch, torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils import NucleusDataset, train_loop, evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def get_dataloaders(img_dir, csv_path, batch_size=32):
    tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    ds = NucleusDataset(img_dir, csv_path, tfms)
    train_idx, val_idx = train_test_split(range(len(ds)), test_size=0.2, stratify=ds.labels)
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size)
    )


def main(args):
    train_dl, val_dl = get_dataloaders(args.img_dir, args.csv)
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    train_loop(model, train_dl, val_dl, epochs=args.epochs, lr=1e-4)
    torch.save(model.state_dict(), ROOT / "model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="data/processed")
    p.add_argument("--csv", default="data/annotations/labels.csv")
    p.add_argument("--epochs", type=int, default=5)
    main(p.parse_args())

# ========================= src/inference.py ========================
"""CLI helper to classify a single image."""
import torch, torchvision, sys
from PIL import Image
from torchvision import transforms as T

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(sys.argv[1])
probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
print({"normal": float(probs[0]), "bleb": float(probs[1])})

# ============================== utils.py ===========================
import pandas as pd, torch
from torchvision.io import read_image

class NucleusDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = pathlib.Path(img_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.labels = self.df["label"].map({"normal": 0, "bleb": 1}).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = read_image(str(self.img_dir / self.df.iloc[idx, 0])).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ========================= app/app.py ==============================
"""Streamlit front‑end for drag‑and‑drop nucleus classification."""
import streamlit as st, torch, torchvision
from PIL import Image
from torchvision import transforms as T

st.set_page_config(page_title="Nucleus‑Shape Detective")
st.title("Nucleus‑Shape Detective")

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("Upload a nucleus image", type=["png", "jpg", "tif", "tiff"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Input image", width=256)
    probs = torch.softmax(model(preprocess(img).unsqueeze(0)), dim=1)[0]
    label = "bleb" if probs[1] > probs[0] else "normal"
    st.write(f"**Prediction:** {label} (confidence {float(probs.max()):.2f})")

# ========================= Dockerfile ==============================
# FROM python:3.11-slim
# WORKDIR /code
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# CMD ["streamlit", "run", "app/app.py"]

# ====================== .github/workflows/ci.yml ===================
# name: CI
# on: [push]
# jobs:
#   test:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v3
#       - uses: actions/setup-python@v4
#         with:
#           python-version: '3.11'
#       - name: Install dependencies
#         run: pip install -r requirements.txt
#       - name: Run tests
#         run: pytest
# -------------------------------------------------------------------

