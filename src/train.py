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
