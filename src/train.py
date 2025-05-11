"""Train a simple CNN to classify nuclei as normal or blebbed."""
import pathlib, argparse
from sklearn.model_selection import train_test_split
import torch, torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils import NucleusDataset, train_loop, evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def get_dataloaders(img_dir, csv_path, batch_size=32):
    # Training transforms with augmentation
    train_tfms = T.Compose([
        T.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x.repeat(3, 1, 1)),
        T.Resize((224, 224), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms (no augmentation)
    val_tfms = T.Compose([
        T.Lambda(lambda x: x[:3] if x.shape[0] == 4 else x.repeat(3, 1, 1)),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_ds = NucleusDataset(img_dir, csv_path, train_tfms)
    val_ds = NucleusDataset(img_dir, csv_path, val_tfms)
    
    # Split training data
    train_idx, val_idx = train_test_split(
        range(len(train_ds)), 
        test_size=0.2, 
        stratify=train_ds.labels,
        random_state=42
    )
    
    train_subset = torch.utils.data.Subset(train_ds, train_idx)
    val_subset = torch.utils.data.Subset(val_ds, val_idx)
    
    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_subset, batch_size=batch_size, num_workers=0)
    )


def main(args):
    train_dl, val_dl = get_dataloaders(args.img_dir, args.csv)
    
    # Initialize model
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Train with adjusted parameters
    train_loop(
        model, 
        train_dl, 
        val_dl, 
        epochs=args.epochs, 
        lr=args.lr,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), ROOT / "model.pt")
    print(f"Model saved to {ROOT / 'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="data")
    p.add_argument("--csv", default="data/annotations/labels.csv")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    main(p.parse_args())
