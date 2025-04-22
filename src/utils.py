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
