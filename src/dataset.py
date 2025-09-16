from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.image_col = "file"  # column with image paths
        self.label_cols = self.data.columns[4:]  # assuming first 4 cols are metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_path = row[self.image_col]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Load labels
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))

        return image, labels
