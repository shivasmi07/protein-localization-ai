import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ------------------------
# CONFIG
# ------------------------
CSV_TEST   = "test.csv"
CHECKPOINT = "checkpoints_20_swinv2/ckpt_e20.pth"
MODEL_NAME = "swinv2_small_window8_256"
BATCH_SIZE = 12
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD  = 0.5

# ------------------------
# Dataset
# ------------------------
class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_col  = "file"
        self.label_cols = self.data.columns[4:]
        self.transform  = transform

        for col in self.label_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data[self.label_cols] = self.data[self.label_cols].fillna(0).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row[self.image_col]
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform:
            image = self.transform(image)

        labels = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return image, labels

# ------------------------
# Transforms & Dataloader
# ------------------------
val_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

dataset = MultiLabelImageDataset(CSV_TEST, transform=val_tfms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

# ------------------------
# Load Model
# ------------------------
print(f"[INFO] Loading model: {MODEL_NAME}")
base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=19, global_pool='avg')
model = nn.Sequential(base_model, nn.Sigmoid())  # same as during training

# Load checkpoint safely on CPU
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
state_dict = checkpoint["model_state_dict"]

# Handle 'module.' prefix from DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_k = k[len("module."):]  # strip 'module.' prefix
    else:
        new_k = k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)
model = model.to(DEVICE)
model.eval()

# ------------------------
# Evaluation
# ------------------------
print("[INFO] Running inference on test set...")
all_probs = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(DEVICE)
        outputs = model(images).cpu().numpy()
        labels = labels.numpy()

        all_probs.append(outputs)
        all_targets.append(labels)

all_probs = np.vstack(all_probs)
all_targets = np.vstack(all_targets)
bin_preds = (all_probs > THRESHOLD).astype(int)

# ------------------------
# Metrics
# ------------------------
label_names = dataset.label_cols.tolist()

# Per-class F1
f1_per_class = f1_score(all_targets, bin_preds, average=None, zero_division=0)

# Per-class mAP
aps = []
for i in range(len(label_names)):
    ap = average_precision_score(all_targets[:, i], all_probs[:, i])
    aps.append(ap)

# Overall metrics
f1_macro = f1_score(all_targets, bin_preds, average='macro', zero_division=0)
f1_micro = f1_score(all_targets, bin_preds, average='micro', zero_division=0)
mAP_overall = average_precision_score(all_targets, all_probs, average='macro')

# ------------------------
# Output Table
# ------------------------
df_metrics = pd.DataFrame({
    "Label": label_names,
    "F1 Score": f1_per_class,
    "mAP": aps
})

print("\n=== Per-Class Metrics (Sorted by F1) ===")
print(df_metrics.sort_values("F1 Score", ascending=False).to_string(index=False))

# Save per-class table
df_metrics.to_csv("per_class_f1_map_test.csv", index=False)
print("\n[✔] Saved per-class F1 and mAP to per_class_f1_map_test.csv")

# Print overall
print("\n=== Overall Test Set Metrics ===")
print(f"F1-Macro : {f1_macro:.4f}")
print(f"F1-Micro : {f1_micro:.4f}")
print(f"mAP      : {mAP_overall:.4f}")
