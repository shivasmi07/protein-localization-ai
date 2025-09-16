import os
import multiprocessing
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score
import timm

# =============================================================
# CONFIGURATION
# =============================================================
CSV_TRAIN      = "train.csv"
CSV_VAL        = "val.csv"
CSV_TEST       = "test.csv"
CHECKPOINT_DIR = "checkpoints_20_swinv2"
METRICS_CSV    = "training_metrics_20_swinv2.csv"
NUM_EPOCHS     = 20
BATCH_SIZE     = 12  # Adjusted for SwinV2 memory
NUM_WORKERS    = min(32, multiprocessing.cpu_count())
DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THRESHOLD      = 0.5
MODEL_NAME = "swinv2_small_window8_256"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.cuda.empty_cache()

# =============================================================
# DATASET
# =============================================================
class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file: str, transform=None):
        print(f"[INFO] Reading {csv_file}")
        self.data        = pd.read_csv(csv_file)
        self.image_col   = "file"
        self.label_cols  = self.data.columns[4:]
        self.transform   = transform

        for col in self.label_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data[self.label_cols] = self.data[self.label_cols].fillna(0).astype(np.float32)

        print(f"[INFO] {len(self.data)} samples | label cols: {list(self.label_cols)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row[self.image_col]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}. Using blank image.")
            image = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform:
            image = self.transform(image)

        labels = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return image, labels

# =============================================================
# TRANSFORMS & DATALOADERS
# =============================================================
train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

datasets = {
    "train": MultiLabelImageDataset(CSV_TRAIN, transform=train_tfms),
    "val":   MultiLabelImageDataset(CSV_VAL,   transform=val_tfms),
    "test":  MultiLabelImageDataset(CSV_TEST,  transform=val_tfms),
}

dataloaders = {
    split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                      num_workers=NUM_WORKERS, pin_memory=True)
    for split, ds in datasets.items()
}

# =============================================================
# MODEL
# =============================================================
print(f"[INFO] Loading model: {MODEL_NAME}")
base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=19, global_pool='avg')
model = nn.Sequential(
    base_model,
    nn.Sigmoid(),
)

if torch.cuda.device_count() > 1:
    print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)
model = model.to(DEVICE)

# =============================================================
# LOSS, OPTIMIZER, SCHEDULER
# =============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        bce  = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# =============================================================
# METRIC HELPERS
# =============================================================
def compute_metrics(prob: np.ndarray, target: np.ndarray):
    bin_pred = (prob > THRESHOLD).astype(int)
    f1_macro = f1_score(target, bin_pred, average="macro", zero_division=0)
    f1_micro = f1_score(target, bin_pred, average="micro", zero_division=0)
    mAP      = average_precision_score(target, prob, average="macro")
    return f1_macro, f1_micro, mAP

# =============================================================
# TRAIN & EVAL FUNCTIONS
# =============================================================
def train_one_epoch(epoch: int):
    model.train()
    running_loss, all_prob, all_tgt = 0.0, [], []
    for images, labels in tqdm(dataloaders["train"], desc=f"Epoch {epoch} [train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_prob.append(outputs.detach().cpu())
        all_tgt.append(labels.detach().cpu())

    avg_loss = running_loss / len(dataloaders["train"])
    f1M, f1micro, mAP = compute_metrics(torch.cat(all_prob).numpy(), torch.cat(all_tgt).numpy())
    print(f"Epoch {epoch} DONE | TrainLoss {avg_loss:.4f} | F1_macro {f1M:.4f} | F1_micro {f1micro:.4f} | mAP {mAP:.4f}")
    return avg_loss, f1M, f1micro, mAP

def evaluate(split: str = "val"):
    model.eval()
    running_loss, all_prob, all_tgt = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloaders[split], desc=f"[{split}]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            all_prob.append(outputs.cpu())
            all_tgt.append(labels.cpu())

    avg_loss = running_loss / len(dataloaders[split])
    f1M, f1micro, mAP = compute_metrics(torch.cat(all_prob).numpy(), torch.cat(all_tgt).numpy())
    print(f"{split.upper()} | Loss {avg_loss:.4f} | F1_macro {f1M:.4f} | F1_micro {f1micro:.4f} | mAP {mAP:.4f}")
    return avg_loss, f1M, f1micro, mAP

# =============================================================
# TRAINING LOOP WITH SCHEDULER
# =============================================================
metrics_log = []

for epoch in range(1, NUM_EPOCHS + 1):
    tl, tf1m, tf1micro, tmap = train_one_epoch(epoch)
    vl, vf1m, vf1micro, vmap = evaluate("val")

    scheduler.step()
    print(f"[INFO] Scheduler stepped. Current LR: {scheduler.get_last_lr()}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_e{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    metrics_log.extend([
        {"epoch": epoch, "split": "train", "loss": tl, "f1_macro": tf1m, "f1_micro": tf1micro, "mAP": tmap},
        {"epoch": epoch, "split": "val",   "loss": vl, "f1_macro": vf1m, "f1_micro": vf1micro, "mAP": vmap},
    ])

# =============================================================
# FINAL TEST EVALUATION
# =============================================================
print("[TEST] evaluating on held‑out set …")

test_loss, test_f1m, test_f1micro, test_map = evaluate("test")
metrics_log.append({
    "epoch": NUM_EPOCHS + 1,
    "split": "test",
    "loss": test_loss,
    "f1_macro": test_f1m,
    "f1_micro": test_f1micro,
    "mAP": test_map,
})

pd.DataFrame(metrics_log).to_csv(METRICS_CSV, index=False)
print(f"All metrics written to {METRICS_CSV}")
