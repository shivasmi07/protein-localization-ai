import os, math
import torch, timm, torch.nn as nn
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === CONFIG ===
CSV_TEST = "test.csv"
CHECKPOINT = "checkpoints_20_swinv2/ckpt_e20.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 12
MODEL_NAME = "swinv2_small_window8_256"
GRID_SAVE_PATH = "gradcam_grid_corrected2.png"
IMG_SIZE = 256
N_COLS = 5

# === SAFE RESHAPE for Swin/SwinV2 tokens -> feature map ===
def swin_safe_reshape_transform(tensor):
    """
    Accepts [B, L, C] tokens or [B, C, H, W] tensors.
    Converts tokens to [B, C, H, W] with H=W=sqrt(L), inferred at runtime.
    """
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Unexpected ndim={tensor.ndim}; expected 3 or 4.")
    B, L, C = tensor.shape
    H = W = int(math.sqrt(L))
    if H * W != L:
        raise ValueError(f"Token length {L} is not a perfect square; can't reshape to HxW.")
    return tensor.permute(0, 2, 1).reshape(B, C, H, W)

# === DATASET ===
class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_col = "file"
        self.label_cols = self.data.columns[4:]
        self.transform = transform
        for col in self.label_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data[self.label_cols] = self.data[self.label_cols].fillna(0).astype(np.float32)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row[self.image_col]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        labels = torch.from_numpy(row[self.label_cols].values.astype(np.float32))
        return image, labels

# === TRANSFORMS & LOADER ===
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])
overlay_tfms = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))])

test_dataset = MultiLabelImageDataset(CSV_TEST, transform=val_tfms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

# === MODEL ===
base_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=19, global_pool="avg")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
state_dict = ckpt.get("model_state_dict", ckpt)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
base_model.load_state_dict(state_dict, strict=False)
model = base_model.to(DEVICE).eval()

@torch.no_grad()
def get_top1_images():
    all_probs = []
    for images, _ in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE, non_blocking=True)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    df = pd.read_csv(CSV_TEST)
    label_cols = df.columns[4:]
    pred_cols = pd.DataFrame(all_probs, columns=[f"pred_{c}" for c in label_cols])
    df_out = pd.concat([df.reset_index(drop=True), pred_cols], axis=1)
    top_images = {}
    for label in label_cols:
        top_df = df_out.sort_values(by=f"pred_{label}", ascending=False).head(1)
        top_images[label] = top_df["file"].values[0]
    return top_images

def get_swin_target_layer(m):
    # Last stage's last block; prefer norm2 (standard for CAM)
    if hasattr(m, "stages"):
        last_stage = m.stages[-1]
    elif hasattr(m, "layers"):
        last_stage = m.layers[-1]
    else:
        raise AttributeError("SwinV2 model has neither 'stages' nor 'layers'.")
    block = last_stage.blocks[-1]
    return block.norm2 if hasattr(block, "norm2") else (block.norm1 if hasattr(block, "norm1") else block)

def run_gradcam_and_make_grid(model, top_images_dict):
    cam_outputs = []
    labels = list(top_images_dict.keys())
    label_cols = list(test_dataset.label_cols)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    target_layer = get_swin_target_layer(base_model)

    # IMPORTANT: use our safe reshape (do NOT use swinT_reshape_transform)
    with GradCAM(model=model, target_layers=[target_layer],
                 reshape_transform=swin_safe_reshape_transform) as cam:
        for label in tqdm(labels, desc="Generating Grad-CAM"):
            img_path = top_images_dict[label]
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                pil_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

            input_tensor = val_tfms(pil_img).unsqueeze(0).to(DEVICE)
            rgb_img = np.asarray(overlay_tfms(pil_img)).astype(np.float32) / 255.0

            class_index = label_cols.index(label)
            targets = [ClassifierOutputTarget(class_index)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            vis_pil = Image.fromarray(vis)
            ImageDraw.Draw(vis_pil).text((6, 6), label, fill=(255, 255, 255), font=font)
            cam_outputs.append(vis_pil)

    # === Grid ===
    n_cols = N_COLS
    n_rows = (len(cam_outputs) + n_cols - 1) // n_cols
    cell = IMG_SIZE
    grid_img = Image.new("RGB", (n_cols * cell, n_rows * cell), (0, 0, 0))
    for idx, img in enumerate(cam_outputs):
        r, c = divmod(idx, n_cols)
        grid_img.paste(img, (c * cell, r * cell))
    grid_img.save(GRID_SAVE_PATH)
    print(f"✅ Grad-CAM grid saved as: {GRID_SAVE_PATH}")

@torch.no_grad()
def get_top1_images_unique():
    all_probs = []
    for images, _ in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE, non_blocking=True)
        probs = torch.sigmoid(model(images)).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    
    df = pd.read_csv(CSV_TEST)
    label_cols = df.columns[4:]
    pred_cols = pd.DataFrame(all_probs, columns=[f"pred_{c}" for c in label_cols])
    df_out = pd.concat([df.reset_index(drop=True), pred_cols], axis=1)
    
    used_images = set()
    top_images = {}

    for label in label_cols:
        sorted_df = df_out.sort_values(by=f"pred_{label}", ascending=False)
        for _, row in sorted_df.iterrows():
            candidate_path = row["file"]
            if candidate_path not in used_images:
                top_images[label] = candidate_path
                used_images.add(candidate_path)
                break
        else:
            print(f"⚠️ No unique image found for label: {label}")
            top_images[label] = sorted_df.iloc[0]["file"]  # fallback

    return top_images
def detect_duplicate_top_images(top_images_dict):
    from collections import defaultdict

    reverse_lookup = defaultdict(list)
    for label, path in top_images_dict.items():
        reverse_lookup[path].append(label)

    print("\n🔍 Duplicate image usage across labels:")
    found = False
    for path, labels in reverse_lookup.items():
        if len(labels) > 1:
            found = True
            print(f"🟠 {os.path.basename(path)} is used for labels: {', '.join(labels)}")
    
    if not found:
        print("✅ No duplicates found. Each label has a unique top image.")

if __name__ == "__main__":
    top_images_for_labels = get_top1_images_unique()  # <-- updated function
    detect_duplicate_top_images(top_images_for_labels)
    run_gradcam_and_make_grid(model, top_images_for_labels)
