import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Parameters
CSV_FILE = "swin_better_than_resnet50d.csv"  # Replace with your file if different
GRID_ROWS = 4
GRID_COLS = 5
OUTFILE = "swin_better_grid_unique_labels.png"
IMAGE_SIZE = (224, 224)  # Resize to keep things uniform

# Load CSV
df = pd.read_csv(CSV_FILE)
print(f"[INFO] Total entries in CSV: {len(df)}")

# Function to extract label string from filename
def extract_labels_from_path(filepath):
    return os.path.basename(filepath).split("_")[0]  # or use metadata if available

# Keep one image per unique predicted label combo (if known)
unique_rows = []
seen_labels = set()

for _, row in df.iterrows():
    img_path = row["file"]
    label_key = extract_labels_from_path(img_path)  # Replace with true label string if available

    if label_key not in seen_labels and os.path.exists(img_path):
        seen_labels.add(label_key)
        unique_rows.append((img_path, row["delta"]))

    if len(unique_rows) == GRID_ROWS * GRID_COLS:
        break

print(f"[INFO] Selected {len(unique_rows)} unique images.")

# Create grid
fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(20, 16))
axes = axes.flatten()

for ax, (img_path, delta) in zip(axes, unique_rows):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Could not load image: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"ΔF1={delta:.2f}", fontsize=10, color='white', backgroundcolor='red')

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300)
print(f"[✓] Saved grid to {OUTFILE}")
