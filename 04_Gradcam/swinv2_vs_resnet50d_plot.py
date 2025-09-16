import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIG ---
CSV_PATH = "swin_tp_missed_by_resnet50d.csv"  # Or your filtered file
OUTPUT_FILE = "swinv2_vsresneST50D_grid_unique_labels.png"
MAX_ROWS = 5  # Set total samples to plot
SAMPLES_PER_ROW = 2  # Swin vs ResNeSt
IMAGE_SIZE = (128, 128)

# Columns must include: file, labels_swin_tp_resnet_fn
MODEL_A_NAME = "SwinV2 Small (window8_256)"
MODEL_B_NAME = "ResNeSt-50d_4s2x40d"

# --- Load Data ---
df = pd.read_csv(CSV_PATH)
df = df.head(MAX_ROWS)  # Limit to MAX_ROWS

fig, axes = plt.subplots(nrows=MAX_ROWS, ncols=SAMPLES_PER_ROW, figsize=(SAMPLES_PER_ROW*3, MAX_ROWS*3))
fig.subplots_adjust(hspace=0.4)

# --- Plot ---
for idx, row in enumerate(df.itertuples()):
    # Load the same image twice (just to represent A and B models)
    img_path = getattr(row, "file")
    label_str = getattr(row, "labels_swin_tp_resnet_fn", "")
    
    try:
        img = Image.open(img_path).resize(IMAGE_SIZE)
    except Exception as e:
        print(f"[!] Error loading {img_path} — skipping")
        continue

    for col in range(SAMPLES_PER_ROW):
        ax = axes[idx, col]
        ax.imshow(img)
        ax.axis('off')

        if col == 0:
            ax.set_title(MODEL_A_NAME, fontsize=8)
            ax.set_xlabel(f"detected:\n{label_str}", fontsize=7)
        else:
            ax.set_title(MODEL_B_NAME, fontsize=8)
            ax.set_xlabel(f"missed:\n{label_str}", fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"[✓] Saved grid: {OUTPUT_FILE}")
plt.show()
