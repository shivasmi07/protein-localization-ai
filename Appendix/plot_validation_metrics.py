import pandas as pd
import matplotlib.pyplot as plt

# Define mapping: file → model name
files = {
    "training_metrics_20.csv": "ResNet-50",
    "training_metrics_20_resnest50d_4s2x40d.csv": "ResNeSt-50d_4s2x40d",
    "training_metrics_20_swinv2.csv": "SwinV2 Small"
}

# Loop over each model's metrics file
for filename, model_name in files.items():
    df = pd.read_csv(filename)
    val_df = df[df['split'] == 'val']

    plt.figure(figsize=(6, 4))
    plt.plot(val_df["epoch"], val_df["f1_micro"], label="F1 Micro", color="blue")
    plt.plot(val_df["epoch"], val_df["f1_macro"], label="F1 Macro", color="orange")
    plt.plot(val_df["epoch"], val_df["mAP"], label="mAP", color="green", linestyle='--', marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name}: Validation F1 & mAP Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot as PNG
    save_name = f"{model_name.replace(' ', '_').replace(':', '')}_val_metrics.png"
    plt.savefig(save_name, dpi=300)
    print(f"[✓] Saved: {save_name}")

    plt.close()
