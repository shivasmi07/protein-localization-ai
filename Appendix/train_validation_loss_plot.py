import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
resnet_df = pd.read_csv("training_metrics_20.csv")
resnest_df = pd.read_csv("training_metrics_20_resnest50d_4s2x40d.csv")
swin_df = pd.read_csv("training_metrics_20_swinv2.csv")

def plot_loss(df, model_name, save_name):
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange')
    plt.title(f"{model_name}: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300)
    plt.close()

# Save each plot as PNG
plot_loss(resnet_df, "ResNet-50", "resnet50_train_val_loss")
plot_loss(resnest_df, "ResNeSt-50d_4s2x40d", "resnest50d_train_val_loss")
plot_loss(swin_df, "SwinV2 Small", "swinv2_train_val_loss")
