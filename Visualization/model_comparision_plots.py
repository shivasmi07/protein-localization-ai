import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# ---- STEP 1: Load CSV Files ----
file_resnet   = "training_metrics_20.csv"
file_resnest  = "training_metrics_20_resnest50d_4s2x40d.csv"
file_swinv2   = "training_metrics_20_swinv2.csv"

# Read last row = test performance
resnet_df   = pd.read_csv(file_resnet).tail(1)
resnest_df  = pd.read_csv(file_resnest).tail(1)
swinv2_df   = pd.read_csv(file_swinv2).tail(1)

# ---- STEP 2: Construct Combined DataFrame ----
data = {
    'Model': ['ResNet-50', 'ResNeSt-50d', 'SwinV2 Small'],
    'F1-Macro': [
        resnet_df['f1_macro'].values[0],
        resnest_df['f1_macro'].values[0],
        swinv2_df['f1_macro'].values[0]
    ],
    'F1-Micro': [
        resnet_df['f1_micro'].values[0],
        resnest_df['f1_micro'].values[0],
        swinv2_df['f1_micro'].values[0]
    ],
    'mAP': [
        resnet_df['map'].values[0],
        resnest_df['map'].values[0],
        swinv2_df['map'].values[0]
    ],
    'Loss': [
        resnet_df['loss'].values[0],
        resnest_df['loss'].values[0],
        swinv2_df['loss'].values[0]
    ]
}

df = pd.DataFrame(data)

# ---- STEP 3: Plot Test Metrics ----
df_melted = df.melt(id_vars="Model", value_vars=["F1-Macro", "F1-Micro", "mAP"],
                    var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
barplot.set_title("Test Metrics Comparison by Model", fontsize=14)
barplot.set_ylim(0.5, 1.0)

# Annotate bars
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.3f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)

plt.legend(title='Metric')
plt.savefig("test_metrics_comparison.png")
plt.show()

# ---- STEP 4: Plot Test Loss ----
plt.figure(figsize=(8, 5))
loss_plot = sns.barplot(data=df, x="Model", y="Loss", palette="pastel")
loss_plot.set_title("Test Loss Comparison by Model", fontsize=14)

# Annotate bars
for p in loss_plot.patches:
    loss_plot.annotate(f'{p.get_height():.4f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)

plt.savefig("test_loss_comparison.png")
plt.show()
