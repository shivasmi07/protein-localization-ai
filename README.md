# 🧬 Deep Learning Approaches for Protein Subcellular Localization

This repository contains the code, experiments, and visualizations from my MSc Dissertation at the **[University of Surrey](https://www.surrey.ac.uk/)**, where I compared different deep learning architectures for **multi-label protein subcellular localization** using the **[Human Protein Atlas (HPA)](https://www.proteinatlas.org/)** dataset.

---

## 📌 About the Project
Proteins rarely function in isolation — their **location inside a cell** (nucleus, cytoplasm, mitochondria, etc.) often determines their role.  
This project explores how deep learning can automatically predict these **subcellular localizations** from microscopy images.  

My focus was on evaluating which architectures are best suited for this task:
- **ResNet-50** (baseline CNN, [He et al., 2016](https://arxiv.org/abs/1512.03385))  
- **ResNeSt-50d_4s2x40d** (split-attention CNN, [Zhang et al., 2020](https://arxiv.org/abs/2004.08955))  
- **Swin Transformer V2 Small** (hierarchical vision transformer, [Liu et al., 2022](https://arxiv.org/abs/2111.09883))  

### Key Experiments
- **Step 1 – Model comparison**: We first trained and evaluated three architectures — **ResNet-50, ResNeSt-50d, and SwinV2 Small** — using standard evaluation metrics: **F1-Micro, F1-Macro, and mAP**. This helped establish which model performed best overall.  

- **Step 2 – Best model analysis**: SwinV2 Small achieved the highest performance across all metrics, so we selected it for deeper analysis. We then performed a **per-class F1 evaluation**, which showed how well the model handled both common compartments (like nucleoplasm, cytosol) and rare compartments (such as aggresomes and mitotic spindle).  

- **Step 3 – Threshold-based visualization**: Using a probability cutoff of **0.5**, we generated **Grad-CAM visualizations** to interpret how the models attended to cellular structures.  

- **Step 4 – ΔF1 comparison with ResNeSt**: To understand relative strengths, we directly compared **SwinV2 vs ResNeSt** by looking at **ΔF1 (performance difference per class)** and model confidence scores.  
  - SwinV2 consistently demonstrated **higher recall and confidence**, particularly for rare or subtle compartments.  
  - Even when the two models had similar F1 scores, SwinV2 tended to rank correct labels higher in probability, which led to superior **mAP**.  

- **Step 5 – Interpretability**: Grad-CAM confirmed that SwinV2 focused on biologically meaningful regions of the cell, validating not just its numerical performance but also the relevance of its attention patterns.  

---

## 🗂 Repository Structure
01_Preprocessing_Data_splitting/

├── EDA.ipynb # Exploratory data analysis

├── Step2-Splittingfile.ipynb # Train/val/test split

└── Output/ # Processed CSVs

02_Models/

├── resnet50.py

├── resneST50d_4s2_40d.py

├── swinV2.py

├── test_per_class_F1.py # Evaluate per-class F1 performance

└── Output/ # Training metrics, results

03_Swinv2_vs_ResneST50d/

├── compare_pred.py # Compare SwinV2 vs ResNeSt predictions

├── grid_swin_better.py # Grid visualization of SwinV2 better cases

└── Output/ # Results comparing SwinV2 and ResNeSt

04_Gradcam/

├── gradcam.py # Grad-CAM main script

├── swinV2_better_resneST50d.py # Compare Grad-CAM attention (SwinV2 vs ResNeSt)

├── swinV2_vs_resneST50d_plot.py # Grad-CAM result plots

├── gradcam_grid_corrected2.py # Grid-based visualization of Grad-CAM

└── Output/ # Saved Grad-CAM overlays and plots

Appendix/

├── ResNet-50_val_metrics.png

├── ResNeSt-50d_val_metrics.png

├── SwinV2_val_metrics.png

├── SwinV2_train_val_loss.png

├── plot_validation_metrics.py # Script for plotting validation metrics

├── resnet50_train_val_loss.png

├── resnet50d_train_val_loss.png

└── train_validation_loss_plot.py # Script for plotting train/val loss

Logs/ # Training logs & schedulers

Visualization/

├── model_comparison_plots.py # Script for plotting model performance

├── test_loss_comparison.py # Compare test losses between models

└── test_metrics_comparison.py # Compare test metrics (F1, mAP, etc.)

src/

├── dataset.py # Dataset preparation utilities

├── loss.py # Focal loss and other loss functions

└── model.py # Model setup (ResNet, ResNeSt, SwinV2 wrappers)


---

## 📊 Results Summary
| Model          | F1-Micro | F1-Macro | mAP  |
|----------------|----------|----------|------|
| ResNet-50      | 0.73     | 0.63     | 0.67 |
| ResNeSt-50d    | 0.77     | 0.72     | 0.78 |
| SwinV2 Small   | 0.84     | 0.80     | 0.87 |

- **SwinV2 Small** performed best overall, especially on rare compartments.  
- **Grad-CAM** visualizations confirmed that SwinV2 attended to biologically relevant regions.  

---

## ⚙️ How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing
jupyter notebook 01_Preprocessing_Data_splitting/EDA.ipynb

# 3. Train a model
python 02_Models/resnet50.py
# or
python 02_Models/resneST50d_4s2_40d.py
# or
python 02_Models/swinv2.py
# or
python 02_Models/test_per_class_F1.py

# 4. Generate Grad-CAM visualizations
python 04_Gradcam/gradcam.py
# or
python 04_Gradcam/swinV2_better_resneST50d.py
# or
python 04_Gradcam/swinV2_vs_resneST50d_plot.py

````` 
## Acknowledgements

Supervised at the University of Surrey → 🔗 https://www.surrey.ac.uk/

Dataset: Human Protein Atlas (HPA) → 🔗 https://www.proteinatlas.org/

Pretrained models: timm → 🔗 https://github.com/huggingface/pytorch-image-models

---
## 👤 Author
Shivasmi Sharma
MSc Data Science – University of Surrey

🔗 GitHub: https://github.com/shivasmi07  
🔗 LinkedIn: https://linkedin.com/in/shivasmi-sharma  