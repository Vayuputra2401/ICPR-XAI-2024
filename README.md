# Enhanced Classification and Segmentation of Brain Tumors in MRI Images Using Custom CNN and U-Net Models with XAI

> **Published at ICPR 2024 (Kolkata)**
> Pathikreet Chowdhury and Gargi Srivastava
> Rajiv Gandhi Institute of Petroleum Technology, Jais, Uttar Pradesh 229304, India

---

## Overview

This repository contains the code and resources for our ICPR 2024 paper on brain tumor classification and segmentation using custom deep learning architectures enhanced with Explainable AI (XAI).

We address two key tasks:
- **Classification** — Categorizing brain MRI images into four classes: *glioma*, *meningioma*, *pituitary tumor*, and *no tumor*
- **Segmentation** — Segmenting low-grade gliomas from FLAIR MRI sequences

To move beyond black-box predictions, we apply **LIME (Local Interpretable Model-Agnostic Explanations)** to validate *why* our custom architectures outperform established baselines like ResNet and VGG.

---

## Key Results

### Classification (Custom CNN vs. ResNet32 vs. VGG16)

| Metric | Custom CNN | ResNet32 | VGG16 |
|---|---|---|---|
| Accuracy | **98.70%** | 96.35% | 96.2% |
| Precision | **97.63** | 95.24 | 95.78 |
| Recall | **97.64** | 96.12 | 95.12 |
| F1-Score | **97.47** | 96.15 | 95.76 |
| LIME Stability Score | **0.923** | 0.846 | 0.687 |
| LIME Sparsity Score | 0.208 | 0.196 | 0.225 |
| LIME Fidelity Score | 0.310 | **0.556** | 0.418 |

### Segmentation (Custom U-Net vs. ResUNet)

| Metric | Custom U-Net | U-Net + ResNet Encoder |
|---|---|---|
| Validation Accuracy | **99.79%** | 99.11% |
| Validation Loss | **0.0132** | 0.0425 |
| IoU Score | **0.889** | 0.847 |
| LIME Stability Score | **0.8169** | 0.7873 |
| LIME Sparsity Score | **0.1190** | 0.1221 |
| LIME Fidelity Score | 0.5447 | **0.6036** |

> The Custom U-Net also achieves a **perfect IoU of 1.00** as a negative classifier — correctly identifying MRI scans with no tumor with zero false positives.

---

## Repository Structure

```
ICPR-XAI-2024/
├── ICPR_Kolkata.pdf                          # Full paper
├── README.md
├── code/
│   ├── brain-tumour-cnn-xai.ipynb            # Custom CNN training + LIME
│   ├── brain-tumour-cnn-xai_cropped.ipynb    # CNN with cropped brain regions
│   ├── brain-tumour-cnn-xai_updated.ipynb    # Updated CNN experiments
│   ├── brain-tumour-resunet-xai-implementation.ipynb  # ResUNet baseline + LIME
│   ├── brain-tumour-unets-xai-implementation.ipynb    # Custom U-Net + LIME
│   ├── lits-dl-architecture-implementation.ipynb      # LiTS dataset experiments
│   └── liver-segmentation-resnet-50.ipynb    # Liver segmentation (ResNet-50)
└── images/
    ├── Confusion Matrix of Custom CNN model.png
    ├── LIME Heatmap Custom U Net.png
    ├── LIME Visualizations of Custom CNN *.png
    ├── LIME Visualizations of ResUNet.png
    ├── MRI Scan on which LIME Visualizations are generated *.png
    └── Plot of ground truth scans, masks and predicted masks *.png
```

---

## Model Architectures

### Custom U-Net (Segmentation)

The proposed Enhanced U-Net retains the canonical encoder-decoder structure with three key improvements:

- **Residual Connections** — Added within each convolutional block to mitigate the vanishing gradient problem and allow deeper networks to train effectively
- **Batch Normalization** — Applied after every convolutional layer to stabilize training, reduce internal covariate shift, and allow higher learning rates
- **Spatial Dropout** — Drops entire feature maps (rather than individual neurons) during training to prevent the model from over-relying on specific features

### Custom CNN (Classification)

A purpose-built CNN for 4-class brain tumor classification:

| Layer | Details |
|---|---|
| Conv Layer 1 | 32 filters, 4×4, stride 1, BatchNorm + ReLU |
| Conv Layer 2 | 64 filters, 4×4, stride 1, BatchNorm + ReLU |
| MaxPool 1 | 3×3, stride 3 (after Conv 1 & 2) |
| Conv Layer 3 | 128 filters, 4×4, stride 1, BatchNorm + ReLU |
| MaxPool 2 | 3×3, stride 2 |
| Conv Layer 4 | 128 filters, 4×4, stride 1, BatchNorm + ReLU |
| FC Layer 1 | 512 units, ReLU |
| Dropout | Rate: 0.5 |
| FC Layer 2 | Output classes (4) |

The use of **two different pooling strides** (3 and 2) is a key design choice that enhances the model's ability to capture multi-scale spatial features.

---

## Datasets

### Classification (7,023 images, 4 classes)

| Source | Details |
|---|---|
| Jun Cheng's Brain Tumor Dataset | 3,064 T1-weighted contrast-enhanced images from 233 patients (meningioma, glioma, pituitary) |
| Br35H Dataset | Brain MRI images across all 4 classes |
| SarTaj Dataset | Additional images for training robustness |

Classes: `glioma`, `meningioma`, `pituitary`, `no tumor`

### Segmentation

- **Brain MRI Segmentation Dataset** from The Cancer Imaging Archive (TCIA)
- Sourced from the TCGA Lower-Grade Glioma collection
- 110 patients with FLAIR sequences and expert-annotated segmentation masks
- Binary masks: `1` = tumor, `0` = healthy tissue

### Preprocessing

- Images resized to **256×256** pixels
- Intensity normalization to `[0, 1]`
- For classification: brain region cropping augmentation
- For segmentation: zero-mean, unit-variance normalization
- Data augmentation: horizontal flips, rotations, elastic deformations, contrast adjustment

---

## Explainability with LIME

LIME is applied to both tasks to explain model decisions locally by generating perturbed samples and fitting an interpretable surrogate model.

**LIME Parameters used:**

| Parameter | Value |
|---|---|
| Number of Samples | 1000 |
| Kernel Width | 0.25 |
| Feature Selection | Forward Selection |
| Regularization | L1 |
| Segmenter | Quickshift (kernel=4, max_dist=200, ratio=0.5) |

**XAI Metrics:**

- **Stability** — Consistency of explanations under slight input perturbations (higher is better)
- **Sparsity** — Proportion of image *not* used in the explanation; sparser = more concise
- **Fidelity** — How well the surrogate model approximates the original model's predictions

---

## Hyperparameters

### Segmentation Model

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 50 |
| Loss Function | Binary Cross-Entropy + Dice |
| Dropout (early layers) | 0.1 |
| Dropout (middle layers) | 0.2 |
| Dropout (final layer) | 0.3 |

### Classification Model

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Epochs | 100 |
| Loss Function | Cross-Entropy |
| Dropout | 0.5 |

---

## Evaluation Metrics

**Segmentation:** Dice Coefficient (DSC), IoU (Jaccard Index), Precision, Recall, F1-Score, Hausdorff Distance

**Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{chowdhury2024xai,
  title     = {Enhanced Classification and Segmentation of Brain Tumors in MRI Images
               Using Custom CNN and U-Net Models with XAI},
  author    = {Chowdhury, Pathikreet and Srivastava, Gargi},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2024},
  address   = {Kolkata, India}
}
```

---

## Future Work

- Integrating **attention mechanisms and transformers** into the classification and segmentation pipelines
- Using **larger and more diverse datasets** for better cross-population generalizability
- Developing **real-time inference** pipelines for clinical deployment
- Applying **advanced XAI methods** (e.g., GradCAM, SHAP) for richer interpretability
- **EHR integration** for automated diagnosis and clinical decision support

---

## Contact

- Pathikreet Chowdhury — `21cs2026@rgipt.ac.in`
- Gargi Srivastava — `gsrivastava@rgipt.ac.in`

Rajiv Gandhi Institute of Petroleum Technology, Jais, Uttar Pradesh 229304, India
[http://www.rgipt.ac.in](http://www.rgipt.ac.in)
