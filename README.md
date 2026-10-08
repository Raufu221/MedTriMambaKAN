# MedTriMambaKAN

**A Gated Multi-Pathway Network for Brain Tumor MRI Classification with Patient-Level and External Evaluation**

Farhan Shahriar, Raufu Umme Rhidi, Ferdaus Anam Jibon
Department of Computer Science and Engineering, International University of Business Agriculture and Technology (IUBAT), Dhaka, Bangladesh

📄 **Paper:** under review at the *International Journal of Medical Informatics* (link will be added upon publication)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hx971LELs2FLIzctMLZ_4L5vXltRIt_r)

---

## 🧠 Overview

MedTriMambaKAN is a four-class brain MRI classifier (**glioma, meningioma, pituitary tumor, no tumor**) that combines three feature pathways on a shared convolutional backbone:

| Pathway | Input | Role |
|---|---|---|
| **CNN pathway** | Deepest feature map (256 × 14 × 14) | Global convolutional features |
| **Sequential-context pathway** (Mamba-inspired) | 196 spatial tokens + class token | Token mixing with gated depthwise convolution |
| **Nonlinear pathway** (KAN-inspired) | Mid-level feature map (128 × 56 × 56) | Gaussian-basis KAN-style layers |

A learnable **fusion gate** weights the three pathways for each image. The model has **2.28 M parameters**.

![Architecture](figures/Fig2_architecture.png)

### Evaluation framework
The study focuses on **leakage-controlled and external evaluation**:
- ✅ Patient-level splits: no patient appears in more than one set
- ✅ Five-fold patient-grouped cross-validation
- ✅ Held-out internal test set (53 patients) with image- and patient-level metrics
- ✅ Five comparison models trained under an identical protocol
- ✅ Pathway ablation and fusion-gate analysis
- ✅ External testing **without fine-tuning** on two collections
- ✅ Grad-CAM++ agreement with radiologist tumor masks, calibration and t-SNE analysis

---

## 📓 View the Notebook in Google Colab

The complete pipeline notebook includes **all original outputs** from the Kaggle runs (2× NVIDIA T4). It is hosted on Google Drive because of its size (37 MB).

| Notebook | Contents | Open |
|---|---|---|
| `MedTriMambaKAN_final.ipynb` | Data preparation, patient-level splits, training, cross-validation, ablation, internal and external evaluation, Grad-CAM++, calibration, t-SNE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hx971LELs2FLIzctMLZ_4L5vXltRIt_r) |

> **Note:** The notebook was developed and run on Kaggle. Dataset paths in the configuration cell point to Kaggle inputs; update them before re-running on Colab or locally. Full re-training requires a GPU.

---

## 📊 Key Results

### Internal and external performance

| Evaluation | Accuracy (%) | Macro-F1 (%) | Macro-AUC |
|---|---|---|---|
| 5-fold cross-validation (mean ± SD) | 95.81 ± 0.95 | 94.56 ± 1.19 | 0.9917 |
| Internal test, image level (713 images) | 91.73 | 90.29 | 0.987 |
| Internal test, patient level (53 patients) | 92.45 (49/53) | 91.53 | 0.990 |
| BRISC, external (5,950 images) | 85.82 | 84.11 | 0.975 |
| Mendeley, external (8,711 images) | 72.83 | 63.62 | 0.895 |

### Comparison on the internal test set

| Model | Image accuracy (%) | Image macro-F1 (%) | Patients correct |
|---|---|---|---|
| **MedTriMambaKAN** | **91.73** | **90.29** | **49/53** |
| MedKAN\* | 87.38 | 85.73 | 45/53 |
| MedMamba\* | 86.82 | 85.45 | 45/53 |
| ResNet-50† | 86.82 | 85.20 | 46/53 |
| EfficientNet-B0† | 86.40 | 84.92 | 47/53 |
| VMamba-Tiny\* | 84.29 | 82.91 | 45/53 |

\*Simplified re-implementations. †ImageNet-pretrained.

### ⚠️ Important finding: data-source shortcut
On external data, errors were concentrated in the **no-tumor class** (recall: 94.4% internal → **55.5%** BRISC → **9.2%** Mendeley). All non-tumor training images came from a single source (IXI), and feature-space analysis indicated that the model partly learned to recognize the **data source** rather than the absence of a tumor.

> This model is a **research prototype** and is **not intended for clinical use**. See the paper for the proposed clinical translation roadmap.

---

## 🗂️ Datasets

All datasets are publicly available.

| Dataset | Used for | Source |
|---|---|---|
| Figshare brain tumor dataset (Cheng, 2017) | Tumor classes: 3,064 slices, 233 patients | [doi:10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) |
| IXI T1 (pre-sliced) | No-tumor class: 115 subjects × 13 slices | [Kaggle](https://www.kaggle.com/datasets/haonanzhou1/ixi-t1) · [IXI original](https://brain-development.org/ixi-dataset/) |
| BRISC 2025 (Fateh et al., 2026) | External test + tumor masks | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) · [Paper](https://doi.org/10.1038/s41597-026-06753-y) |
| Mendeley (Hira et al., 2025) | External test | [doi:10.17632/zwr4ntf94j.1](https://doi.org/10.17632/zwr4ntf94j.1) |

### Data split
| Set | Patients | Images |
|---|---|---|
| Development (5-fold CV) | 295 | 3,846 |
| Internal test (held out) | 53 | 713 |
| **Total** | **348** | **4,559** |

---

## ⚙️ Training Setup

| Setting | Value |
|---|---|
| Input | 224 × 224 grayscale, replicated to 3 channels |
| Optimizer | AdamW (lr 1e-4, weight decay 1e-2) |
| Scheduler | Cosine annealing (80 epochs, min lr 1e-6) |
| Loss | Class-weighted focal loss + label smoothing (0.1) |
| Augmentation | Affine, flips, intensity jitter, blur, MixUp / CutMix |
| Other | Batch 32, early stopping (patience 15), SWA from epoch 60, AMP |
| Hardware | 2× NVIDIA T4 (Kaggle), PyTorch 2.10 |

---

## 🚀 How to Run

1. Open the notebook with the **Open in Colab** badge above, or upload it to **Kaggle**.
2. Add the four datasets listed above as inputs.
3. Update the dataset paths in the configuration cell.
4. Select a GPU runtime and run the cells in order.

### Requirements
```
torch>=2.1
torchvision
timm
einops
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
opencv-python
Pillow
tqdm
```

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{shahriar2026medtrimambakan,
  title   = {MedTriMambaKAN: A Gated Multi-Pathway Network for Brain Tumor MRI
             Classification with Patient-Level and External Evaluation},
  author  = {Shahriar, Farhan and Rhidi, Raufu Umme and Jibon, Ferdaus Anam},
  journal = {Under review},
  year    = {2026}
}
```

---

## 📄 License

Code is released under the **MIT License**. The datasets remain under their original licenses.

## 📬 Contact

- **Ferdaus Anam Jibon** (corresponding author): jibon.cse@iubat.edu
- **Farhan Shahriar**: farhan.rahat.sr27@gmail.com
- **Raufu Umme Rhidi**: rhidiraufu24@gmail.com
