# Pneumonia Detection using CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harika-Yendapalli/Pneumonia-Detection-using-CNN/blob/main/Code.ipynb)

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a custom Convolutional Neural Network (CNN) and a transfer-learning model built on **VGG16**.

Built as the final project for **CSCI 167 – Introduction to Deep Learning** (Dr. David Ruby, Fall 2024).

---

## Why this project

Pneumonia is a leading cause of illness and death worldwide, and chest X-rays are the standard way to diagnose it. Reading X-rays by hand is slow and can vary from one radiologist to another. This project explores how a CNN can give fast, consistent predictions that could support doctors in reviewing scans.

## Objectives

- Classify chest X-rays into NORMAL and PNEUMONIA with a deep learning model
- Reduce variability in X-ray interpretation by using a consistent automated approach
- Use preprocessing, data augmentation and class balancing to make the model robust and limit overfitting
- Compare a CNN trained from scratch with a pre-trained VGG16 model

---

## Dataset

**Chest X-Ray Images (Pneumonia)** from Kaggle
[paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Original source: Kermany, D., Zhang, K., Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data, v2.

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 234 | 390 | 624 |
| Test | 8 | 8 | 16 |

> The dataset's original `test` folder (624 images) is used for validation, and its small `val` folder (16 images) is used as the final test set.

The training data is **imbalanced**: pneumonia images outnumber normal images about 3 to 1. Class weights (`{0: 1.945, 1: 0.673}`) were tested to counter this.

---

## Project Pipeline

```
Kaggle dataset ─► Load & explore ─► Preprocess ─► Augment ─► Train CNN / VGG16 ─► Evaluate
```

### 1. Exploratory Data Analysis
- Counted images per class in each split
- Plotted class distribution bar charts
- Displayed sample NORMAL and PNEUMONIA X-rays

### 2. Preprocessing
- Resized all images to **256 × 256**
- Normalized pixel values to **[0, 1]**
- Experimented with **grayscale conversion**, **Gaussian blur** and **morphological erosion** (5×5 kernel) to reduce noise and sharpen lung boundaries
- Labeled images: `0 = NORMAL`, `1 = PNEUMONIA`

### 3. Data Augmentation
Real-time augmentation with Keras `ImageDataGenerator`:
- Rotation (±20°), width/height shift (0.2), shear (0.2), zoom (0.2), horizontal flip, `fill_mode='nearest'`

### 4. Models

**Model 1: Custom CNN**

| Layer | Details |
|---|---|
| Conv2D + MaxPool | 32 filters, 3×3, ReLU, 2×2 pool |
| Conv2D + MaxPool + Dropout | 64 filters, 3×3, ReLU, 2×2 pool, dropout 0.25 |
| Conv2D + MaxPool + Dropout | 128 filters, 3×3, ReLU, 2×2 pool, dropout 0.5 |
| Flatten → Dense + Dropout | 128 units, ReLU, dropout 0.5 |
| Output | 1 unit, Sigmoid |

- ~16.9M trainable parameters
- Loss: binary cross-entropy · Metric: accuracy
- Callbacks: `ReduceLROnPlateau` (on val accuracy) and `EarlyStopping` (on val loss, restores best weights)
- Trained for up to 12 epochs, batch size 8

**Model 2: Transfer Learning with VGG16**
- VGG16 pre-trained on ImageNet, top removed, global max pooling, **base layers frozen**
- Custom head: Dense 128 → 64 → 32 (ReLU) → 1 (Sigmoid)
- Optimizer: Adam · Loss: binary cross-entropy
- Early stopping on validation loss

---

## Results

| Model | Accuracy | Loss |
|---|---:|---:|
| Model 1: Custom CNN | 81.25% | 0.4438 |
| Model 2: VGG16 transfer learning | **87.5%** | **0.3533** |

*Results as reported in the final project report.* The VGG16 model performed better, with higher accuracy, lower loss and more stable validation curves. On the 624-image validation set, the custom CNN reached about 81% accuracy and VGG16 about 86%.

**Confusion matrix (Model 1, test set):** TN = 6, TP = 7, FP = 2, FN = 1

> **Note:** The final test set has only 16 images, so a single misclassified image changes accuracy by 6.25%. The validation-set numbers are a more reliable indicator of real performance.

### Hyperparameter experiments (Model 1)

| Epochs | Kernel | Learning rate | Other | Accuracy | Loss |
|---:|---:|---|---|---:|---:|
| 8 | 3 | 1e-6 | – | 75.0% | 0.3295 |
| 12 | 3 | 1e-6 | – | 87.5% | 0.2740 |
| 12 | 5 | 1e-6 | – | 87.5% | 0.2744 |
| 12 | 3 | 1e-3 | – | 87.5% | 0.2740 |
| 12 | 3 | 1e-6 | No class weights | 87.5% | 0.3340 |
| 12 | 3 | 0.01 | **SGD + Nesterov momentum (0.9)**, no class weights | **93.75%** | **0.1969** |

**Key takeaways**
- More epochs helped with a low learning rate (75% → 87.5% going from 8 to 12 epochs)
- A higher learning rate reached the same accuracy faster
- Kernel size 3 vs 5 made no meaningful difference on this dataset
- **SGD with Nesterov momentum** gave the best result (93.75%, loss 0.1969)

---

## Tech Stack

- **Python**, **TensorFlow / Keras**
- **OpenCV** for image processing
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn** for visualization
- **scikit-learn** for the confusion matrix and metrics
- **Google Colab** (GPU) and **KaggleHub** for the dataset

---

## How to Run

1. Click the **Open in Colab** badge at the top.
2. Set the runtime to GPU: **Runtime → Change runtime type → GPU**.
3. Get your own Kaggle API token: Kaggle → **Settings → API → Create New Token**. This downloads `kaggle.json`.
4. Run the notebook cells in order. When prompted, upload your `kaggle.json`. The dataset (~2.3 GB) downloads automatically through `kagglehub`.

---

## Repository Contents

| File | Description |
|---|---|
| `Code.ipynb` | Full notebook: data loading, EDA, preprocessing, augmentation, both models, evaluation |
| `Final Report_167.docx` | Final project report |
| `Final PPT 167.pptx` | Project presentation slides |
| `Hyperparameters.docx` | Hyperparameter experiments and observations |

---

## Future Work

- Tune normalization using dataset-specific mean and standard deviation
- Try more optimizers (AdamW, SGD) and regularization such as weight decay
- Fine-tune the top VGG16 layers instead of freezing all of them
- Evaluate with **precision, recall, F1-score and ROC-AUC**
- Add **Grad-CAM** heatmaps to show which lung regions drive each prediction
- Evaluate on a larger, held-out test set

---
