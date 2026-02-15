# Amharic Character Classifier

A single notebook that walks through building and comparing Convolutional Neural Networks for the 237-class Amharic character recognition task using both TensorFlow and PyTorch.

## Overview
- Demonstrates an end-to-end workflow: data preparation, modeling, training, and evaluation of both frameworks.
- Provides insight into how TensorFlow and PyTorch handle the same dataset and architecture, highlighting minor differences in their training behavior.
- Records metrics beyond accuracy (Macro / Weighted F1) and surfaces per-class performance to reveal imbalanced or visually confusing character groups.

## Data
| Attribute | Details |
|-----------|---------|
| Instances | 37,652 grayscale character images |
| Classes | 237 unique Amharic characters |
| Input size | 64 × 64 pixels |
| Normalization | Pixel values scaled to the [0, 1] range |
| Train / Validation / Test | 70% / 15% / 15% |

The notebook expects a pre-extracted dataset folder. The original archive is stored on Google Drive as `uni_dataset.rar` and is extracted into `/content/uni_dataset/` inside Colab.

## Pipeline Highlights
1. Mount Google Drive and install `unrar` to extract the dataset (see the first cells in the notebook).
2. Standardize the images into tensors/arrays for TensorFlow and PyTorch loaders.
3. Define matching CNNs in each framework with convolutional layers, ReLU activations, max pooling, dense blocks, and a softmax output layer for 237 classes.
4. Train both models, track loss/accuracy curves, and compute confusion-aware metrics such as Macro F1 and Weighted F1 to gauge performance on rare characters.
5. Visualize class-wise performance to expose difficult characters, overfitting trends, and potential data imbalance issues.

## Running the Notebook
```
# 1. Open the notebook in Colab via the badge link at the top.
# 2. Mount your Google Drive (the code already does this).
# 3. Install unrar:
!apt-get install unrar
# 4. Extract the dataset (adjust path if your .rar is elsewhere):
!unrar x "/content/drive/MyDrive/Colab Notebooks/uni_dataset.rar" "/content/uni_dataset/"
# 5. Run from the top, inspect TensorFlow and PyTorch sections as desired.
```

If you prefer to run locally, install the dependencies listed in the third section of the notebook (`torch`, `tensorflow`, `numpy`, `pandas`, etc.), point the dataset paths to your own storage, and execute the notebook in your favorite Jupyter environment.

## Results Snapshot
| Metric | TensorFlow | PyTorch |
|--------|------------|---------|
| Test Accuracy | 73.96% | 71.29% |
| Macro F1 | 0.67 | 0.65 |
| Weighted F1 | 0.74 | 0.71 |

TensorFlow exhibits slightly better generalization across all metrics. Both models still show overfitting: validation accuracy plateaus early and validation loss climbs after epoch 5 (TensorFlow) / epoch 7 (PyTorch).

## Notebook Contents
- `amharic_chartr_classifier1.ipynb`: Colab-ready notebook containing all data prep, TensorFlow and PyTorch modeling, evaluation, and visualization steps. The first cells focus on drive mounting and dataset unpacking, while the bulk builds the learning pipeline.

