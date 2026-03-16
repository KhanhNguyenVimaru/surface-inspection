# Surface Defect Detection on NEU-DET

End-to-end project template for industrial surface defect classification using Computer Vision and Deep Learning.

## Overview
- Task: multi-class image classification.
- Dataset: NEU Surface Defect Database (NEU-DET).
- Classes (6): `Crazing`, `Inclusion`, `Patches`, `Pitted_surface`, `Rolled-in_scale`, `Scratches`.
- Input image size: resized to `224x224`.
- Supported models:
  - `custom_cnn`
  - `resnet18`
  - `mobilenet_v3_small`

## Project Structure
```text
.
├── data/
│   ├── raw/                 # Original NEU-DET data
│   └── processed/           # Split output: train/val/test
├── outputs/
│   ├── checkpoints/         # Saved model checkpoints (.pt)
│   ├── figures/             # Curves and confusion matrices
│   └── reports/             # Metrics JSON files
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── eda.py
├── prepare_data.py
├── train.py
├── evaluate.py
├── predict.py
└── requirements.txt
```

## Setup Guide

### 1. Prerequisites
- Python `3.10+`
- `pip`
- Optional but recommended: NVIDIA GPU + CUDA-enabled PyTorch

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download NEU-DET with `kagglehub`
Install `kagglehub`:
```bash
pip install kagglehub
```

Download dataset:
```bash
python -c "import kagglehub; print(kagglehub.dataset_download('kaustubhdikshit/neu-surface-defect-database'))"
```

Then copy images into this structure:
```text
data/raw/
├── Crazing
├── Inclusion
├── Patches
├── Pitted_surface
├── Rolled-in_scale
└── Scratches
```

### 4. Check dataset and create train/val/test splits
Run EDA:
```bash
python eda.py --data-dir data/raw
```

Split data with ratio `70/15/15`:
```bash
python prepare_data.py --input-dir data/raw --output-dir data/processed --seed 42
```

### 5. Verify GPU availability (optional)
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

## Training
Train with ResNet18 (recommended baseline):
```bash
python train.py \
  --data-dir data/processed \
  --model resnet18 \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.0005
```

Train with custom CNN:
```bash
python train.py \
  --data-dir data/processed \
  --model custom_cnn \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001
```

## Evaluation
```bash
python evaluate.py \
  --data-dir data/processed \
  --checkpoint outputs/checkpoints/best_resnet18.pt \
  --model resnet18
```

## Single Image Inference (Demo)
```bash
python predict.py \
  --image path/to/image.jpg \
  --checkpoint outputs/checkpoints/best_resnet18.pt \
  --model resnet18
```

Example output:
```text
Prediction: Scratches
Confidence: 0.92
```

## Output Artifacts
- `outputs/checkpoints/best_<model>.pt`
- `outputs/figures/curves_<model>.png`
- `outputs/figures/confusion_<model>.png`
- `outputs/reports/metrics_<model>.json`

## Suggested Thesis Flow
1. Problem statement and industrial motivation.
2. Dataset description and EDA.
3. Preprocessing and augmentation strategy.
4. Model design and training setup.
5. Results: accuracy, precision, recall, F1, confusion matrix.
6. Model comparison and discussion.
7. Demo pipeline and future improvements.
