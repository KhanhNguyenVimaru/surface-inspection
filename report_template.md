# Report Template

## Title
Surface Defect Detection in Industrial Manufacturing Using Deep Learning on NEU-DET

## Abstract
- Problem context in manufacturing.
- Objective of this project.
- Dataset and methods used.
- Main results (ResNet18): best validation accuracy `1.0000`, test accuracy `0.9815`, macro F1 `0.98`.
- Practical contribution and limitations.

## 1. Introduction
### 1.1 Problem Statement
- Describe why surface defect inspection matters.
- Explain limits of manual inspection.

### 1.2 Objectives
- Build an automatic classifier for 6 NEU-DET defect classes.
- Compare CNN and transfer learning models.
- Deliver a simple inference demo.

### 1.3 Scope
- Input: steel surface images.
- Output: one of 6 defect categories.
- Task type: image classification.

## 2. Dataset Description
### 2.1 NEU-DET Overview
- Number of classes: 6
- Images per class: about 300
- Total images: 1800
- Original size: 200x200

### 2.2 Class List
| Code | Class Name |
|---|---|
| Cr | Crazing |
| In | Inclusion |
| Pa | Patches |
| Ps | Pitted_surface |
| Rs | Rolled-in_scale |
| Sc | Scratches |

### 2.3 Data Organization
- Raw folder: `data/raw/`
- Split folder: `data/processed/{train,val,test}`
- Split ratio: 70/15/15

### 2.4 EDA
- Add class count chart from: `outputs/figures/raw_class_distribution.png`
- Comment on data balance.

## 3. Methodology
### 3.1 Preprocessing
- Resize images to 224x224.
- Normalize with ImageNet mean/std.

### 3.2 Data Augmentation
- Random horizontal flip.
- Random rotation.
- Brightness/contrast jitter.

### 3.3 Models
- Custom CNN baseline.
- ResNet18 transfer learning.
- MobileNetV3 Small transfer learning.

### 3.4 Training Setup
- Loss: CrossEntropyLoss.
- Optimizer: Adam.
- Batch size: 32
- Learning rate: 0.0005
- Epochs: 25
- Device (CPU/GPU): GPU (if CUDA available), otherwise CPU

## 4. Experiments
### 4.1 Experiment Matrix
| Experiment ID | Model | Epochs | Batch Size | LR | Notes |
|---|---|---:|---:|---:|---|
| E1 | custom_cnn |  |  |  |  |
| E2 | resnet18 | 25 | 32 | 0.0005 | Best validation accuracy = 1.0000 |
| E3 | mobilenet_v3_small |  |  |  |  |

### 4.2 Training Curves
- Insert curve figures:
  - `outputs/figures/curves_custom_cnn.png`
  - `outputs/figures/curves_resnet18.png`
  - `outputs/figures/curves_mobilenet_v3_small.png`
- Discuss overfitting/underfitting.

## 5. Results and Evaluation
### 5.1 Quantitative Results
| Model | Validation Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| custom_cnn |  |  |  |  |  |
| resnet18 | 1.0000 | 0.9815 | 0.98 | 0.98 | 0.98 |
| mobilenet_v3_small |  |  |  |  |  |

### 5.2 Confusion Matrix Analysis
- Insert confusion matrix per model:
  - `outputs/figures/confusion_custom_cnn.png`
  - `outputs/figures/confusion_resnet18.png`
  - `outputs/figures/confusion_mobilenet_v3_small.png`
- Explain which classes are confused and why.

### 5.3 Class-wise Metrics
- Use `outputs/reports/metrics_<model>.json`.
- ResNet18 results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Crazing | 0.96 | 1.00 | 0.98 | 45 |
| Inclusion | 1.00 | 0.93 | 0.97 | 45 |
| Patches | 1.00 | 0.96 | 0.98 | 45 |
| Pitted_surface | 0.94 | 1.00 | 0.97 | 45 |
| Rolled-in_scale | 1.00 | 1.00 | 1.00 | 45 |
| Scratches | 1.00 | 1.00 | 1.00 | 45 |

- Overall:
  - Accuracy: `0.9815` (270 samples)
  - Macro avg: Precision `0.98`, Recall `0.98`, F1 `0.98`
  - Weighted avg: Precision `0.98`, Recall `0.98`, F1 `0.98`

## 6. Demo System
### 6.1 Inference Pipeline
Input image -> preprocessing -> trained model -> predicted defect + confidence.

### 6.2 Demo Command
```bash
python predict.py \
  --image path/to/image.jpg \
  --checkpoint outputs/checkpoints/best_resnet18.pt \
  --model resnet18
```

### 6.3 Example Output
- Prediction:
- Confidence:

## 7. Discussion
- Why one model performed better.
- Error patterns and failure cases.
- Deployment considerations in a real factory line.

## 8. Conclusion and Future Work
- Key achievements.
- Current limitations.
- Future improvements:
  - More data and domain adaptation.
  - Better augmentations and hyperparameter tuning.
  - Object detection/segmentation for localization.

## References
- NEU Surface Defect Database.
- PyTorch documentation.
- Related papers on surface defect inspection.

## Appendix
### A. Reproducibility Commands
```bash
python eda.py --data-dir data/raw
python prepare_data.py --input-dir data/raw --output-dir data/processed --seed 42
python train.py --data-dir data/processed --model resnet18 --epochs 25 --batch-size 32 --lr 0.0005
python evaluate.py --data-dir data/processed --checkpoint outputs/checkpoints/best_resnet18.pt --model resnet18
```

### B. Hardware and Software
- OS:
- Python version:
- PyTorch version:
- GPU/CPU:
