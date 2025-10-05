# Vehicle Classification with ResNet18
**Authors:** Pranay Katyal, Anirudh Ramanathan  
**Course:** RBE 577 - Machine Learning for Robotics

## Overview
This project implements transfer learning with ResNet18 to classify road vehicles for autonomous driving applications. The model achieves 96% validation accuracy using a two-phase training strategy.

## Requirements
- Python 3.10.18
- PyTorch 2.7.1+cu128
- torchvision 0.22.1+cu128
- numpy 2.0.1
- matplotlib 3.10.5
- tensorboard 2.20.0

## Installation
```bash
# Create conda environment

conda create -n MLRenv python=3.10
conda activate MLRenv

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia

# Install additional packages
pip install matplotlib tensorboard numpy

# Verify Structure 

Download the dataset from Kaggle Vehicle Classification

Extract to archive/ folder inside the HW2/ directory

HW2/
├── archive/
│   ├── train/
│   ├── val/
│   └── test/
├── ResNet18.py
└── EvaluateResNet18.py
```

# Running - Training the model
```bash
cd HW2
python ResNet18.py
```
- Training takes approximately 22 minutes on RTX 4080 12GB
- Checkpoints saved to checkpoints/HW2_ResNet18_Vehicle_Classifier/
- TensorBoard logs saved to runs/HW2_ResNet18_Vehicle_Classifier/

# Monitoring the progress 
```bash
cd HW2
tensorboard --logdir=runs
```
- Open browser to http://localhost:6006 (might be different for you)

# Evaluating the model
```bash
cd HW2
python EvaluateResNet18.py
```
- Generates predictions.png with 10 randomly selected validation images showing predicted vs true labels.

# Test the model
```bash
cd HW2
python TestResNet18.py
```
- Generates test_predictions.png with 10 randomly selected test images showing predicted labels.

 ## Model Architecture

- Base Model: ResNet18 pretrained on ImageNet
- Modification: Final FC layer changed from 1000 to 10 classes
- Parameters: ~11M total

## Training Strategy - Adaptive freezing

- Phase 1 (20 epochs): Frozen backbone, train classification head only
- Phase 2 (30 epochs): Fine-tune all layers with lower learning rate

# Results

- Final Validation Accuracy: 96.0%
- Final Training Accuracy: 95.4%
- No overfitting observed ( thus we did not need Regularization - probably because of our smaller ResNet18 choice. )
