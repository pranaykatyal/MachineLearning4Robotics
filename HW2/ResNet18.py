# Author: Pranay Katyal, Anirudh Ramanathan
# Standard libraries
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
import matplotlib.pyplot as plt

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# Torchvision
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Path handling
from pathlib import Path

# Device configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Params:
    def __init__(self):
        self.batch_size = 64
        self.lr = 1e-3
        self.lr_phase2 = 1e-4
        self.epochs_phase1 = 10
        self.epochs_phase2 = 30
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.name = "HW2_ResNet18_Vehicle_Classifier"
        self.workers = 4  # for data loading
        
    def __repr__(self):
        return str(self.__dict__)


# Training transforms (WITH augmentation)
train_transforms = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256 first
    transforms.RandomResizedCrop(224),  # Then random crop to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flip
    # transforms.RandomPerspective(p = 0.5, distortion_scale=0.2), # 50% chance of perspective change
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1), # Randomly change brightness, contrast, saturation and hue
    transforms.RandomRotation(degrees=10), # Randomly rotate the image by 15 degrees
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Apply Gaussian Blur
    # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # Randomly adjust sharpness
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Randomly translate the image by 10% in both directions
    # transforms.RandomGrayscale(p=0.1), # 10% chance of converting to grayscale
    # ADD MORE AUGMENTATIONS HERE - what makes sense for vehicles?
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Validation transforms (NO augmentation)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Center crop (deterministic, not random)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])