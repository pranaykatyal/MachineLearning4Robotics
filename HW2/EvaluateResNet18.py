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
        self.workers = 4
        
    def __repr__(self):
        return str(self.__dict__)



# Device configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 1. Create the same model architecture
model = models.resnet18(pretrained=False)  # Don't need pretrained weights
model.fc = nn.Linear(512, 10)  # Same modification you did before

# 2. Load the saved weights
checkpoint = torch.load("HW2/checkpoints/HW2_ResNet18_Vehicle_Classifier/checkpoint.pth", weights_only=False)
model.load_state_dict(checkpoint["model"])

# 3. Set to evaluation mode
model.eval()
model = model.to(device)


val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("HW2/archive/val", transform=val_transforms)

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image for display
    tensor: (C, H, W) tensor
    Returns: (H, W, C) numpy array in [0, 1] range
    """
    # Clone and move to CPU
    tensor = tensor.clone().cpu()
    
    # Convert mean/std to tensors with shape (3, 1, 1)
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Permute to (H, W, C)
    tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and clip to [0, 1]
    img = tensor.numpy()
    img = np.clip(img, 0, 1)
    
    return img

# Get class names
class_names = val_dataset.classes

# Select 10 random images
num_images = 10
indices = np.random.choice(len(val_dataset), num_images, replace=False)

# Create figure
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Mean and std for denormalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for i, idx in enumerate(indices):
    # Get image and label
    image, true_label = val_dataset[idx]
    
    # Predict
    image_batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_batch)
        pred_label = output.argmax(1).item()
    
    # Denormalize for display
    denorm = denormalize(image, mean, std)
    
    # Plot
    axes[i].imshow(denorm)
    axes[i].axis('off')
    
    # Color: green if correct, red if wrong
    color = "green" if pred_label == true_label else "red"
    axes[i].set_title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}", 
                      color=color, fontsize=10)

# After loop - save once
plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved predictions to predictions.png")