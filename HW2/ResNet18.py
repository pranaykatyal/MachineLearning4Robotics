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
        self.epochs_phase1 = 20
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
    transforms.ColorJitter(brightness=0.25,contrast=0.2,saturation=0.2,hue=0.25), # Randomly change brightness, contrast, saturation and hue
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


# Params
params = Params()
# Defining directory
train_dir = "archive/train"
val_dir = "archive/val"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.workers,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.workers,pin_memory=True)

model = models.resnet18(pretrained=True)
print(model.fc)
model.fc = nn.Linear(512, 10)
# model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 10))
print(model.fc)
model = model.to(device)

def freeze_backbone(model):
    """
    Freeze all layers EXCEPT the final fc layer (head)
    """
    # TODO: Loop through model parameters
    # For all layers except fc, set requires_grad = False
    for params in model.parameters():
        params.requires_grad = False
    for params in model.fc.parameters():
        params.requires_grad = True
    return model

def unfreeze_all(model):
    """
    Unfreeze all layers
    """
    for params in model.parameters():
        params.requires_grad = True
    return model


def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    start0 = time.time()
    
    running_loss = 0.0
    correct = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Print progress every 5 batches
        if batch % 5 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    
    # After loop - calculate epoch metrics
    avg_train_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / size
    
    # Log to TensorBoard
    writer.add_scalar('train_loss_per_epoch', avg_train_loss, epoch)
    writer.add_scalar('train_accuracy_per_epoch', accuracy, epoch)
    
    print(f"Epoch {epoch} done in {time.time() - start0:.1f}s - Train Accuracy: {accuracy:.1f}%, Avg loss: {avg_train_loss:.4f}")
    
    
def validate(dataloader, model, loss_fn, epoch, writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    val_loss, correct = 0, 0  
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    val_loss /= num_batches
    accuracy = 100 * correct / size
    
    # Log to TensorBoard
    
    if writer is not None:
        writer.add_scalar('validation_loss_epoch', val_loss, epoch) 
        writer.add_scalar('validation_accuracy_epoch', accuracy, epoch)
    
    print(f"Validation - Accuracy: {accuracy:>0.1f}%, Avg loss: {val_loss:>8f}\n")
    
    return accuracy


if __name__ == '__main__':
    # 1. Setup
    loss_fn = nn.CrossEntropyLoss()  # What loss function for classification?
    
    # 2. Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / params.name  # Use params.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True) 
    
    # 3. TensorBoard writer
    writer = SummaryWriter('runs/' + params.name)
    
    # ===== PHASE 1: Train head only =====
    print("="*50)
    print("PHASE 1: Training head only (frozen backbone)")
    print("="*50)
    
    model = freeze_backbone(model)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)  # Use params.lr
    
    for epoch in range(params.epochs_phase1):
        # Train
        train(train_loader,model,loss_fn,optimizer,epoch,writer)
        # print("Finished epoch : ", epoch)
        # Validate
        acc = validate(val_loader, model, loss_fn, epoch, writer)
        
        # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),        # model's state_dict
            "optimizer":   optimizer.state_dict() , # optimizer's state_dict
            "epoch": epoch,        # current epoch number
            "accuracy": acc,     # what validate() returned
            "params": params,        # params object
        }

        torch.save(checkpoint, checkpoint_dir / f"model_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_dir / "checkpoint.pth")  # Also save as latest
    
    # ===== PHASE 2: Fine-tune everything =====
    print("="*50)
    print("PHASE 2: Fine-tuning all layers")
    print("="*50)
    
    model = unfreeze_all(model)
    optimizer = optim.Adam(model.parameters(), lr=params.lr_phase2, weight_decay=params.weight_decay)   # Use params.lr_phase2 (lower!)
    
    for epoch in range(params.epochs_phase1, params.epochs_phase1 + params.epochs_phase2):
        # Train
        train(train_loader,model,loss_fn,optimizer,epoch,writer)
        # Validate
        acc = validate(val_loader, model, loss_fn, epoch, writer)
        # Save checkpoint
                # Save checkpoint
        checkpoint = {
            "model": model.state_dict(),        # model's state_dict
            "optimizer":   optimizer.state_dict() , # optimizer's state_dict
            "epoch": epoch,        # current epoch number
            "accuracy": acc,     # what validate() returned
            "params": params,        # params object
        }

        torch.save(checkpoint, checkpoint_dir / f"model_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_dir / "checkpoint.pth")  # Also save as latest