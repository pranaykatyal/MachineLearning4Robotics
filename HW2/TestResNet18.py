import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class Params:
    def __init__(self):
        self.name = "HW2_ResNet18_Vehicle_Classifier"

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
# model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 10))

checkpoint = torch.load(f"checkpoints/{Params().name}/checkpoint.pth", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()
model = model.to(device)

# Load test images
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get all test images
test_dir = Path("archive/test")
all_images = list(test_dir.glob("*.jpg"))
selected = random.sample(all_images, 10)

# Class names from validation set
val_dataset = datasets.ImageFolder("archive/val", transform=test_transforms)
class_names = val_dataset.classes

def denormalize(tensor, mean, std):
    tensor = tensor.clone().cpu()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.permute(1, 2, 0)
    return np.clip(tensor.numpy(), 0, 1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i, img_path in enumerate(selected):
    from PIL import Image
    image = Image.open(img_path).convert('RGB')
    image_t = test_transforms(image)
    
    # Predict
    with torch.no_grad():
        output = model(image_t.unsqueeze(0).to(device))
        pred = output.argmax(1).item()
    
    # Display
    denorm = denormalize(image_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    axes[i].imshow(denorm)
    axes[i].axis('off')
    axes[i].set_title(f"Prediction: {class_names[pred]}", fontsize=10)

plt.tight_layout()
plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved test_predictions.png")