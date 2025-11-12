# HW3 Dubin's Plane Trajectory Prediction with LSTMs
**Authors:** Pranay Katyal, Anirudh Ramanathan
**Course:** RBE 577 - Machine Learning for Robotics

## Overview
This project utilizes Long Short-Term Memory (LSTM) neural networks to predict Dubin's plane trajectories, relevant for autonomous vehicle path planning. The model is designed to forecast trajectories in a body frame for real-time and simulation-based planning scenarios.

## Installation
``` bash

# Create conda environment
conda create -n MLRenv python=3.10
conda activate MLRenv

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia

```

# Install additional packages
```bash
pip install matplotlib tensorboard numpy

# Directory 

HW3/
├─ trajectory_plots_body_frame/*.png
├── HW3.py
├── visualize_traj.py
```
# Training the model
python3 HW3.py
# Visualization (generates 100 trajectory plots)
python3 visualize_traj.py

# Monitor training progress with TensorBoard
tensorboard --logdir=runs/dubins_body_frame
Open a browser to http://localhost:6006 for TensorBoard dashboard.


# Python Environment

- Python Version: 3.10 or higher

# Required Packages
- torch>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- tensorboard>=2.13.0

# Hardware

- GPU: CUDA-compatible GPU recommended 
- RAM: 16GB+ recommended for dataset generation
- Storage: ~500MB for dataset and model checkpoints