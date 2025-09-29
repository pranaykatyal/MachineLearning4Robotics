# HW1.py Setup & Usage Guide

This guide explains how to set up and run `HW1.py`.

## 1. Prerequisites
- Python 3.8 or newer (recommended: 3.10)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Git (optional, for cloning the repository)

## 2. Clone or Transfer the Project
- **Option 1:** Clone from GitHub (if available)
  ```bash
  git clone https://github.com/pranaykatyal/MachineLearning4Robotics/tree/main/HW%20Folder
  cd MachineLearning4Robotics/HW Folder
  ```

## 3. Create and Activate Conda Environment
```bash
conda create -n MLRenv python=3.10 -y
conda activate MLRenv
```

## 4. Install Required Python Packages
```bash
pip install torch numpy matplotlib scikit-learn tensorboard
```

## 5. Run HWdebugging.py
```bash
python3 HW1.py
```

## 6. (Optional) View TensorBoard Logs
If the script uses TensorBoard:
```bash
tensorboard --logdir runs/control_allocation_experiment
```
Then open the displayed URL in your browser.

## 7. Troubleshooting
- If you see missing package errors, install them using pip (e.g., `pip install <package>`).
- For CUDA/GPU support, install the correct version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).
- If you encounter permission errors, try running with `sudo` or check file permissions.

## 8. Notes
- The script expects all dependencies to be installed in the active environment.
- Output and logs will be generated in the current folder or as specified in the script.

---
For further help, contact pkatyal@wpi.edu.
