#!/bin/bash

#SBATCH --mail-user=pkatyal@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH -J m4depth_setup
#SBATCH --output=/home/pkatyal/MachineLearning4Robotics/HWFin/logs/m4depth_setup_%j.out
#SBATCH --error=/home/pkatyal/MachineLearning4Robotics/HWFin/logs/m4depth_setup_%j.err

#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100|A30
#SBATCH -p academic
#SBATCH -t 2:00:00

# Load modules
module load cuda/12.4.0/3mdaov5
module load miniconda3

# Create conda environment
conda create -n m4depth_tf python=3.8 -y

# Activate environment
source "$("conda" info --base)/etc/profile.d/conda.sh"
conda activate m4depth_tf

# Install dependencies
pip install tensorflow==2.10.0
pip install numpy pandas opencv-python matplotlib tqdm pillow

# Test installation
cd /home/pkatyal/MachineLearning4Robotics/HWFin/M4Depth
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

echo "Environment setup complete!"