#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Setup Script for sam2roman
# Optimized for g6.xlarge (1x NVIDIA L4 24GB GPU) and g5.xlarge (1x NVIDIA A10G 24GB GPU)
# ==============================================================================

set -e

echo "=============================================================================="
echo "          sam2roman: AWS EC2 NVIDIA GPU Environment Setup                     "
echo "=============================================================================="

# 1. Check NVIDIA GPU presence
echo -e "\n[Step 1/5] Checking GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[Warning] nvidia-smi not found. Ensure NVIDIA driver and CUDA are installed on this instance."
fi

# 2. System dependencies (Ubuntu / Debian / Amazon Linux)
echo -e "\n[Step 2/5] Updating system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y git wget curl build-essential libgl1 libglib2.0-0
elif command -v dnf &> /dev/null; then
    sudo dnf install -y git wget curl gcc mesa-libGL glib2
fi

# 3. Setup Python / Conda environment
echo -e "\n[Step 3/5] Setting up Python environment..."
CONDA_ENV_NAME="sam2roman-gpu"

if command -v conda &> /dev/null; then
    echo "Found conda: $(conda --version)"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    
    if conda env list | grep -q "$CONDA_ENV_NAME"; then
        echo "Activating existing conda environment '$CONDA_ENV_NAME'..."
        conda activate "$CONDA_ENV_NAME"
    else
        echo "Creating new conda environment '$CONDA_ENV_NAME' with Python 3.11..."
        conda create -y -n "$CONDA_ENV_NAME" python=3.11 pip
        conda activate "$CONDA_ENV_NAME"
    fi
else
    echo "Conda not found in PATH. Checking virtualenv..."
    if [ ! -d "$HOME/venv_sam2roman" ]; then
        python3 -m venv "$HOME/venv_sam2roman"
    fi
    source "$HOME/venv_sam2roman/bin/activate"
fi

# 4. Install PyTorch with CUDA 12.1
echo -e "\n[Step 4/5] Installing PyTorch with CUDA 12 support..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install sam2roman and dependencies
echo -e "\n[Step 5/5] Installing sam2roman package and cloud requirements..."
# Navigate to repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pip install -e .
pip install boto3 awscli tqdm matplotlib astropy scipy transformers accelerate

echo -e "\n=============================================================================="
echo "          Environment Validation                                              "
echo "=============================================================================="
python3 -c "
import torch
print('PyTorch Version:   ', torch.__version__)
print('CUDA Available:    ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:        ', torch.cuda.get_device_name(0))
    print('VRAM Available:    ', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
    print('bfloat16 Supported:', torch.cuda.is_bf16_supported())
"

echo -e "\n[Setup Complete] You are ready to train on AWS!"
echo "To start training:"
echo "  python cloud/train_aws.py --images_dir /path/to/fits/images --masks_dir /path/to/fits/masks"
