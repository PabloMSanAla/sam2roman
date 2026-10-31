#!/usr/bin/env bash
# ==============================================================================
# NASA HECC (NAS) Grace Hopper (GH200) Setup Script for sam2roman
# ==============================================================================
# Automates environment configuration on NASA Advanced Supercomputing (NAS)
# Grace Hopper nodes (e.g. Cabeus cluster).
#
# Reference: https://www.nas.nasa.gov/hecc/support/kb/machine-learning-overview_572.html
# ==============================================================================

set -e

echo "=============================================================================="
echo "      sam2roman: NASA HECC Grace Hopper (GH200) Environment Setup             "
echo "=============================================================================="

# 1. Architecture Check
ARCH=$(uname -m)
echo "[Step 1/5] Checking CPU Architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "[Warning] Current architecture is '$ARCH', but Grace Hopper nodes use 'aarch64' (ARM64)."
    echo "          Ensure you are running on a GH200 compute node (model=gh200)."
fi

# 2. Check GPU Hardware
echo -e "\n[Step 2/5] Checking GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[Notice] nvidia-smi not yet in PATH or running on head node. GPU will be checked after module load."
fi

# 3. Load NAS Modules
echo -e "\n[Step 3/5] Loading NAS Grace Hopper miniconda module..."
if command -v module &> /dev/null || [ -f /usr/share/modules/init/bash ]; then
    [ -f /usr/share/modules/init/bash ] && source /usr/share/modules/init/bash
    module use -a /swbuild/analytix/tools/modulefiles
    module load miniconda3/gh2
    echo "Loaded module: miniconda3/gh2"
else
    echo "[Notice] 'module' command not available in subshell; assuming environment is already loaded."
fi

# 4. Install Supplemental Dependencies via pip (--user and --no-cache-dir flags)
echo -e "\n[Step 4/5] Installing supplemental dependencies from cloud/requirements_nas.txt..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# If numpy or astropy were previously installed in ~/.local, remove them to fall back to the system NAS build
pip uninstall -y numpy astropy 2>/dev/null || true

# Install supplemental dependencies from requirements_nas.txt
pip install --user --no-cache-dir -r cloud/requirements_nas.txt

# Install sam2roman package in editable mode without overriding PyTorch or system packages
pip install --user --no-cache-dir --no-deps -e .

# 5. Verification
echo -e "\n[Step 5/5] Validating Python environment and hardware acceleration..."
python3 -c "
import sys
import numpy as np
import torch
import webdataset
import astropy
import asdf

print('=' * 60)
print('Python:           ', sys.version.split()[0], f'({sys.platform})')
print('NumPy:            ', np.__version__)
print('PyTorch:          ', torch.__version__)
print('CUDA Available:   ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:       ', torch.cuda.get_device_name(0))
    print('VRAM Available:   ', f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
    print('bfloat16 Support: ', torch.cuda.is_bf16_supported())
print('WebDataset:       ', webdataset.__version__)
print('Astropy:          ', astropy.__version__)
print('ASDF:             ', asdf.__version__)
print('=' * 60)
"

echo -e "\n[Setup Complete] NASA HECC GH200 environment successfully configured!"
echo "To launch high-throughput training:"
echo "  python cloud/train_gh200.py --train_shards '/dev/shm/shards/train/*.tar' --val_shards '/dev/shm/shards/val/*.tar' --batch_size 32"
