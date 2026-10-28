# ==============================================================================
# Dockerfile for sam2roman AWS Training
# Compatible with AWS EC2 (NVIDIA L4 / A10G), AWS Batch, ECS, and SageMaker
# ==============================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="sam2roman team"
LABEL description="SAM 2 Fine-Tuning Container for Roman Space Telescope Imagery on AWS"

# Set non-interactive timezone/frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system libraries needed by OpenCV, Astropy, and AWS CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install AWS CLI and cloud dependencies
RUN pip install --no-cache-dir \
    boto3 \
    awscli \
    transformers>=4.50.0 \
    accelerate>=0.30.0 \
    astropy>=6.0.0 \
    matplotlib>=3.8.0 \
    scipy>=1.11.0 \
    tqdm

# Create app directory
WORKDIR /workspace/sam2roman

# Copy codebase
COPY . /workspace/sam2roman

# Install sam2roman in editable mode
RUN pip install --no-cache-dir -e .

# Set default entrypoint to cloud training script
ENTRYPOINT ["python", "cloud/train_aws.py"]
CMD ["--help"]
