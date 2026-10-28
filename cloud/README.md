# Running `sam2roman` on AWS EC2 (24GB NVIDIA GPU)

This directory contains scripts and configurations for training and fine-tuning **SAM 2** on Roman Space Telescope FITS imagery using **Amazon Web Services (AWS)** GPU instances.

---

## 1. Recommended AWS Hardware: `g6.xlarge`

The **EC2 `g6.xlarge`** (and `g5.xlarge`) instance family provides **24 GB of GPU VRAM**, which is the sweet spot for fine-tuning SAM 2 on high-resolution ($4088 \times 4088$) Roman WFI cutouts:

| Instance Type | GPU | VRAM | vCPUs | System RAM | Tensor Cores | Recommended Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`g6.xlarge`** | **1x NVIDIA L4** | **24 GB GDDR6** | **4** | **16 GiB** | **4th Gen (Ada Lovelace, native bfloat16)** | **8 – 16** |
| `g6.2xlarge` | 1x NVIDIA L4 | 24 GB GDDR6 | 8 | 32 GiB | 4th Gen (Ada Lovelace) | 8 – 16 |
| `g5.xlarge` | 1x NVIDIA A10G | 24 GB GDDR6 | 4 | 16 GiB | 3rd Gen (Ampere) | 8 – 16 |

---

## 2. Quickstart: Launching on AWS

### Step 1: Launch EC2 Instance
- **AMI**: *Deep Learning AMI GPU PyTorch 2.X (Ubuntu 22.04)*
- **Instance Type**: `g6.xlarge`
- **Storage**: Minimum **100 GB** EBS (GP3, 3000 IOPS, 125 MB/s) to store FITS images and model checkpoints.
- **Security Group**: Open SSH port 22.
- **IAM Role**: Attach an IAM role with S3 Read/Write permissions if storing FITS data on Amazon S3.

### Step 2: Connect to Instance
```bash
ssh -i /path/to/key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3: Clone Repo & Run Automated Setup
```bash
git clone https://github.com/PabloMSanAla/sam2roman.git
cd sam2roman

# Runs one-command environment installation (Conda + PyTorch CUDA 12 + sam2roman)
bash cloud/setup_aws.sh
```

---

## 3. Staging Your Astronomical Data

### Option A: Direct Sync from Amazon S3
If your Roman FITS files are stored in an S3 bucket:
```bash
# Sync images and masks to local instance storage
bash cloud/sync_s3.sh download s3://my-roman-bucket/synthetic/original/ /data/roman/original/
bash cloud/sync_s3.sh download s3://my-roman-bucket/synthetic/masks/ /data/roman/masks/
```

### Option B: Direct S3 Streaming in Training Script
`train_aws.py` natively supports `s3://` URIs:
```bash
python cloud/train_aws.py \
    --images_dir s3://my-roman-bucket/synthetic/original/ \
    --masks_dir s3://my-roman-bucket/synthetic/masks/ \
    --output_dir s3://my-roman-bucket/checkpoints/
```

---

## 4. Running Training on AWS GPU

Execute `cloud/train_aws.py`:

```bash
python cloud/train_aws.py \
    --images_dir /data/roman/original \
    --masks_dir /data/roman/masks \
    --batch_size 8 \
    --epochs 10 \
    --crop_size 512 \
    --prompt_type point \
    --stretch_type leakysb \
    --all_masks_in_crop \
    --max_objects_per_crop 16 \
    --output_dir ./checkpoints_aws
```

### Key Parameters Explained:
- `--batch_size 8`: Highly optimal for 24GB VRAM. (Can increase to `16` for maximum throughput).
- `--all_masks_in_crop`: Segments **all** objects inside each $512 \times 512$ cutout simultaneously in a single forward pass.
- `--stretch_type leakysb`: Preserves full physical dynamic range (magnitude surface brightness) across identical RGB channels.
- `--prompt_type point`: Trains on foreground source coordinates.
- `--max_objects_per_crop 16`: Bounds memory in crowded stellar fields.

---

## 5. Monitoring GPU Utilization

Open a secondary terminal on the EC2 instance:
```bash
watch -n 1 nvidia-smi
```

On a `g6.xlarge` (NVIDIA L4 24GB):
- Expected VRAM utilization with `batch_size=8`: **~6.5 GB to 8.5 GB**
- Expected VRAM utilization with `batch_size=16`: **~11.5 GB to 14.0 GB**
- GPU Compute Utilization: **90% – 100%** during backward pass

---

## 6. Output Artifacts

At the conclusion of training, the script generates:
- `checkpoint_best.pt`: Model weights with highest validation IoU / lowest loss.
- `checkpoint_epoch_X.pt`: Checkpoints saved at each epoch.
- `training_metrics.json`: Full epoch-by-epoch loss and IoU records.
- `training_curves.png`: High-resolution plot of training loss and validation IoU.

---

## 7. Containerized Deployment (Docker)

To run in AWS Batch, ECS, or SageMaker:

```bash
# Build Docker image
docker build -t sam2roman:latest -f cloud/Dockerfile .

# Run with GPU support
docker run --gpus all --rm \
    -v /data/roman:/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    sam2roman:latest \
    --images_dir /data/original \
    --masks_dir /data/masks \
    --batch_size 8 \
    --epochs 10 \
    --output_dir /workspace/checkpoints
```
