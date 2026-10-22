# sam2roman 🌌

A modular, lightweight Python library for **inference** and **fine-tuning** with Meta's **Segment Anything Model 2 (SAM 2)** on **Roman Space Telescope** and astronomical FITS imagery using **Hugging Face** and **PyTorch**.

---

## Features

- 🔭 **Astronomical FITS Support**: Seamlessly reads single-band or multi-extension FITS files (`SCI`, `PRIMARY`, etc.) preserving celestial WCS headers (RA, Dec) via Astropy.
- 🎨 **Astronomical Intensity Stretching**: Preserves wide dynamic ranges (spanning $10^{-2}$ to $10^5$ counts) using Lupton `asinh`, `zscale`, and percentile stretches into RGB formats expected by SAM 2 vision encoders.
- 🤗 **Hugging Face & PyTorch Native**: Built directly on `transformers>=4.50.0` and PyTorch. Supports all official SAM 2 checkpoints:
  - `facebook/sam2.1-hiera-tiny` (Fastest, default for quick inference)
  - `facebook/sam2.1-hiera-small`
  - `facebook/sam2.1-hiera-base-plus`
  - `facebook/sam2.1-hiera-large`
- ⚡ **Hardware Acceleration**: Automatic accelerator detection: NVIDIA CUDA, Apple Silicon MPS (Metal Performance Shaders), or CPU.
- 🪐 **"Segment Everything" Out-of-the-Box**: Automatically generates segmentation masks for all discrete sources (stars and galaxies) across an astronomical image.
- 🎯 **Targeted Prompting**: Segment specific sources by providing point coordinates (e.g. from an astronomical source catalog like Gaia or SExtractor) or bounding boxes.
- 📊 **Source Photometry & Morphology**: Measures integrated flux, signal-to-noise ratio (SNR), half-light radius, area, and centroid positions on detected masks.
- 🛠️ **Fine-Tuning & Training Engine**: Includes custom `AstronomicalDataset`, combined BCE + Dice segmentation losses, and `SAM2Trainer` with frozen vision backbones for domain adaptation on Roman WFI data.
- 💾 **Astronomical FITS Export**: Saves detected masks directly into multi-extension FITS format (`SEGMAP` HDU).

---

## Repository Structure

```text
sam2roman/
├── environment.yml                # Conda environment definition (sam2roman-dev)
├── pyproject.toml                 # PEP 517/621 package metadata & configuration
├── README.md                      # Library documentation
├── src/
│   └── sam2roman/
│       ├── __init__.py            # Top-level API exports
│       ├── io/
│       │   └── fits.py            # FITSImage, WCS handling, asinh/zscale stretch
│       ├── models/
│       │   └── sam2.py            # SAM2ModelWrapper (HF Transformers + PyTorch)
│       ├── inference/
│       │   └── segmentor.py       # SAM2Segmentor & SegmentationResult
│       ├── training/
│       │   ├── dataset.py         # AstronomicalDataset with prompt sampling
│       │   ├── losses.py          # DiceLoss, FocalLoss, CombinedSegmentationLoss
│       │   └── trainer.py         # SAM2Trainer fine-tuning loop
│       ├── visualization/
│       │   └── display.py         # Plotting overlays, diagnostic summaries
│       └── utils/
│           ├── synthetic.py       # Realistic Roman WFI synthetic FITS generator
│           └── metrics.py         # Source photometry, aperture flux, SNR
├── tutorials/
│   ├── README.md                  # Tutorial guide
│   ├── 01_fits_segment_everything.ipynb  # Interactive Jupyter notebook
│   ├── 01_fits_segment_everything.py     # Executable CLI script
│   └── sample_data/
│       ├── generate_sample_fits.py       # Script to generate sample data
│       └── sample_roman_wfi.fits         # Ready-to-use Roman WFI sample
└── tests/                         # Pytest test suite
    ├── conftest.py
    ├── test_fits_io.py
    ├── test_synthetic.py
    ├── test_losses.py
    └── test_inference.py
```

---

## Installation

### 1. Create and Activate Conda Environment

The repository provides an `environment.yml` configuring Python 3.12, PyTorch, Hugging Face Transformers, Astropy, and development dependencies:

```bash
# Clone repository
git clone https://github.com/your-org/sam2roman.git
cd sam2roman

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate sam2roman-dev
```

### 2. Install Package in Editable Mode

```bash
pip install -e .
```

---

## Quickstart

### 1. Automatic Mask Generation ("Segment Everything")

```python
from sam2roman.io.fits import FITSImage
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.visualization.display import plot_detection_summary

# 1. Load astronomical FITS image
fits_img = FITSImage.from_file("tutorials/sample_data/sample_roman_wfi.fits")

# 2. Initialize SAM 2 from Hugging Face
segmentor = SAM2Segmentor(model_id="facebook/sam2.1-hiera-tiny")

# 3. Segment everything across the field
results = segmentor.segment_everything(
    image=fits_img,
    points_per_side=32,
    pred_iou_thresh=0.65,
    min_mask_region_area=15,  # Filter out hot pixels / cosmic rays
    stretch_type="asinh",
)
print(f"Detected {len(results)} sources!")

# 4. Generate diagnostic plot
fig = plot_detection_summary(fits_img, results, title_prefix="Roman WFI")
fig.savefig("segmentation_summary.png", dpi=200)

# 5. Save masks back into FITS format
fits_img.save_masks_to_fits(results.masks, output_path="segmented_output.fits")
```

### 2. Prompt-Based Segmentation (Source Catalog)

```python
# Provide pixel coordinates of target stars/galaxies from an astronomical catalog
catalog_points = [(152.4, 210.1), (305.8, 88.5)]

results = segmentor.segment_points(
    image=fits_img,
    points=catalog_points,
    stretch_type="asinh",
)
```

### 3. Fine-Tuning SAM 2 on Custom Astronomical Datasets

```python
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.trainer import SAM2Trainer

# Prepare dataset with FITS files and ground truth masks
dataset = AstronomicalDataset(
    images=["sample_1.fits", "sample_2.fits"],
    masks=[mask_1, mask_2],
    stretch_type="asinh",
)

# Initialize trainer with frozen vision encoder
trainer = SAM2Trainer(
    model_id="facebook/sam2.1-hiera-tiny",
    learning_rate=1e-4,
    freeze_vision_encoder=True,
)

# Run fine-tuning
history = trainer.train(dataset, epochs=5, batch_size=2, output_dir="checkpoints/")
```

---

## Tutorials

Head over to the [`tutorials/`](tutorials/) folder to get started:
- **[Tutorial 01: Segment Everything in a FITS Image](tutorials/01_fits_segment_everything.ipynb)**
  - Available as interactive Jupyter notebook (`01_fits_segment_everything.ipynb`) or CLI script (`01_fits_segment_everything.py`).

To launch in Jupyter:
```bash
conda activate sam2roman-dev
jupyter lab tutorials/
```

---

## Running Tests

Execute the unit test suite:
```bash
conda activate sam2roman-dev
pytest tests/ -vv
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
