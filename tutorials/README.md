# sam2roman Tutorials

Interactive walkthroughs and reproducible examples for running and fine-tuning Meta's Segment Anything Model 2 (SAM 2) on Roman Space Telescope and astronomical FITS images using Hugging Face and PyTorch.

---

## Available Tutorials

### [Tutorial 01: Segment Everything in a FITS Image](01_fits_segment_everything.ipynb)
- **Files**: Notebook ([`01_fits_segment_everything.ipynb`](01_fits_segment_everything.ipynb)) and CLI script ([`01_fits_segment_everything.py`](01_fits_segment_everything.py))
- **Key Concepts**:
  1. Reading FITS images and WCS headers (RA, Dec) with `sam2roman.io.FITSImage`.
  2. Preserving high dynamic range using non-linear `asinh` stretching.
  3. High-performance single-pass automatic mask generation ("segment everything") with `verbose=True`.
  4. Decoupled astronomical source photometry (integrated flux, SNR, peak, half-light radius).
  5. Generating diagnostic summary plots and exporting labeled masks to multi-extension FITS.

---

### [Tutorial 02: Training & Fine-Tuning SAM 2 on One Dataset](02_training_sam2_fits.ipynb)
- **Files**: Notebook ([`02_training_sam2_fits.ipynb`](02_training_sam2_fits.ipynb)) and CLI script ([`02_training_sam2_fits.py`](02_training_sam2_fits.py))
- **Key Concepts**:
  1. **What things you need**: FITS images, integer labeled ground-truth segmaps, prompt pairs, pretrained weights, accelerator.
  2. **Single Master Tiling**: Slicing 1 large master FITS image (e.g. 4088x4088) into overlapping $512\times 512$ cutouts with `AstronomicalDataset.from_single_fits_tiled()`.
  3. **Organization of training**: Freezing the Hiera vision backbone, training the prompt encoder and mask decoder with Combined BCE + Dice loss.
  4. **Parameters of training**: Detailed hyperparameter guide (learning rate $10^{-4}$, weight decay $10^{-4}$, batch size 2–4).
  5. **Live validation**: Tracking both validation loss and mean validation IoU per epoch.
  6. **Inference with fine-tuned checkpoint**: Saving `checkpoint_best.pt` and loading fine-tuned weights into `SAM2Segmentor`.

---

### [Tutorial 03: Training from Paired Directories with Point Prompts (A100 40GB Ready)](03_train_from_directories.ipynb)
- **Files**: Notebook ([`03_train_from_directories.ipynb`](03_train_from_directories.ipynb)) and CLI script ([`03_train_from_directories.py`](03_train_from_directories.py))
- **Key Concepts**:
  1. **Two Directories Setup**: Passing separate `images_dir` (`.../original/*.fits`) and `masks_dir` (`.../masks/*.fits`).
  2. **Point Prompts (`prompt_type='point'`)**: Training SAM 2 using foreground clicks instead of bounding boxes.
  3. **Multi-Mask Crops (`all_masks_in_crop=True`)**: Concurrently training on all source masks present inside each $512 \times 512$ cutout.
  4. **NVIDIA A100 GPU (40GB) Resource Saturation**: Auto-detected high-throughput profile with batch size 16–32, TensorFloat-32 (TF32), native bfloat16 AMP, and multi-process DataLoader (`num_workers=4`, `pin_memory=True`).
  5. **Default `leakySB` Stretch**: Surface brightness transformation preserving faint galaxy wings with identical RGB channels.
  6. **Low-Epoch Verification & Inference**: Fast fine-tuning with live validation IoU and test patch segmentation.

---

### [Tutorial 04: Segment Everything with Fine-Tuned Checkpoints](04_segment_everything_finetuned.ipynb)
- **Files**: Notebook ([`04_segment_everything_finetuned.ipynb`](04_segment_everything_finetuned.ipynb)) and CLI script ([`04_segment_everything_finetuned.py`](04_segment_everything_finetuned.py))
- **Key Concepts**:
  1. **Checkpoint Ingestion**: Loading fine-tuned prompt encoder and mask decoder weights from `checkpoint_best.pt` via `SAM2Segmentor.from_checkpoint()`.
  2. **Promptless Automatic Mask Generation**: Running `segment_everything()` across an entire FITS exposure using fine-tuned weights.
  3. **Prompt Modes**: Comparing `astronomical` (fast candidate peak detection) and `grid` (dense query lattice).
  4. **Decoupled Source Photometry**: Measuring net flux, SNR, centroid, and half-light radius on all segmented sources.
  5. **Visualizations**: Generating 3-panel detection summaries and colorful mask overlays.
  6. **Export**: Writing labeled masks to multi-extension FITS with a standard `SEGMAP` HDU.

---

### [Tutorial 05: Boundary-Aware UNet++ Training & Deblending](05_train_unetplusplus.ipynb)
- **Files**: Notebook ([`05_train_unetplusplus.ipynb`](05_train_unetplusplus.ipynb)) and CLI script ([`05_train_unetplusplus.py`](05_train_unetplusplus.py))
- **Key Concepts**:
  1. **3-Class Boundary Targets**: Training on Background (0), Source Interior (1), and Source Boundary/Edge (2).
  2. **Ultra-Fast Prompt-Free Inference**: Single forward pass in $<15$ ms with zero prompt clicks.
  3. **Touching Instance Deblending**: Using predicted interior cores and boundary channels with marker-controlled watershed to cleanly split overlapping sources.
  4. **Automatic Hardware Optimization**: Auto-detects Apple Silicon MPS (FP32) or NVIDIA CUDA (TF32, bfloat16 AMP, pinned memory).

---

### [Tutorial 06: Full-Image Astronomical Inference with UNet++ & LeakySB](06_unetpp_fits_inference.ipynb)
- **Files**: Notebook ([`06_unetpp_fits_inference.ipynb`](06_unetpp_fits_inference.ipynb)) and CLI script ([`06_unetpp_fits_inference.py`](06_unetpp_fits_inference.py))
- **Key Concepts**:
  1. **Unified FITS & ASDF I/O**: Reading science pixel data, physical moments, and celestial WCS headers.
  2. **Leaky Surface Brightness (`leakySB`) Stretch**: Normalizing astronomical images across $>10^5$ dynamic range.
  3. **Tiled Dense Inference (`predict_dense`)**: Sliding-window execution with 2D cosine window blending for seamless full-frame inference on arbitrary image sizes (including 4088x4088 Roman WFI exposures).
  4. **Publication 2-Panel Diagnostic Figures**: Side-by-side display of the LeakySB image and the discrete 3-class segmentation map (Sky, Core, Deblending Edge).
  5. **Crowded Field Cutout Inspection**: Zooming into high-density regions to verify 1-2 pixel contact boundary resolution.
  6. **Watershed Deblending & Export**: Converting probability maps into discrete source catalogs (`segment_image`) and exporting multi-extension FITS with a `PRED_MASK` HDU.

---

## Running the Tutorials

### Method 1: Interactive JupyterLab
Ensure the `sam2roman-dev` conda environment is active:
```bash
conda activate sam2roman-dev
jupyter lab tutorials/
```
Inside Jupyter, select the **Python (sam2roman-dev)** kernel.

### Method 2: Running via Python CLI
```bash
conda activate sam2roman-dev

# Run Tutorial 01 (Segmentation & Photometry)
python tutorials/01_fits_segment_everything.py

# Run Tutorial 02 (Fine-Tuning on 1 Dataset)
python tutorials/02_training_sam2_fits.py

# Run Tutorial 03 (Fine-Tuning from Paired Directories with Point Prompts)
python tutorials/03_train_from_directories.py

# Run Tutorial 04 (Segment Everything with Fine-Tuned Checkpoints)
python tutorials/04_segment_everything_finetuned.py

# Run Tutorial 05 (Boundary-Aware UNet++ Training & Deblending)
python tutorials/05_train_unetplusplus.py

# Run Tutorial 06 (Full-Image Inference & LeakySB Visualization)
python tutorials/06_unetpp_fits_inference.py
```
