# sam2roman Tutorials

Interactive walkthroughs and reproducible examples for running and fine-tuning Meta's Segment Anything Model 2 (SAM 2) on Roman Space Telescope and astronomical FITS images using Hugging Face and PyTorch.

---

## Available Tutorials

### [Tutorial 01: Segment Everything in a FITS Image](01_fits_segment_everything.ipynb)
- **Files**: Notebook ([`01_fits_segment_everything.ipynb`](01_fits_segment_everything.ipynb)) and CLI script ([`01_fits_segment_everything.py`](01_fits_segment_everything.py))
- **Key Concepts**:
  1. Reading FITS images and WCS headers (RA, Dec) with `sam2roman.io.FITSImage`.
  2. Preserving high dynamic range using non-linear `asinh` stretching.
  3. Running high-performance automatic mask generation ("segment everything") with `verbose=True`.
  4. Decoupled astronomical source photometry (integrated flux, SNR, peak, half-light radius).
  5. Generating diagnostic summary plots and exporting labeled masks to multi-extension FITS.

---

### [Tutorial 02: Training & Fine-Tuning SAM 2 on One Dataset](02_training_sam2_fits.ipynb)
- **Files**: Notebook ([`02_training_sam2_fits.ipynb`](02_training_sam2_fits.ipynb)) and CLI script ([`02_training_sam2_fits.py`](02_training_sam2_fits.py))
- **Key Concepts**:
  1. **What things you need**: FITS images, integer labeled ground-truth segmaps, prompt pairs, pretrained weights, accelerator.
  2. **Expected formats**: Tiling a single large master FITS image (e.g. 4088x4088) into overlapping $512\times 512$ cutouts using `AstronomicalDataset.from_single_fits_tiled()`.
  3. **Organization of training**: Freezing the Hiera vision backbone, training the prompt encoder and mask decoder with Combined BCE + Dice loss.
  4. **Parameters of training**: Detailed hyperparameter guide (learning rate $10^{-4}$, weight decay $10^{-4}$, batch size 2–4, prompt types).
  5. **Live validation**: Tracking both validation loss and mean validation IoU per epoch.
  6. **Inference with fine-tuned checkpoint**: Saving `checkpoint_best.pt` and loading fine-tuned weights into `SAM2Segmentor`.

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
```
