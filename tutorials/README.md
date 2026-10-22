# sam2roman Tutorials

Interactive walkthroughs and reproducible examples for running Meta's Segment Anything Model 2 (SAM 2) on Roman Space Telescope and astronomical FITS images using Hugging Face and PyTorch.

---

## Available Tutorials

### [Tutorial 01: Segment Everything in a FITS Image](01_fits_segment_everything.ipynb)
- **Format**: Jupyter Notebook ([`01_fits_segment_everything.ipynb`](01_fits_segment_everything.ipynb)) and CLI script ([`01_fits_segment_everything.py`](01_fits_segment_everything.py))
- **Overview**:
  1. Reading single-band or multi-extension FITS images using Astropy & `sam2roman.io.FITSImage`.
  2. Preserving celestial WCS coordinates (RA, Dec) and pixel scale (~0.11 arcsec/pixel for Roman WFI).
  3. Preprocessing wide dynamic range flux data using non-linear `asinh` / `zscale` stretching into RGB format.
  4. Initializing SAM 2 models (`facebook/sam2.1-hiera-tiny`, `facebook/sam2.1-hiera-small`, etc.) from Hugging Face Hub.
  5. Performing automatic mask generation across the entire astronomical field.
  6. Extracting source photometry: integrated flux, signal-to-noise ratio (SNR), half-light radius, and centroid positions.
  7. Visualizing multi-panel detection summaries with mask overlays and writing results back to multi-extension FITS files.

---

## Running the Tutorials

### Method 1: Using JupyterLab
Ensure the `sam2roman-dev` conda environment is active:
```bash
conda activate sam2roman-dev
jupyter lab tutorials/
```
Inside Jupyter, select the **Python (sam2roman-dev)** kernel.

### Method 2: Running via Python CLI
You can run the tutorial directly from the terminal:
```bash
conda activate sam2roman-dev
python tutorials/01_fits_segment_everything.py
```

Outputs will be saved to:
- `tutorials/01_segmentation_summary.png`
- `tutorials/sample_data/sample_roman_wfi_segmented.fits`
