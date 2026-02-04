## Data Directory

This directory does **not** contain real experimental or patient data.

It is intended to document the expected input format for the pipeline and to
store small example files used for testing and demonstration purposes only.

---

## Expected Input Files

The pipeline expects **multi-channel fluorescence microscopy images** stored as
TIFF files (`.tif` or `.tiff`).

Each input file is processed independently and generates a corresponding output
folder containing segmentation results and extracted features.

---

## Channel Conventions and Assumptions

The pipeline assumes a fixed channel order in the input TIFF files:

| Channel index | Description |
|--------------|------------|
| 0 | Nuclear stain (e.g. DAPI) |
| 1 | Marker channel (red) |
| 2 | Marker channel (green) |
| 3 | Cytoplasmic stain (CellCyto) |

If your data uses a different channel order, the channel mapping must be updated
in the source code (`create_channel_images()` in `pipeline.py`).

Spatial resolution metadata (X/Y resolution) is read directly from the TIFF
header when available.

---

## Example Data

The `example/` subfolder contains a **small, downsampled TIFF file** that is used
exclusively for:

- automated testing
- pipeline validation
- demonstration of expected inputs

These files are **not** representative of full experimental datasets and are
safe to include in the repository.

---

## Using Real Data Locally

To run the pipeline on real data:

1. Store your TIFF files **outside** this repository (recommended), or in a
   local, non-versioned folder.
2. Provide file paths to the pipeline using the command-line interface:
   ```bash
   python -m package_name.cli --input "path/to/your/data/*.tif"
