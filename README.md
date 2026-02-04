# Cell Well Segmentation and Feature Extraction

Classical computer vision pipeline for **instance segmentation and feature extraction of low-density cell wells** from multi-channel fluorescence microscopy images.

This repository implements a **fully reproducible, non–deep learning workflow** that produces segmentation masks, per-cell quantitative features, QuPath-compatible annotations, and visualization previews, with an emphasis on robustness, clarity, and memory management.

---

## Key Features

- Multi-channel TIFF image handling  
- Nuclei-based seed detection  
- Marker-controlled watershed instance segmentation  
- Morphological and intensity feature extraction  
- CSV export of per-cell features  
- GeoJSON annotations compatible with **QuPath**  
- Visualization previews for quality control  
- Designed for **low-density cell wells**  
- No machine learning or deep learning dependencies  

---

## Repository Structure

```
cell-well-segmentation/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── README.md
│   └── example/
│       ├── README.md
│       └── D1_CropMini.tif
│
├── src/
│   └── cellwell/
│       ├── __init__.py
│       ├── pipeline.py
│       └── cli.py
│
├── tests/
│   └── test_code.py
│
└── docs/
```

---

## Input Data Format

**Input:** multi-channel fluorescence microscopy images stored as TIFF files (`.tif` or `.tiff`).

Each input image is processed independently and generates a corresponding output folder.

### Channel Convention

The pipeline assumes the following channel order:

| Channel index | Description |
|--------------|------------|
| 0 | Nuclear stain (e.g. DAPI, blue) |
| 1 | Marker channel (red) |
| 2 | Marker channel (green) |
| 3 | Cytoplasmic stain (CellCyto) |

⚠️ **Important:**  
If your data uses a different channel order, the channel mapping must be updated in `create_channel_images()` within `pipeline.py`.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/cell-well-segmentation.git
cd cell-well-segmentation
pip install -r requirements.txt
```

Install the project in editable mode (required for the `src/` layout):

```bash
pip install -e .
```

**Requirements**
- Python ≥ 3.9  
- Tested on Windows and Linux  

---

## Running the Pipeline (Recommended)

The pipeline is executed via a **command-line interface (CLI)**.

### Run on the provided example data

```bash
python -m cellwell.cli --input "data/example/D1_CropMini.tif"
```

### Run on your own data

```bash
python -m cellwell.cli --input "path/to/your/data/*.tif"
```

For each input file, an output folder is created automatically.

---

## Output Files

For each processed image, the following files are generated:

- **`RGB.tif`**  
  RGB composite image created from selected channels.

- **`CellCyto.tif`**  
  Composite image highlighting cytoplasmic and nuclear signals, used for segmentation.

- **`instances.tif`**  
  Instance-labeled segmentation mask (`uint16`), where each cell has a unique label.

- **`cell_features.csv`**  
  Per-cell quantitative features, including:
  - area and perimeter  
  - per-channel mean, max, median, standard deviation  
  - coefficient of variation (CV)  
  - integrated intensity per channel  
  - centroid coordinates  

- **`qupath_final.geojson`**  
  Polygon annotations compatible with **QuPath**, enabling visualization and manual inspection.

- **`preview.png`**  
  Downsampled visualization summarizing nuclei detection, segmentation, and final instances.

---

## Testing and Reproducibility

Minimal automated tests are provided using **pytest**.

Run tests from the repository root:

```bash
python -m pytest
```

Tests verify:
- correct package installation
- successful imports
- end-to-end execution on example data

⚠️ Tests must be run via `pytest`.  
Direct execution of test files (e.g. `python test_code.py`) is not supported when using a `src/` layout.

---

## Method Summary

The segmentation pipeline consists of:

1. Gaussian smoothing of the nuclear channel  
2. Local maxima detection for seed generation  
3. Adaptive foreground detection using Otsu thresholding  
4. Marker-controlled watershed segmentation  
5. Post-segmentation filtering based on object area  

Feature extraction is performed on the original-resolution RGB data to preserve intensity fidelity.

---

## Notes and Limitations

- Optimized for **low-density cell wells**
- Parameters may require tuning for different magnifications or staining conditions
- Not designed for highly confluent or overlapping cells
- No machine learning or deep learning models are used

---

## License

This project is released under the **MIT License**.

---

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact:

**Jose J. Rodriguez Rojas**  
📧 juaco2r@gmail.com

