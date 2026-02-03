# cell-well-segmentation
Classical computer vision pipeline for instance segmentation and feature extraction of low-density cell wells from multi-channel fluorescence microscopy images, with CSV outputs, GeoJSON annotations, and visual previews.

# Cell Well Segmentation and Feature Extraction

Python pipeline for **instance segmentation and feature extraction of low-density cell wells** from multi-channel fluorescence microscopy images.

This repository implements a classical computer vision workflow (no deep learning) that produces segmentation masks, per-cell quantitative features, QuPath-compatible annotations, and visualization previews, with an emphasis on robustness and memory management.

---

## Key Features

- Multi-channel TIFF image handling
- Nuclei-based seed detection
- Watershed instance segmentation
- Morphological and intensity feature extraction
- CSV export of per-cell features
- GeoJSON annotations compatible with **QuPath**
- Visualization previews for quality control
- Designed for **non-dense cell wells**

---

## Input Data Format

**Input:** multi-channel fluorescence TIFF image

### Channel Convention

The pipeline assumes the following channel order:

| Channel index | Description |
|--------------|------------|
| 0 | Nuclear stain (e.g. DAPI, blue) |
| 1 | Marker channel (red) |
| 2 | Marker channel (green) |
| 3 | Cytoplasmic stain (CellCyto) |

⚠️ **Important:** If your data uses a different channel order, you must update the channel mapping in `create_channel_images()`.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/cell-well-segmentation.git
cd cell-well-segmentation
pip install -r requirements.txt
```

- Python ≥ 3.9 is recommended
- Tested on Windows and Linux

---

## Running the Pipeline

At the current stage, input files are defined directly inside the script.

```bash
python src/cellwell/pipeline.py
```

Each input image is processed independently and generates a corresponding output folder.

---

## Output Files

For each processed image, the following files are generated:

### `RGB.tif`
RGB composite image created from selected channels.

### `CellCyto.tif`
Composite image highlighting cytoplasmic and nuclear signals, used for segmentation.

### `instances.tif`
Instance-labeled segmentation mask (uint16), where each cell has a unique label.

### `cell_features.csv`
Tabular file containing per-cell quantitative features, including:

- area and perimeter
- per-channel mean, max, median, standard deviation
- coefficient of variation (CV)
- integrated intensity per channel
- centroid coordinates

### `qupath_final.geojson`
Polygon annotations compatible with **QuPath**, enabling visualization and manual inspection of segmented cells.

### `preview.png`
Downsampled visualization summarizing nuclei detection, segmentation, and final instances.

---

## Method Summary

The segmentation pipeline consists of:

1. Gaussian smoothing of the nuclear channel
2. Local maxima detection for seed generation
3. Adaptive foreground detection using Otsu thresholding
4. Marker-controlled watershed segmentation
5. Post-segmentation filtering based on object area

Feature extraction is performed directly on the original-resolution RGB data to preserve intensity fidelity.

---

## Notes and Limitations

- Optimized for **low-density cell wells**
- Parameters may require tuning for different magnifications or staining conditions
- Not designed for highly confluent or overlapping cells
- No machine learning or deep learning models are used

---

## Project Structure

```
cell-well-segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── cellwell/
│       └── pipeline.py
└── docs/
```

---

## License

This project is released under the **MIT License**.

---

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact: juaco2r@gmail.com
