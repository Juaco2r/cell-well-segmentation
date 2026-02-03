# cell-well-segmentation
Classical computer vision pipeline for instance segmentation and feature extraction of low-density cell wells from multi-channel fluorescence microscopy images, with CSV outputs, GeoJSON annotations, and visual previews.

# Cell Well Segmentation and Feature Extraction

Python pipeline for **instance segmentation and feature extraction of low-density cell
wells** from multi-channel fluorescence microscopy images.

The pipeline uses classical computer vision techniques (no deep learning) and is
designed to be memory-conscious while producing segmentation masks, per-cell features,
QuPath-compatible annotations, and visual summaries.

---

## Features

- Multi-channel TIFF handling
- Nuclei-based seed detection
- Watershed instance segmentation
- Per-cell morphological and intensity features
- CSV feature export
- GeoJSON annotations for QuPath
- Visualization previews for quality control

---

## Input Data Format

Expected input: **multi-channel TIFF image**

### Channel convention
| Channel index | Meaning |
|--------------|--------|
| 0 | Nuclear stain (DAPI / blue) |
| 1 | Marker channel (red) |
| 2 | Marker channel (green) |
| 3 | Cytoplasmic stain (CellCyto) |

⚠️ If your channel order differs, update `create_channel_images()` accordingly.

---

## Installation

```bash
git clone https://github.com/<username>/cell-well-segmentation.git
cd cell-well-segmentation
pip install -r requirements.txt
