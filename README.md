# Cell Well Segmentation

**Cell Well Segmentation** is a Python/PyQt5 desktop application for immunofluorescence cell segmentation, per-cell feature extraction, Manders colocalization analysis, QuPath-compatible GeoJSON export, and optional DICE/IoU validation using ground-truth GeoJSON annotations.

The app is designed for microscopy and digital pathology workflows where images may be large, multichannel, and metadata-sensitive.

> First public GitHub-ready structure. Add screenshots, example images, and Zenodo DOI after the first release.

## Main features

- GUI-based single-image and bulk-image processing.
- Multichannel TIFF/OME-TIFF support through `tifffile`.
- WSI-style format support through OpenSlide when available, including SVS, NDPI, MRXS, SCN, VMS/VMU, BIF, SVSLIDE, and DICOM-like slide files.
- RGB fallback for standard PNG/JPEG/RGB images.
- ROI-based parameter exploration before full-image processing.
- Instance segmentation using nuclei seeding and watershed over cytoplasmic foreground.
- Per-cell measurements exported to CSV.
- Manders colocalization metrics exported per cell and as summary JSON.
- QuPath-compatible GeoJSON export.
- Resume/skip options for previously processed output folders.
- Optional DICE/IoU validation against GeoJSON ground truth.

## Expected input

The default channel mapping is:

| Parameter | Default channel index | Purpose |
|---|---:|---|
| Nuclei channel | 0 | Nuclei / seed detection |
| Red channel | 1 | Red biological signal |
| Green channel | 2 | Green biological signal |
| Cyto channel | 3 | Cytoplasmic / foreground segmentation signal |

Use **Custom** parameters if your images have a different channel order.

## Main outputs

For every processed image, the app creates one output folder named after the image stem. Main files include:

| Output | Description |
|---|---|
| `instances.tif` | Labeled instance mask; each segmented cell has a unique integer label. |
| `cell_features.csv` | Per-cell morphology and intensity features. |
| `manders_features.csv` | Per-cell Manders colocalization metrics. |
| `cell_features_with_manders.csv` | Combined features and Manders metrics. |
| `manders_summary.json` | Summary thresholds and processing metadata for Manders analysis. |
| `RGB.tif` | RGB composite from selected channels. |
| `CellCyto.tif` | Composite used for segmentation preview/processing. |
| `qupath_final.geojson` | QuPath-compatible segmentation polygons. |
| `preview.png` | Visual QC summary. |
| `validation/dice_report.csv` | Optional validation report when ground-truth GeoJSON is provided. |

## Installation from source

```bash
git clone https://github.com/Juaco2r/cell-well-segmentation.git
cd cell-well-segmentation
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m cell_well_segmentation
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m cell_well_segmentation
```

## OpenSlide support

For WSI formats such as SVS and NDPI, install OpenSlide support:

```bash
pip install openslide-python openslide-bin
```

If OpenSlide is unavailable, TIFF/OME-TIFF and standard raster images can still work through `tifffile` and Pillow.

## Windows executable build

Install development dependencies and run the packaging helper:

```bash
pip install -r requirements.txt
pip install pyinstaller
packaging\build_windows.bat
```

The executable will be created under `dist/`.

## Recommended repository structure

```text
cell-well-segmentation/
├─ src/cell_well_segmentation/
│  ├─ __init__.py
│  ├─ __main__.py
│  └─ app.py
├─ docs/
│  └─ zenodo_release_checklist.md
├─ packaging/
│  ├─ CellWellSegmentation.spec
│  └─ build_windows.bat
├─ .github/workflows/python-check.yml
├─ .gitignore
├─ CITATION.cff
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ CHANGELOG.md
├─ CONTRIBUTING.md
└─ .zenodo.json
```

## Citation

Use the `CITATION.cff` file in this repository. After creating the first Zenodo release, replace the placeholder DOI in `CITATION.cff`, `.zenodo.json`, and this README.

Suggested plain-text citation after DOI generation:

> Rodriguez Rojas JJ. Cell Well Segmentation: Immunofluorescence Cell Segmentation, Quantification and Validation. Version 1.0.0. Zenodo. 2026. doi: ADD_DOI_HERE.

## License

This project is released under the MIT License. See `LICENSE`.

## Author

José J. Rodriguez Rojas  
PhD Candidate, Universitat de Barcelona
