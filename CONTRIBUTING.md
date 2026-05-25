# Contributing

Contributions are welcome. For now, please use GitHub issues for bug reports, feature requests, and reproducibility notes.

## Good bug reports include

- Operating system and Python version.
- Installation method.
- Image format and approximate image size.
- Channel order used.
- Full error message from the processing log.
- Whether the error occurs during loading, segmentation, GeoJSON export, Manders calculation, or validation.

## Development setup

```bash
git clone https://github.com/Juaco2r/cell-well-segmentation.git
cd cell-well-segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m cell_well_segmentation
```
