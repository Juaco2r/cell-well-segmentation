# Zenodo release checklist

1. Confirm the app name and version in `src/cell_well_segmentation/app.py`.
2. Confirm the same version in `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, and `CHANGELOG.md`.
3. Add screenshots or example output images to the README if desired.
4. Commit all changes.
5. Push to GitHub.
6. Create a GitHub release using a tag such as `v1.0.0`.
7. Let Zenodo archive the GitHub release.
8. Confirm DOI: 10.5281/zenodo.20387083.
9. Confirm badge: https://zenodo.org/badge/1149252759.svg.
8. Copy the generated DOI into:
   - `README.md`
   - `CITATION.cff`
   - `.zenodo.json`
9. Commit the DOI update.
10. For later versions, keep using the concept DOI for the project and version DOI for each release.
