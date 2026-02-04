"""
Pipeline for segmentation and feature extraction on cell wells.

Created on Wed Jan 21 16:25:30 2026
Author: Jose J. Rodriguez Rojas
"""

from __future__ import annotations

import gc
import json
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import feature, filters, measure, morphology, segmentation
from skimage.measure import regionprops_table


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def cleanup_memory() -> None:
    """Force garbage collection and clear matplotlib cache."""
    plt.close("all")
    gc.collect()


def create_channel_images(
    stack: np.ndarray,
    xres,
    yres,
    output_folder: Path
) -> Tuple[Path, Path]:
    """Create RGB and CellCyto TIFF composites."""
    print("Creating channel composites...")

    # RGB: R=c1, G=c2, B=c0
    rgb_path = output_folder / "RGB.tif"
    rgb_stack = None
    try:
        rgb_r, rgb_g, rgb_b = stack[1], stack[2], stack[0]
        rgb_stack = np.stack([rgb_r, rgb_g, rgb_b], axis=-1)
        with tifffile.TiffWriter(rgb_path, bigtiff=True) as tif:
            tif.write(rgb_stack, resolution=(xres, yres), compression=None, photometric="rgb")
        print(f"  Saved: {rgb_path.name}")
    finally:
        # ensure variables exist before deleting (defensive)
        for v in ("rgb_r", "rgb_g", "rgb_b", "rgb_stack"):
            if v in locals():
                del locals()[v]
        gc.collect()

    # CellCyto: R=0, G=c3, B=c0
    cellcyto_path = output_folder / "CellCyto.tif"
    cellcyto_stack = None
    try:
        cyto_r, cyto_g, cyto_b = np.zeros_like(stack[0]), stack[3], stack[0]
        cellcyto_stack = np.stack([cyto_r, cyto_g, cyto_b], axis=-1)
        with tifffile.TiffWriter(cellcyto_path, bigtiff=True) as tif:
            tif.write(cellcyto_stack, resolution=(xres, yres), compression=None, photometric="rgb")
        print(f"  Saved: {cellcyto_path.name}")
    finally:
        for v in ("cyto_r", "cyto_g", "cyto_b", "cellcyto_stack"):
            if v in locals():
                del locals()[v]
        gc.collect()

    return cellcyto_path, rgb_path


def run_segmentation(
    cellcyto_path: Path,
    output_folder: Path,
    labels_shape: Tuple[int, int]
):
    """Run instance segmentation."""
    print("Running instance segmentation...")

    stack = None
    rgb_norm = None
    nuclei_norm = None
    blurred = None
    distance = None
    markers = None

    try:
        stack = tifffile.imread(cellcyto_path)

        # Normalize to float [0,1]
        if len(stack.shape) == 3 and stack.shape[2] == 3:  # HWC
            rgb_norm = np.transpose(stack, (2, 0, 1)).astype(np.float32) / 65535.0
        else:  # CHW
            rgb_norm = stack.astype(np.float32) / 65535.0 if stack.dtype == np.uint16 else stack

        blue_nuclei = rgb_norm[2]  # DAPI
        green_cyto = rgb_norm[1]   # CellCyto

        # Enhanced nuclei detection
        nuclei_norm = np.clip(blue_nuclei, 0, 1)
        blurred = filters.gaussian(nuclei_norm, sigma=2.5)

        # Seeds
        peaks_coords = feature.peak_local_max(
            blurred,
            min_distance=14,
            threshold_abs=0.04,
            threshold_rel=0.15,
            exclude_border=5,
        )
        seeds_bool = np.zeros(labels_shape, dtype=bool)
        if len(peaks_coords) > 0:
            seeds_bool[tuple(peaks_coords.T)] = True

        seeds = measure.label(seeds_bool)
        print(f"  Seeds detected: {len(peaks_coords)}")

        # Foreground
        otsu_thresh = filters.threshold_otsu(green_cyto)
        foreground = green_cyto > (otsu_thresh * 0.4)
        foreground = morphology.remove_small_holes(foreground, area_threshold=200)
        foreground = morphology.closing(foreground, morphology.disk(2))

        # Hybrid distance field
        distance_from_seeds = ndimage.distance_transform_edt(~seeds_bool & foreground)
        distance_border = ndimage.distance_transform_edt(~foreground)
        distance = distance_border * 0.3 + distance_from_seeds * 0.7

        # Markers
        markers = measure.label(seeds, connectivity=2)
        markers = morphology.dilation(markers, morphology.disk(2))

        # Watershed
        instance_labels = segmentation.watershed(-distance * 1.4, markers, mask=foreground)

        # Filter tiny objects
        props = regionprops_table(instance_labels, properties=["label", "area"])
        valid_labels = props["label"][props["area"] > 30]
        n_before = len(np.unique(instance_labels)) - 1
        instance_labels = np.isin(instance_labels, valid_labels) * instance_labels
        n_after = len(np.unique(instance_labels)) - 1
        print(f"  Valid cells: {n_before} → {n_after}")

        # Save instances
        inst_path = output_folder / "instances.tif"
        tifffile.imwrite(inst_path, instance_labels.astype(np.uint16))

        return instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, n_after

    finally:
        for v in ("stack", "rgb_norm", "nuclei_norm", "blurred", "distance", "markers"):
            if v in locals() and locals()[v] is not None:
                del locals()[v]
        gc.collect()


def extract_features(
    instance_labels: np.ndarray,
    rgb_path: Path,
    output_folder: Path
) -> pd.DataFrame:
    """Extract features from raw RGB."""
    print("Extracting cell features...")

    raw_stack = None
    rgb_hwc = None
    red = green = blue = None

    try:
        raw_stack = tifffile.imread(rgb_path)

        labels_shape = instance_labels.shape
        if len(raw_stack.shape) == 3 and raw_stack.shape[2] == 3:  # HWC
            rgb_hwc = raw_stack[: labels_shape[0], : labels_shape[1], :].astype(np.uint16)
        else:  # CHW
            rgb_hwc = raw_stack[:, : labels_shape[0], : labels_shape[1]].transpose(1, 2, 0).astype(np.uint16)

        print(f"  RGB ranges - R: {rgb_hwc[:,:,0].min()}-{rgb_hwc[:,:,0].max()}")
        print(f"                 G: {rgb_hwc[:,:,1].min()}-{rgb_hwc[:,:,1].max()}")
        print(f"                 B: {rgb_hwc[:,:,2].min()}-{rgb_hwc[:,:,2].max()}")

        red, green, blue = rgb_hwc[:, :, 0], rgb_hwc[:, :, 1], rgb_hwc[:, :, 2]

        props_shape = regionprops_table(instance_labels, properties=["label", "area", "perimeter"])
        props_red = regionprops_table(instance_labels, intensity_image=red, properties=["label", "max_intensity", "mean_intensity"])
        props_green = regionprops_table(instance_labels, intensity_image=green, properties=["label", "max_intensity", "mean_intensity"])
        props_blue = regionprops_table(instance_labels, intensity_image=blue, properties=["label", "max_intensity", "mean_intensity"])
        props_centroids = regionprops_table(instance_labels, properties=["label", "centroid"])

        df = pd.DataFrame({
            "label": props_shape["label"],
            "area_px": props_shape["area"],
            "perimeter_px": props_shape["perimeter"],
            "red_max": props_red["max_intensity"],
            "red_mean": props_red["mean_intensity"],
            "green_max": props_green["max_intensity"],
            "green_mean": props_green["mean_intensity"],
            "blue_max": props_blue["max_intensity"],
            "blue_mean": props_blue["mean_intensity"],
            "centroid_y": props_centroids["centroid-0"],
            "centroid_x": props_centroids["centroid-1"],
        })

        # Manual statistics
        for idx, label in enumerate(df["label"]):
            mask = instance_labels == label
            for channel, prefix in ((red, "red"), (green, "green"), (blue, "blue")):
                vals = channel[mask].ravel()
                df.at[idx, f"{prefix}_median"] = float(np.median(vals))
                df.at[idx, f"{prefix}_std"] = float(np.std(vals))
                df.at[idx, f"{prefix}_p25"] = float(np.percentile(vals, 25))
                df.at[idx, f"{prefix}_p75"] = float(np.percentile(vals, 75))

        # Derived features
        df["seg_intensity"] = df["green_mean"]
        df["red_cv"] = df["red_std"] / (df["red_mean"] + 1e-6)
        df["green_cv"] = df["green_std"] / (df["green_mean"] + 1e-6)
        df["blue_cv"] = df["blue_std"] / (df["blue_mean"] + 1e-6)
        df["red_total"] = df["red_mean"] * df["area_px"]
        df["green_total"] = df["green_mean"] * df["area_px"]
        df["blue_total"] = df["blue_mean"] * df["area_px"]

        # Filter artifacts
        df = df[(df["area_px"] > 100) & (df["area_px"] < 50000)].reset_index(drop=True)

        csv_path = output_folder / "cell_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved: {len(df)} cells")

        return df

    finally:
        for v in ("raw_stack", "rgb_hwc", "red", "green", "blue"):
            if v in locals() and locals()[v] is not None:
                del locals()[v]
        gc.collect()


def create_geojson_and_preview(
    instance_labels: np.ndarray,
    rgb_norm: np.ndarray,
    blue_nuclei: np.ndarray,
    green_cyto: np.ndarray,
    seeds_bool: np.ndarray,
    df: pd.DataFrame,
    output_folder: Path
) -> None:
    """Create GeoJSON and preview."""
    print("Generating GeoJSON...")

    features_list = []
    for cell_id in np.unique(instance_labels)[1:]:
        contours = measure.find_contours(instance_labels == cell_id, 0.0)
        if len(contours) > 0:
            contour = max(contours, key=len)
            if len(contour) > 2 and not np.allclose(contour[0], contour[-1], atol=1e-3):
                contour = np.vstack([contour, contour[0]])

            contour_px = np.round(contour).astype(int)
            polygon = [[int(col), int(row)] for row, col in contour_px]
            if polygon and polygon[0] != polygon[-1]:
                polygon[-1] = polygon[0]

            features_list.append({
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {"objectType": "annotation", "isLocked": False},
            })

    geojson_path = output_folder / "qupath_final.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features_list}, f, cls=NumpyEncoder, indent=2)

    print(f"  GEOJSON saved: {len(features_list)} polygons")

    print("Generating preview...")
    try:
        ds_factor = 4
        rgb_ds = rgb_norm.transpose(1, 2, 0) if len(rgb_norm.shape) == 3 else rgb_norm
        rgb_ds = rgb_ds[::ds_factor, ::ds_factor]
        instance_ds = instance_labels[::ds_factor, ::ds_factor]
        seeds_ds = ndimage.distance_transform_edt(seeds_bool)[::ds_factor, ::ds_factor]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(rgb_ds); axes[0, 0].set_title("CellCyto RGB"); axes[0, 0].axis("off")
        axes[0, 1].imshow(blue_nuclei[::ds_factor, ::ds_factor], cmap="Blues_r")
        axes[0, 1].contour(seeds_ds > 0, colors="red", linewidths=2)
        axes[0, 1].set_title("Nuclei + Seeds"); axes[0, 1].axis("off")
        axes[0, 2].imshow(green_cyto[::ds_factor, ::ds_factor], cmap="Greens")
        axes[0, 2].contour(instance_ds, levels=[0.5], colors="white", linewidths=1)
        axes[0, 2].set_title("Cyto + Contours"); axes[0, 2].axis("off")

        axes[1, 0].imshow(instance_ds, cmap="tab20"); axes[1, 0].set_title(f"Instances ({len(df)})"); axes[1, 0].axis("off")
        axes[1, 1].imshow(rgb_ds)
        for label in np.unique(instance_ds)[1 : min(20, len(np.unique(instance_ds)))]:
            axes[1, 1].contour(instance_ds == label, colors="C0", linewidths=0.8)
        axes[1, 1].set_title("RGB + Contours"); axes[1, 1].axis("off")

        axes[1, 2].imshow(rgb_ds)
        axes[1, 2].imshow(instance_ds, cmap="tab20", alpha=0.4)
        axes[1, 2].contour(seeds_ds > 0, colors="red", linewidths=1.5)
        axes[1, 2].set_title("Final Result"); axes[1, 2].axis("off")

        plt.tight_layout()
        preview_path = output_folder / "preview.png"
        plt.savefig(preview_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  Preview saved")

    finally:
        plt.close("all")
        gc.collect()


def process_single_file(original_path: Union[str, Path]) -> None:
    """Complete processing pipeline for a single TIFF file."""
    base_path = Path(original_path)
    output_folder = base_path.parent / base_path.stem
    output_folder.mkdir(exist_ok=True)

    print(f"\nProcessing: {base_path.name}")
    print(f"Output folder: {output_folder}")

    stack = None
    cellcyto_path = None
    rgb_path = None

    try:
        stack = tifffile.imread(str(original_path))
        print(f"Input shape: {stack.shape}")

        with tifffile.TiffFile(str(original_path)) as tif:
            xres = tif.pages[0].tags.get("XResolution").value
            yres = tif.pages[0].tags.get("YResolution").value

        labels_shape = stack.shape[1:3] if len(stack.shape) == 3 else stack.shape[1:]
        cellcyto_path, rgb_path = create_channel_images(stack, xres, yres, output_folder)

        instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, _n_after = run_segmentation(
            cellcyto_path, output_folder, labels_shape
        )

        df = extract_features(instance_labels, rgb_path, output_folder)

        create_geojson_and_preview(
            instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, df, output_folder
        )

        print(f"✅ COMPLETE: {output_folder.name} ({len(df)} cells)")
        print(f"   All files: {list(output_folder.glob('*'))}")

    except Exception as e:
        print(f"❌ ERROR {base_path.name}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # cleanup big arrays
        for v in ("stack", "cellcyto_path", "rgb_path"):
            if v in locals() and locals()[v] is not None:
                del locals()[v]
        cleanup_memory()


def run_batch(files: Iterable[Union[str, Path]]) -> int:
    """Process multiple files sequentially. Returns number of successfully processed inputs found."""
    processed = 0
    files_list = list(files)

    print("CELL SEGMENTATION PIPELINE")
    print("=" * 70)
    print(f"Processing {len(files_list)} files sequentially\n")

    for i, fpath in enumerate(files_list, 1):
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"File not found: {fpath.name}")
            continue

        print(f"[{i}/{len(files_list)}]")
        process_single_file(fpath)
        processed += 1
        print(f"Memory cleaned. Processed: {processed}/{len(files_list)}")
        print("-" * 70)

    print(f"\nPIPELINE COMPLETED: {processed}/{len(files_list)} successful")
    return processed


if __name__ == "__main__":
    # Example usage:
    # Replace these with your own file paths
    example_files = [
        r"C:\path\to\your\image1.tif",
        r"C:\path\to\your\image2.tif",
    ]
    run_batch(example_files)
