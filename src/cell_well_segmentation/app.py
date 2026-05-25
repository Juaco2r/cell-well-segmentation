"""Cell Well Segmentation main GUI application.

This file is intentionally kept as a mostly single-file application so it can
be run directly during development and packaged with PyInstaller.
"""

import os
import sys
import gc
import csv
import json
import math
import uuid
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import filters, segmentation, morphology, measure, feature
from skimage.measure import regionprops_table

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QTextEdit, QMessageBox,
    QListWidget, QListWidgetItem, QSplitter, QScrollArea, QDialog, QDialogButtonBox,
    QAction
)


# ============================================================
# Robust JSON helpers
# ============================================================
# Previous versions of this script used cls=NumpyEncoder in GeoJSON export,
# but the class was missing in the final file. That caused:
#     name 'NumpyEncoder' is not defined
#
# These helpers make JSON export robust for GeoJSON and summaries.

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that safely converts numpy objects to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def make_json_safe(obj):
    """Recursively convert numpy/scientific Python objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def safe_json_dump(data, output_path, indent=2):
    """
    Save JSON robustly.

    First tries NumpyEncoder. If something still fails because of a non-standard
    object type, it recursively converts the data to JSON-safe values and retries.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=indent)
    except TypeError:
        safe_data = make_json_safe(data)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(safe_data, f, indent=indent)


# ============================================================
# App metadata
# ============================================================

APP_NAME = "Cell Well Segmentation"
APP_VERSION = "1.0.0"
APP_TITLE = "Cell Well Segmentation: Immunofluorescence Cell Segmentation, Quantification and Validation"
APP_AUTHOR = "José J. Rodriguez Rojas"
APP_YEAR = "2026"


# ============================================================
# Application icon / bundled resource helpers
# ============================================================

def resource_path(relative_path: str) -> str:
    """
    Return an absolute path to a bundled resource.

    Works when running:
      - directly from source code
      - as a PyInstaller onefile/onedir executable

    Expected project location when running from source:
      project_root/
        assets/icons/cell_well_segmentation_icon_option1.ico
        src/cell_well_segmentation/app.py
    """
    relative_path = str(relative_path).replace("\\", "/")

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        # app.py is usually in src/cell_well_segmentation/
        # parents[2] should be the project root.
        try:
            base_path = Path(__file__).resolve().parents[2]
        except Exception:
            base_path = Path.cwd()

    return str(base_path / relative_path)


def get_app_icon() -> QIcon:
    """
    Load the application icon.

    The .ico file is preferred on Windows and for PyInstaller builds.
    The .png fallback is useful during development or on macOS/Linux.
    """
    icon_candidates = [
        "assets/icons/cell_well_segmentation_icon_option1.ico",
        "assets/icons/cell_well_segmentation_icon_option1.png",
        "assets/icon/cell_well_segmentation_icon_option1.ico",
        "assets/icon/cell_well_segmentation_icon_option1.png",
    ]

    icon = QIcon()
    for rel_path in icon_candidates:
        full_path = Path(resource_path(rel_path))
        if full_path.exists() and full_path.is_file():
            icon = QIcon(str(full_path))
            if not icon.isNull():
                return icon

    return icon


# ============================================================
# OpenSlide DLL helper
# Same idea/structure as your TiffCropper code
# ============================================================

def _setup_openslide_dll_path():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        dll_dir = os.path.join(sys._MEIPASS, "openslide_bin")
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


_setup_openslide_dll_path()


# ============================================================
# Supported formats
# Same function/structure style as your TiffCropper code
# ============================================================

SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".svslide", ".dcm",
    ".jpg", ".jpeg", ".png"
)

OPENSLIDE_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".bif", ".svslide", ".dcm"
)

TIFF_EXTENSIONS = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".svs"
)

RASTER_EXTENSIONS = (
    ".jpg", ".jpeg", ".png"
)


def _has_ext(path_or_name, exts):
    name = str(path_or_name).lower()
    return any(name.endswith(e) for e in exts)


def _image_file_filter():
    exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
    return f"Image Files ({exts});;All Files (*)"


# ============================================================
# Optional imports
# Same function/structure style as your TiffCropper code
# ============================================================

def _try_import_openslide():
    try:
        import openslide
        return openslide
    except Exception:
        return None


def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


# ============================================================
# Metadata helpers
# Same function/structure style as your TiffCropper code
# ============================================================

def _mpp_to_dpi(mpp: float) -> float:
    return 25400.0 / float(mpp)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _ascii_safe(s: str) -> str:
    if s is None:
        return ""
    return str(s).encode("ascii", errors="ignore").decode("ascii")


def _tag_to_float(tag):
    if tag is None:
        return None
    v = tag.value
    try:
        if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None


def _resolution_to_mpp(xres: float, yres: float, unit: str):
    if not xres or not yres or not unit:
        return None
    unit = str(unit).upper()
    try:
        if unit == "INCH":
            return 25400.0 / float(xres), 25400.0 / float(yres)
        if unit == "CENTIMETER":
            return 10000.0 / float(xres), 10000.0 / float(yres)
    except Exception:
        return None
    return None


def _mpp_to_resolution_tuple(mpp_x: float, mpp_y: float):
    if not mpp_x or not mpp_y:
        return None
    try:
        return (_mpp_to_dpi(float(mpp_x)), _mpp_to_dpi(float(mpp_y)), "INCH")
    except Exception:
        return None


def _convert_physical_size_to_um(value, unit):
    try:
        value = float(value)
    except Exception:
        return None

    unit = str(unit or "um").strip().lower().replace("µ", "u")

    if unit in ("um", "micrometer", "micrometre", "micrometers", "micrometres"):
        return value
    if unit in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
        return value / 1000.0
    if unit in ("mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        return value * 1000.0
    if unit in ("cm", "centimeter", "centimetre", "centimeters", "centimetres"):
        return value * 10000.0
    if unit in ("m", "meter", "metre", "meters", "metres"):
        return value * 1000000.0

    return value


def _extract_ome_physical_size_um(ome_xml):
    if not ome_xml:
        return None
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(ome_xml)
        pixels = None
        for elem in root.iter():
            if elem.tag.endswith("Pixels"):
                pixels = elem
                break
        if pixels is None:
            return None
        psx = pixels.attrib.get("PhysicalSizeX")
        psy = pixels.attrib.get("PhysicalSizeY")
        if psx is None or psy is None:
            return None
        unit_x = pixels.attrib.get("PhysicalSizeXUnit", "um")
        unit_y = pixels.attrib.get("PhysicalSizeYUnit", "um")
        mpp_x = _convert_physical_size_to_um(psx, unit_x)
        mpp_y = _convert_physical_size_to_um(psy, unit_y)
        if mpp_x is None or mpp_y is None or mpp_x <= 0 or mpp_y <= 0:
            return None
        return float(mpp_x), float(mpp_y)
    except Exception:
        return None


def _ome_map_annotation_xml(kv: dict, ann_id: str = "Annotation:0") -> str:
    items = []
    for k, v in kv.items():
        k = xml_escape(_ascii_safe(k))
        v = xml_escape(_ascii_safe(v))
        items.append(f'<M K="{k}" V="{v}"/>')
    items_xml = "\n            ".join(items) if items else ""
    return f"""
    <StructuredAnnotations>
      <MapAnnotation ID="{ann_id}">
        <Value>
          <Map>
            {items_xml}
          </Map>
        </Value>
      </MapAnnotation>
    </StructuredAnnotations>
    """.strip()


def _build_ome_xml_rgb(size_x, size_y, physical_size_x_um, physical_size_y_um, image_name, annotation_kv=None):
    psx = f' PhysicalSizeX="{physical_size_x_um:.6f}" PhysicalSizeXUnit="um"' if physical_size_x_um else ""
    psy = f' PhysicalSizeY="{physical_size_y_um:.6f}" PhysicalSizeYUnit="um"' if physical_size_y_um else ""
    ann_xml = _ome_map_annotation_xml(annotation_kv, ann_id="Annotation:0") if annotation_kv else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="{xml_escape(_ascii_safe(image_name))}">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16"
            SizeX="{size_x}" SizeY="{size_y}" SizeC="3" SizeZ="1" SizeT="1"{psx}{psy}>
      <Channel ID="Channel:0" SamplesPerPixel="3"/>
      <TiffData IFD="0" PlaneCount="1"/>
    </Pixels>
  </Image>
  {ann_xml}
</OME>
"""


# ============================================================
# Image helpers
# ============================================================

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    arr_float = arr.astype(np.float32, copy=False)
    if arr_float.size == 0:
        return arr_float.astype(np.uint8)

    finite = np.isfinite(arr_float)
    if not np.any(finite):
        return np.zeros(arr_float.shape, dtype=np.uint8)

    valid = arr_float[finite]
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))

    if vmax <= vmin:
        vmax = float(np.max(valid))
        vmin = float(np.min(valid))
    if vmax <= vmin:
        return np.zeros(arr_float.shape, dtype=np.uint8)

    arr_float = np.clip((arr_float - vmin) / (vmax - vmin), 0, 1)
    return (arr_float * 255).astype(np.uint8)


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = _normalize_to_uint8(arr)
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = _normalize_to_uint8(arr[:, :, 0])
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 2:
            arr = _normalize_to_uint8(arr[:, :, 0])
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 3:
            arr = _normalize_to_uint8(arr)
        elif arr.shape[2] >= 4:
            arr = _normalize_to_uint8(arr[:, :, :3])
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image array shape after squeeze: {arr.shape}")

    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _downsample_for_preview(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    rgb = _to_uint8_rgb(rgb)
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    step = int(np.ceil(m / max_side))
    return rgb[::step, ::step, :]


def _numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(_to_uint8_rgb(rgb))
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def cleanup_memory():
    plt.close("all")
    gc.collect()


# ============================================================
# Image backend
# Same function/structure style as your TiffCropper code,
# extended with read_processing_stack() for this segmentation pipeline.
# ============================================================

class ImageBackend:
    def __init__(self):
        self.path = None
        self.path_obj = None
        self.reader = None
        self.file_kind = None
        self.slide_dims = None
        self.source_resolution = None
        self.source_mpp = None
        self.openslide_props = {}
        self._fail_log = []
        self._openslide_obj = None
        self._os_level_count = None
        self._os_downsamples = None
        self._os_level_dimensions = None
        self._tif_obj = None
        self._tif_series = None
        self._tif_axes = None
        self._zarr_array = None
        self._zarr_error = None

    def load(self, path: str):
        self.close()
        self.path = path
        self.path_obj = Path(path)
        self.reader = None
        self.file_kind = None
        self.slide_dims = None
        self.source_resolution = None
        self.source_mpp = None
        self.openslide_props = {}
        self._fail_log = []

        lower_name = self.path_obj.name.lower()

        if _has_ext(lower_name, OPENSLIDE_EXTENSIONS):
            try:
                w, h, res, mpp, props = self._probe_openslide(path)
                self.reader = "openslide"
                self.file_kind = "wsi"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = res
                self.source_mpp = mpp
                self.openslide_props = props or {}
                return self
            except Exception as e:
                self._fail_log.append(("OpenSlide", str(e)))

        if _has_ext(lower_name, TIFF_EXTENSIONS):
            try:
                w, h, res, mpp = self._probe_tifffile(path)
                self.reader = "tifffile"
                self.file_kind = "tiff"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = res
                self.source_mpp = mpp
                self.openslide_props = {}
                return self
            except Exception as e:
                self._fail_log.append(("tifffile", str(e)))

        if _has_ext(lower_name, RASTER_EXTENSIONS):
            try:
                arr = self._read_with_pil(path)
                h, w = arr.shape[:2]
                self.reader = "pil"
                self.file_kind = "raster"
                self.slide_dims = (int(w), int(h))
                self.source_resolution = None
                self.source_mpp = None
                self.openslide_props = {}
                return self
            except Exception as e:
                self._fail_log.append(("PIL", str(e)))

        msg = [f"Could not open image:\n{path}\n"]
        msg.append(f"Detected extension: {self.path_obj.suffix}")
        msg.append("\nTried the following readers:")
        for reader_name, err in self._fail_log:
            msg.append(f"\n- {reader_name} failed:\n  {err}")
        msg.append(
            "\n\nSupported extensions:\n"
            f"{SUPPORTED_EXTENSIONS}\n\n"
            "Suggestions:\n"
            "- Install OpenSlide support:\n"
            "  pip install openslide-python openslide-bin\n\n"
            "- For TIFF/OME-TIFF, make sure tifffile and zarr are installed:\n"
            "  pip install tifffile zarr\n"
        )
        raise RuntimeError("\n".join(msg))

    def close(self):
        if getattr(self, "_openslide_obj", None) is not None:
            try:
                self._openslide_obj.close()
            except Exception:
                pass
        self._openslide_obj = None
        if getattr(self, "_tif_obj", None) is not None:
            try:
                self._tif_obj.close()
            except Exception:
                pass
        self._tif_obj = None
        self._tif_series = None
        self._tif_axes = None
        self._zarr_array = None
        self._zarr_error = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_openslide(self):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError("OpenSlide not available.")
        if self._openslide_obj is None:
            self._openslide_obj = openslide.OpenSlide(self.path)
            self._os_level_count = self._openslide_obj.level_count
            self._os_downsamples = list(self._openslide_obj.level_downsamples)
            self._os_level_dimensions = list(self._openslide_obj.level_dimensions)
        return self._openslide_obj

    def _read_with_pil(self, path: str) -> np.ndarray:
        Image = _try_import_pil()
        if Image is None:
            raise RuntimeError("PIL/Pillow is not installed. Install with: pip install pillow")
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _probe_openslide(self, path: str):
        openslide = _try_import_openslide()
        if openslide is None:
            raise RuntimeError(
                "OpenSlide is not installed or cannot be imported.\n"
                "Install with:\n"
                "pip install openslide-python openslide-bin"
            )
        slide = openslide.OpenSlide(path)
        try:
            w, h = slide.dimensions
            props = dict(slide.properties or {})
            mpp_x = _safe_float(props.get("openslide.mpp-x"))
            mpp_y = _safe_float(props.get("openslide.mpp-y"))
            res_tuple = None
            mpp = None
            if mpp_x and mpp_y:
                res_tuple = (_mpp_to_dpi(mpp_x), _mpp_to_dpi(mpp_y), "INCH")
                mpp = (mpp_x, mpp_y)
            return int(w), int(h), res_tuple, mpp, props
        finally:
            try:
                slide.close()
            except Exception:
                pass

    def _probe_tifffile(self, path: str):
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("No image series found in TIFF/OME-TIFF.")

            ome_mpp = None
            try:
                ome_mpp = _extract_ome_physical_size_um(tif.ome_metadata)
            except Exception:
                ome_mpp = None

            s0 = tif.series[0]
            shape0 = s0.shape
            axes = getattr(s0, "axes", "")
            if "X" in axes and "Y" in axes:
                w = int(shape0[axes.index("X")])
                h = int(shape0[axes.index("Y")])
            else:
                if len(shape0) < 2:
                    raise ValueError(f"TIFF series has unsupported shape: {shape0}")
                h, w = int(shape0[-2]), int(shape0[-1])
                if len(shape0) >= 3 and shape0[-1] in (3, 4):
                    h, w = int(shape0[0]), int(shape0[1])

            page0 = s0.pages[0] if getattr(s0, "pages", None) else tif.pages[0]
            tags = page0.tags
            xres_f = _tag_to_float(tags.get("XResolution"))
            yres_f = _tag_to_float(tags.get("YResolution"))
            unit_str = None
            ru = tags.get("ResolutionUnit")
            if ru is not None:
                try:
                    u = int(ru.value)
                    if u == 2:
                        unit_str = "INCH"
                    elif u == 3:
                        unit_str = "CENTIMETER"
                except Exception:
                    unit_str = None

            res_tuple = (xres_f, yres_f, unit_str) if (xres_f and yres_f and unit_str) else None

            if ome_mpp is not None:
                mpp = ome_mpp
            else:
                mpp = _resolution_to_mpp(xres_f, yres_f, unit_str) if res_tuple else None

            if res_tuple is None and mpp is not None:
                res_tuple = _mpp_to_resolution_tuple(mpp[0], mpp[1])

            return int(w), int(h), res_tuple, mpp

    def input_thumbnail(self, max_side=512):
        if not self.path or not self.reader:
            raise RuntimeError("No image loaded.")

        if self.reader == "openslide":
            slide = self._get_openslide()
            try:
                img = slide.get_thumbnail((max_side, max_side)).convert("RGB")
                return _to_uint8_rgb(np.asarray(img, dtype=np.uint8))
            except Exception:
                lvl = slide.level_count - 1
                w, h = slide.level_dimensions[lvl]
                img = slide.read_region((0, 0), lvl, (int(w), int(h))).convert("RGB")
                return _downsample_for_preview(_to_uint8_rgb(np.asarray(img)), max_side=max_side)

        if self.reader == "pil":
            return _downsample_for_preview(_to_uint8_rgb(self._read_with_pil(self.path)), max_side=max_side)

        if self.reader == "tifffile":
            with tifffile.TiffFile(self.path) as tif:
                s0 = tif.series[0]
                try:
                    if hasattr(s0, "levels") and s0.levels:
                        arr = s0.levels[-1].asarray()
                    else:
                        arr = s0.asarray()
                except Exception:
                    arr = tif.pages[0].asarray()
            return _downsample_for_preview(_to_uint8_rgb(arr), max_side=max_side)

        raise RuntimeError(f"Unknown reader: {self.reader}")


    def _get_tiff_zarr(self):
        """Open a TIFF/OME-TIFF once and reuse its zarr view for ROI reads."""
        if self._tif_obj is None:
            self._tif_obj = tifffile.TiffFile(self.path)
            self._tif_series = self._tif_obj.series[0]
            self._tif_axes = getattr(self._tif_series, "axes", "")
            try:
                z = self._tif_series.aszarr()
                import zarr
                self._zarr_array = zarr.open(z, mode="r")
                self._zarr_error = None
            except Exception as e:
                self._zarr_array = None
                self._zarr_error = e
        return self._zarr_array, self._tif_series, self._tif_axes, self._zarr_error

    @staticmethod
    def _build_spatial_slicer_for_shape(shape, axes, x, y, w, h):
        """
        Build a robust ROI slicer for TIFF/zarr arrays.

        Why this was added:
        In Parameter Exploration, some TIFF/zarr files report axes in a way
        that does not clearly mark the channel dimension as C or S. The older
        ROI reader selected index 0 for unknown axes, which could turn a
        multichannel ROI into a single-channel array like (H, W). That caused:

            Input image is single-channel with shape (...).
            This pipeline needs 4 channels...

        This function preserves small channel-like axes, usually <= 20, with
        slice(None), so ROI reading keeps all channels whenever possible.
        """
        shape = tuple(int(s) for s in shape)
        ndim = len(shape)
        axes = axes or ""

        # Case 1: axes are known and match ndim.
        if axes and len(axes) == ndim and "Y" in axes and "X" in axes:
            slicer = []
            for ax, size in zip(axes, shape):
                ax = str(ax).upper()

                if ax == "Y":
                    slicer.append(slice(y, y + h))
                elif ax == "X":
                    slicer.append(slice(x, x + w))
                elif ax in ("C", "S"):
                    # Preserve all channels/samples.
                    slicer.append(slice(None))
                else:
                    # Robust improvement: if an unknown axis is small, it is
                    # probably channels/samples. Preserve it instead of taking 0.
                    if int(size) <= 20:
                        slicer.append(slice(None))
                    else:
                        # Z/T/scene-like dimension: use first plane.
                        slicer.append(0)
            return slicer

        # Case 2: no reliable axes metadata. Infer from array shape.
        if ndim == 2:
            # Y, X
            return [slice(y, y + h), slice(x, x + w)]

        if ndim == 3:
            # C, Y, X
            if shape[0] <= 20 and shape[1] > 20 and shape[2] > 20:
                return [slice(None), slice(y, y + h), slice(x, x + w)]

            # Y, X, C
            if shape[-1] <= 20 and shape[0] > 20 and shape[1] > 20:
                return [slice(y, y + h), slice(x, x + w), slice(None)]

            # Fallback for uncommon 3D arrays: assume Y, X, C.
            return [slice(y, y + h), slice(x, x + w), slice(None)]

        if ndim == 4:
            # Z/T, C, Y, X
            if shape[1] <= 20 and shape[2] > 20 and shape[3] > 20:
                return [0, slice(None), slice(y, y + h), slice(x, x + w)]

            # C, Z/T, Y, X
            if shape[0] <= 20 and shape[2] > 20 and shape[3] > 20:
                return [slice(None), 0, slice(y, y + h), slice(x, x + w)]

            # Z/T, Y, X, C
            if shape[-1] <= 20 and shape[1] > 20 and shape[2] > 20:
                return [0, slice(y, y + h), slice(x, x + w), slice(None)]

        raise ValueError(f"Cannot build spatial slicer for shape={shape}, axes={axes!r}")

    @staticmethod
    def _crop_loaded_array_spatial(arr, x, y, w, h):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr[y:y+h, x:x+w]
        if arr.ndim == 3:
            if arr.shape[-1] <= 20 and arr.shape[0] > 20 and arr.shape[1] > 20:
                return arr[y:y+h, x:x+w, :]
            if arr.shape[0] <= 20 and arr.shape[1] > 20 and arr.shape[2] > 20:
                return arr[:, y:y+h, x:x+w]
            return arr[y:y+h, x:x+w, :]
        if arr.ndim == 4:
            # Common cases supported by ensure_cyx_stack_for_pipeline.
            if arr.shape[-1] <= 20:
                return arr[0, y:y+h, x:x+w, :]
            if arr.shape[1] <= 20:
                return arr[0, :, y:y+h, x:x+w]
            if arr.shape[0] <= 20:
                return arr[:, 0, y:y+h, x:x+w]
        raise ValueError(f"Cannot crop loaded array with shape {arr.shape}")

    def read_processing_stack_region(self, x, y, w, h, max_region_pixels=25_000_000):
        """Read a rectangular ROI for fast parameter exploration.

        For TIFF/OME-TIFF this attempts ROI reading through tifffile/zarr first.
        If zarr is unavailable, it falls back to a full read and then crops.
        For OpenSlide/PIL it returns an RGB ROI, so RGB fallback must be enabled
        for processing those formats.
        """
        if not self.path or not self.reader or not self.slide_dims:
            raise RuntimeError("No image loaded.")

        full_w, full_h = self.slide_dims
        x = max(0, min(int(x), int(full_w) - 1))
        y = max(0, min(int(y), int(full_h) - 1))
        w = max(1, min(int(w), int(full_w) - x))
        h = max(1, min(int(h), int(full_h) - y))
        region_pixels = int(w) * int(h)
        if region_pixels > int(max_region_pixels):
            raise RuntimeError(
                f"Selected exploration rectangle is too large: {w} x {h} = {region_pixels:,} px.\n"
                f"Please select a smaller area or increase the exploration limit in the code."
            )

        metadata = {
            "reader": self.reader,
            "source_resolution": self.source_resolution,
            "source_mpp": self.source_mpp,
            "openslide_props": self.openslide_props,
        }

        if self.reader == "tifffile":
            za, series, axes, zarr_error = self._get_tiff_zarr()
            if za is not None:
                slicer = self._build_spatial_slicer_for_shape(za.shape, axes, x, y, w, h)
                arr = np.asarray(za[tuple(slicer)])

                # Extra safety for Parameter Exploration:
                # if the ROI read still returns a single-channel 2D crop from a
                # TIFF that likely contains multiple channels, fall back to a
                # full-series read and then crop while preserving the channels.
                # This keeps exploration consistent with full-image processing.
                if arr.ndim == 2:
                    try:
                        full_arr = series.asarray()
                        cropped = self._crop_loaded_array_spatial(full_arr, x, y, w, h)
                        if np.asarray(cropped).ndim > 2:
                            arr = cropped
                    except Exception:
                        # Keep the original 2D ROI; the later pipeline will raise
                        # a clear error if RGB fallback is not appropriate.
                        pass

                return arr, metadata

            arr = series.asarray()
            arr = self._crop_loaded_array_spatial(arr, x, y, w, h)
            return arr, metadata

        if self.reader == "pil":
            arr = self._read_with_pil(self.path)
            return arr[y:y+h, x:x+w, :], metadata

        if self.reader == "openslide":
            slide = self._get_openslide()
            img = slide.read_region((x, y), 0, (w, h)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            return arr, metadata

        raise RuntimeError(f"Unknown reader: {self.reader}")

    def read_processing_stack(self, max_full_read_pixels=250_000_000):
        """
        Read the image for the segmentation pipeline.

        Returns:
            stack_cyx_or_hwc, metadata dict

        For tifffile, this preserves multichannel arrays whenever possible.
        For OpenSlide/PIL, the returned image is RGB only, because those readers expose RGB.
        """
        if not self.path or not self.reader:
            raise RuntimeError("No image loaded.")

        full_w, full_h = self.slide_dims
        total_pixels = int(full_w) * int(full_h)
        if total_pixels > int(max_full_read_pixels):
            raise RuntimeError(
                f"Image is too large for full processing: {full_w} x {full_h} = {total_pixels:,} px.\n"
                f"Current safety limit: {int(max_full_read_pixels):,} px.\n"
                "Increase 'Max full-read pixels' only if your RAM is sufficient, or process a crop first."
            )

        if self.reader == "tifffile":
            arr = tifffile.imread(self.path)
            return arr, {
                "reader": self.reader,
                "source_resolution": self.source_resolution,
                "source_mpp": self.source_mpp,
                "openslide_props": self.openslide_props,
            }

        if self.reader == "pil":
            arr = self._read_with_pil(self.path)
            return arr, {
                "reader": self.reader,
                "source_resolution": self.source_resolution,
                "source_mpp": self.source_mpp,
                "openslide_props": self.openslide_props,
            }

        if self.reader == "openslide":
            slide = self._get_openslide()
            img = slide.read_region((0, 0), 0, (full_w, full_h)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            return arr, {
                "reader": self.reader,
                "source_resolution": self.source_resolution,
                "source_mpp": self.source_mpp,
                "openslide_props": self.openslide_props,
            }

        raise RuntimeError(f"Unknown reader: {self.reader}")


# ============================================================
# Parameters
# ============================================================

@dataclass
class PipelineParams:
    nuclei_channel: int = 0
    red_channel: int = 1
    green_channel: int = 2
    cyto_channel: int = 3

    use_rgb_fallback_if_needed: bool = False

    nuclei_gaussian_sigma: float = 2.5
    peak_min_distance: int = 14
    peak_threshold_abs: float = 0.04
    peak_threshold_rel: float = 0.15
    peak_exclude_border: int = 5

    foreground_otsu_factor: float = 0.4
    remove_small_holes_size: int = 200
    foreground_closing_radius: int = 2
    marker_dilation_radius: int = 2
    min_segmented_area_for_instance: int = 30

    min_cell_area_features: int = 100
    max_cell_area_features: int = 50000

    biological_red_threshold: float = 7000.0
    biological_green_threshold: float = 3500.0

    preview_downsample_factor: int = 4
    preview_dpi: int = 200
    max_full_read_pixels: int = 250_000_000

    save_intermediate_rgb_cellcyto: bool = True
    save_geojson: bool = True
    save_preview: bool = True

    # ------------------------------------------------------------
    # Existing output behavior
    # ------------------------------------------------------------
    # Reprocess from zero:
    #     Always recompute the full image and overwrite known outputs.
    # Skip completed:
    #     If the output folder already contains all expected outputs, skip it.
    #     If incomplete, process from scratch.
    # Resume missing outputs:
    #     If the output folder has enough intermediate files, regenerate missing
    #     post-processing outputs such as Manders or GeoJSON. If not enough data
    #     are available, process from scratch.
    existing_output_action: str = "Ask before run"

    # ------------------------------------------------------------
    # GeoJSON speed options
    # ------------------------------------------------------------
    # "Fast bounding-box" is the recommended default. It does NOT tile
    # the segmentation or change the mask. It only extracts each polygon
    # inside the cell's own bounding box, then adds the global X/Y offset.
    # This avoids the very slow operation of scanning the full image once
    # for every label.
    geojson_mode: str = "Fast bounding-box"  # "Fast bounding-box" or "Original full-mask"

    # 0.0 keeps the most detailed contour. Try 0.5 or 1.0 only if you want
    # smaller/faster-loading GeoJSON files in QuPath, with slight boundary
    # simplification.
    geojson_simplify_tolerance: float = 0.0

    # Progress message interval for GeoJSON polygon export.
    geojson_log_every: int = 250

    # ------------------------------------------------------------
    # Validation / DICE evaluation
    # ------------------------------------------------------------
    # Pixel-level validation compares the predicted binary cell mask
    # (instances.tif > 0) against a ground-truth GeoJSON rasterized to
    # the same image size. This does not alter the segmentation result.
    validation_enabled: bool = False
    validation_mode: str = "Single GeoJSON"  # "Single GeoJSON" or "Match by image name in folder"
    validation_geojson_path: str = ""
    validation_geojson_folder: str = ""
    validation_iou_threshold: float = 0.50
    save_validation_overlay: bool = True


def default_params() -> PipelineParams:
    return PipelineParams()


# ============================================================
# Pipeline array helpers
# ============================================================

def get_output_folder_from_original(original_path, output_parent=None):
    base_path = Path(original_path)
    parent = Path(output_parent) if output_parent else base_path.parent
    return parent / base_path.stem


def get_tiff_resolution_or_default_from_backend(metadata):
    res = metadata.get("source_resolution") if metadata else None
    if res is None:
        return None
    try:
        xres, yres, unit = res
        if unit:
            return (float(xres), float(yres)), unit
    except Exception:
        pass
    return None


def ensure_cyx_stack_for_pipeline(arr, params: PipelineParams):
    """
    Convert loaded array to C,Y,X.

    Supports common layouts:
      C,Y,X
      Y,X,C
      Z,C,Y,X or T,C,Y,X -> first non-channel dimension selected
      C,Z,Y,X -> first Z selected

    If RGB fallback is enabled and only 3 channels exist:
      nuclei = B
      red = R
      green = G
      cyto = G
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        if not params.use_rgb_fallback_if_needed:
            raise ValueError(
                f"Input image is single-channel with shape {arr.shape}. "
                "This pipeline needs 4 channels, or RGB fallback must be enabled."
            )
        cyx = np.stack([arr, arr, arr, arr], axis=0)
        return cyx

    if arr.ndim == 3:
        # C,Y,X
        if arr.shape[0] <= 20 and arr.shape[1] > 20 and arr.shape[2] > 20:
            cyx = arr
        # Y,X,C
        elif arr.shape[-1] <= 20 and arr.shape[0] > 20 and arr.shape[1] > 20:
            cyx = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Could not infer channel axis from shape {arr.shape}")

    elif arr.ndim == 4:
        # Try axes-free common cases.
        # If one axis has <=20 channels, use it as C and select index 0 from one extra axis.
        shape = arr.shape
        candidate_axes = [i for i, s in enumerate(shape) if s <= 20]
        if not candidate_axes:
            raise ValueError(f"Could not infer channel axis from 4D shape {arr.shape}")

        # Prefer axis 1 for Z,C,Y,X or T,C,Y,X; otherwise axis 0 for C,Z,Y,X.
        if 1 in candidate_axes and shape[-1] > 20 and shape[-2] > 20:
            # Z,C,Y,X or T,C,Y,X -> select first Z/T
            cyx = arr[0]
        elif 0 in candidate_axes and shape[-1] > 20 and shape[-2] > 20:
            # C,Z,Y,X -> select first Z
            cyx = arr[:, 0, :, :]
        elif shape[-1] <= 20:
            # Z,Y,X,C or T,Y,X,C -> select first Z/T and move channel last to first
            cyx = np.moveaxis(arr[0], -1, 0)
        else:
            raise ValueError(f"Could not infer channel axis from 4D shape {arr.shape}")
    else:
        raise ValueError(f"Unsupported image dimensions for this pipeline: {arr.shape}")

    n_ch = cyx.shape[0]
    max_required = max(params.nuclei_channel, params.red_channel, params.green_channel, params.cyto_channel)

    if n_ch <= max_required:
        if params.use_rgb_fallback_if_needed and n_ch >= 3:
            # RGB-style fallback: assume C,Y,X order where C0=R, C1=G, C2=B
            r = cyx[0]
            g = cyx[1]
            b = cyx[2]
            cyx = np.stack([b, r, g, g], axis=0)
            return cyx
        raise ValueError(
            f"Image has {n_ch} channel(s), but channel index {max_required} is requested.\n"
            "Use Custom parameters to change channel indices, or enable RGB fallback for RGB images."
        )

    return cyx


def save_label_image(label_image, output_path):
    max_label = int(label_image.max()) if label_image.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        out = label_image.astype(np.uint16, copy=False)
    else:
        out = label_image.astype(np.uint32, copy=False)
    tifffile.imwrite(output_path, out, bigtiff=True)


def write_rgb_tiff(path, rgb, metadata=None, image_name=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(rgb)
    if rgb.dtype == np.bool_:
        rgb = rgb.astype(np.uint16) * 65535
    elif not np.issubdtype(rgb.dtype, np.integer):
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 65535).astype(np.uint16)

    resolution = None
    resolutionunit = None
    source_mpp = None
    annotation = None
    if metadata:
        source_mpp = metadata.get("source_mpp")
        source_resolution = metadata.get("source_resolution")
        if source_resolution:
            try:
                xres, yres, unit = source_resolution
                if xres and yres and unit:
                    resolution = (float(xres), float(yres))
                    resolutionunit = unit
            except Exception:
                pass
        annotation = metadata.get("openslide_props") or None

    description = None
    if source_mpp:
        try:
            ome_xml = _build_ome_xml_rgb(
                size_x=int(rgb.shape[1]),
                size_y=int(rgb.shape[0]),
                physical_size_x_um=float(source_mpp[0]),
                physical_size_y_um=float(source_mpp[1]),
                image_name=image_name or path.stem,
                annotation_kv=annotation,
            )
            description = _ascii_safe(ome_xml)
        except Exception:
            description = _ascii_safe(f"Generated by {APP_NAME} v{APP_VERSION}")
    else:
        description = _ascii_safe(f"Generated by {APP_NAME} v{APP_VERSION}")

    kwargs = {
        "bigtiff": True,
        "photometric": "rgb",
        "metadata": None,
        "description": description,
    }
    if resolution is not None and resolutionunit is not None:
        kwargs["resolution"] = resolution
        kwargs["resolutionunit"] = resolutionunit

    tifffile.imwrite(str(path), rgb, **kwargs)


# ============================================================
# Create channel images
# ============================================================

def create_channel_images(stack_cyx, output_folder, params: PipelineParams, metadata=None, log=print):
    log("Creating channel composites...")
    output_folder = Path(output_folder)

    rgb_path = output_folder / "RGB.tif"
    cellcyto_path = output_folder / "CellCyto.tif"

    nuclei = stack_cyx[params.nuclei_channel]
    red = stack_cyx[params.red_channel]
    green = stack_cyx[params.green_channel]
    cyto = stack_cyx[params.cyto_channel]

    rgb_stack = np.stack([red, green, nuclei], axis=-1)
    cellcyto_stack = np.stack([np.zeros_like(nuclei), cyto, nuclei], axis=-1)

    write_rgb_tiff(rgb_path, rgb_stack, metadata=metadata, image_name="RGB")
    write_rgb_tiff(cellcyto_path, cellcyto_stack, metadata=metadata, image_name="CellCyto")

    log(f"  Saved: {rgb_path.name}")
    log(f"  Saved: {cellcyto_path.name}")

    return cellcyto_path, rgb_path, rgb_stack, cellcyto_stack


# ============================================================
# Segmentation
# ============================================================

def run_segmentation_from_cellcyto(cellcyto_stack_hwc, output_folder, params: PipelineParams, log=print):
    log("Running instance segmentation...")

    if cellcyto_stack_hwc.ndim != 3 or cellcyto_stack_hwc.shape[-1] != 3:
        raise ValueError(f"CellCyto image must be H,W,3. Got {cellcyto_stack_hwc.shape}")

    rgb_norm = np.moveaxis(cellcyto_stack_hwc, -1, 0).astype(np.float32)
    if np.issubdtype(cellcyto_stack_hwc.dtype, np.integer):
        max_possible = np.iinfo(cellcyto_stack_hwc.dtype).max
        if max_possible > 0:
            rgb_norm /= float(max_possible)
    else:
        max_val = np.nanmax(rgb_norm) if rgb_norm.size else 1.0
        if max_val > 1.0:
            rgb_norm /= max_val

    blue_nuclei = rgb_norm[2]
    green_cyto = rgb_norm[1]
    labels_shape = blue_nuclei.shape

    nuclei_norm = np.clip(blue_nuclei, 0, 1)
    blurred = filters.gaussian(nuclei_norm, sigma=params.nuclei_gaussian_sigma)

    peaks_coords = feature.peak_local_max(
        blurred,
        min_distance=params.peak_min_distance,
        threshold_abs=params.peak_threshold_abs,
        threshold_rel=params.peak_threshold_rel,
        exclude_border=params.peak_exclude_border,
    )

    seeds_bool = np.zeros(labels_shape, dtype=bool)
    if len(peaks_coords) > 0:
        seeds_bool[tuple(peaks_coords.T)] = True

    seeds = measure.label(seeds_bool)
    log(f"  Seeds detected: {len(peaks_coords)}")

    otsu_thresh = filters.threshold_otsu(green_cyto)
    foreground = green_cyto > (otsu_thresh * params.foreground_otsu_factor)
    foreground = morphology.remove_small_holes(
        foreground,
        area_threshold=params.remove_small_holes_size,
    )
    foreground = morphology.closing(
        foreground,
        morphology.disk(params.foreground_closing_radius),
    )

    distance_from_seeds = ndimage.distance_transform_edt(~seeds_bool & foreground)
    distance_border = ndimage.distance_transform_edt(~foreground)
    distance = distance_border * 0.3 + distance_from_seeds * 0.7

    markers = measure.label(seeds, connectivity=2)
    markers = morphology.dilation(
        markers,
        morphology.disk(params.marker_dilation_radius),
    )

    instance_labels = segmentation.watershed(
        -distance * 1.4,
        markers,
        mask=foreground,
    )

    props = regionprops_table(instance_labels, properties=["label", "area"])
    if len(props["label"]) > 0:
        valid_labels = props["label"][props["area"] > params.min_segmented_area_for_instance]
        n_before = len(np.unique(instance_labels)) - 1
        instance_labels = np.isin(instance_labels, valid_labels) * instance_labels
        n_after = len(np.unique(instance_labels)) - 1
    else:
        n_before = 0
        n_after = 0

    log(f"  Valid cells after instance filter: {n_before} -> {n_after}")

    inst_path = Path(output_folder) / "instances.tif"
    save_label_image(instance_labels, inst_path)
    log(f"  Saved: {inst_path.name}")

    return instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, n_after


# ============================================================
# Feature extraction
# ============================================================

def extract_features(instance_labels, rgb_stack_hwc, output_folder, params: PipelineParams, log=print):
    log("Extracting cell features...")

    rgb_hwc = np.asarray(rgb_stack_hwc)
    labels_shape = instance_labels.shape
    rgb_hwc = rgb_hwc[:labels_shape[0], :labels_shape[1], :]

    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"RGB image must be H,W,3. Got {rgb_hwc.shape}")

    red = rgb_hwc[:, :, 0]
    green = rgb_hwc[:, :, 1]
    blue = rgb_hwc[:, :, 2]

    log(f"  RGB ranges:")
    log(f"    R: {red.min()} - {red.max()}")
    log(f"    G: {green.min()} - {green.max()}")
    log(f"    B: {blue.min()} - {blue.max()}")

    props_shape = regionprops_table(instance_labels, properties=["label", "area", "perimeter"])

    if len(props_shape["label"]) == 0:
        df = pd.DataFrame(columns=[
            "label", "area_px", "perimeter_px",
            "red_max", "red_mean", "green_max", "green_mean", "blue_max", "blue_mean",
            "centroid_y", "centroid_x",
            "red_positive", "green_positive", "double_positive",
        ])
        csv_path = Path(output_folder) / "cell_features.csv"
        df.to_csv(csv_path, index=False)
        log("  No cells found. Empty cell_features.csv saved.")
        return df

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

    for idx, label in enumerate(df["label"]):
        mask = instance_labels == label
        for channel, prefix in [(red, "red"), (green, "green"), (blue, "blue")]:
            vals = channel[mask].ravel()
            df.at[idx, f"{prefix}_median"] = float(np.median(vals))
            df.at[idx, f"{prefix}_std"] = float(np.std(vals))
            df.at[idx, f"{prefix}_p25"] = float(np.percentile(vals, 25))
            df.at[idx, f"{prefix}_p75"] = float(np.percentile(vals, 75))

    df["seg_intensity"] = df["green_mean"]
    df["red_cv"] = df["red_std"] / (df["red_mean"] + 1e-6)
    df["green_cv"] = df["green_std"] / (df["green_mean"] + 1e-6)
    df["blue_cv"] = df["blue_std"] / (df["blue_mean"] + 1e-6)
    df["red_total"] = df["red_mean"] * df["area_px"]
    df["green_total"] = df["green_mean"] * df["area_px"]
    df["blue_total"] = df["blue_mean"] * df["area_px"]

    # ------------------------------------------------------------
    # Biological positivity flags
    # ------------------------------------------------------------
    # These columns are annotations for downstream analysis. They do not change
    # segmentation, cell filtering, or Manders computation. A cell is marked as
    # red_positive/green_positive when its mean channel intensity is greater than
    # or equal to the user-defined biological threshold. double_positive means
    # both red_positive and green_positive are True.
    df["red_positive"] = df["red_mean"] >= float(params.biological_red_threshold)
    df["green_positive"] = df["green_mean"] >= float(params.biological_green_threshold)
    df["double_positive"] = df["red_positive"] & df["green_positive"]

    before_filter = len(df)
    df = df[
        (df["area_px"] > params.min_cell_area_features) &
        (df["area_px"] < params.max_cell_area_features)
    ].reset_index(drop=True)
    after_filter = len(df)

    csv_path = Path(output_folder) / "cell_features.csv"
    df.to_csv(csv_path, index=False)

    log(f"  Feature cells: {before_filter} -> {after_filter}")
    log(f"  Saved: {csv_path.name}")

    return df


# ============================================================
# Manders colocalization
# ============================================================

def safe_otsu(values):
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    vmin = values.min()
    vmax = values.max()
    if vmin == vmax:
        return float(vmin)
    return float(filters.threshold_otsu(values))


def get_global_thresholds(red, green):
    red_nonzero = red[red > 0]
    green_nonzero = green[green > 0]
    red_thr = safe_otsu(red_nonzero) if red_nonzero.size > 0 else 0.0
    green_thr = safe_otsu(green_nonzero) if green_nonzero.size > 0 else 0.0
    return float(red_thr), float(green_thr)


def compute_manders(red_vals, green_vals, red_thr, green_thr):
    red_vals = red_vals.astype(np.float64, copy=False)
    green_vals = green_vals.astype(np.float64, copy=False)

    red_sum = red_vals.sum()
    green_sum = green_vals.sum()

    red_pos = red_vals > red_thr
    green_pos = green_vals > green_thr
    both_pos = red_pos & green_pos

    manders_red_in_green = red_vals[green_pos].sum() / red_sum if red_sum > 0 else np.nan
    manders_green_in_red = green_vals[red_pos].sum() / green_sum if green_sum > 0 else np.nan

    return {
        "manders_red_in_green": float(manders_red_in_green),
        "manders_green_in_red": float(manders_green_in_red),
        "overlap_fraction_pixels": float(np.mean(both_pos)) if both_pos.size > 0 else np.nan,
        "red_positive_fraction": float(np.mean(red_pos)) if red_pos.size > 0 else np.nan,
        "green_positive_fraction": float(np.mean(green_pos)) if green_pos.size > 0 else np.nan,
    }


def compute_manders_for_all_cells(instance_labels, rgb_hwc, df_features, params: PipelineParams, log=print):
    red = rgb_hwc[:, :, 0]
    green = rgb_hwc[:, :, 1]

    global_red_thr, global_green_thr = get_global_thresholds(red, green)

    if df_features is not None and not df_features.empty and "label" in df_features.columns:
        labels = df_features["label"].astype(int).values
    else:
        labels = np.unique(instance_labels)
        labels = labels[labels > 0]

    log("Computing Manders metrics...")
    log(f"  Global thresholds: red={global_red_thr:.2f}, green={global_green_thr:.2f}")
    log(f"  Biological thresholds: red={params.biological_red_threshold:.2f}, green={params.biological_green_threshold:.2f}")
    log(f"  Labels to process: {len(labels)}")

    rows = []

    for i, label in enumerate(labels, 1):
        mask = instance_labels == label
        red_vals = red[mask].ravel()
        green_vals = green[mask].ravel()

        if red_vals.size == 0 or green_vals.size == 0:
            continue

        global_metrics = compute_manders(red_vals, green_vals, global_red_thr, global_green_thr)

        cell_red_thr = safe_otsu(red_vals)
        cell_green_thr = safe_otsu(green_vals)
        percell_metrics = compute_manders(red_vals, green_vals, cell_red_thr, cell_green_thr)

        bio_metrics = compute_manders(
            red_vals,
            green_vals,
            params.biological_red_threshold,
            params.biological_green_threshold,
        )

        red_mean = float(np.mean(red_vals))
        green_mean = float(np.mean(green_vals))
        red_positive = bool(red_mean >= float(params.biological_red_threshold))
        green_positive = bool(green_mean >= float(params.biological_green_threshold))

        rows.append({
            "label": int(label),
            "red_mean_for_positivity": red_mean,
            "green_mean_for_positivity": green_mean,
            "red_positive": red_positive,
            "green_positive": green_positive,
            "double_positive": bool(red_positive and green_positive),
            "global_red_threshold": float(global_red_thr),
            "global_green_threshold": float(global_green_thr),
            "cell_otsu_red_threshold": float(cell_red_thr),
            "cell_otsu_green_threshold": float(cell_green_thr),
            "biological_red_threshold": float(params.biological_red_threshold),
            "biological_green_threshold": float(params.biological_green_threshold),
            "manders_global_red_in_green": global_metrics["manders_red_in_green"],
            "manders_global_green_in_red": global_metrics["manders_green_in_red"],
            "manders_global_overlap_fraction_pixels": global_metrics["overlap_fraction_pixels"],
            "manders_global_red_positive_fraction": global_metrics["red_positive_fraction"],
            "manders_global_green_positive_fraction": global_metrics["green_positive_fraction"],
            "manders_cellotsu_red_in_green": percell_metrics["manders_red_in_green"],
            "manders_cellotsu_green_in_red": percell_metrics["manders_green_in_red"],
            "manders_cellotsu_overlap_fraction_pixels": percell_metrics["overlap_fraction_pixels"],
            "manders_cellotsu_red_positive_fraction": percell_metrics["red_positive_fraction"],
            "manders_cellotsu_green_positive_fraction": percell_metrics["green_positive_fraction"],
            "manders_biological_red_in_green": bio_metrics["manders_red_in_green"],
            "manders_biological_green_in_red": bio_metrics["manders_green_in_red"],
            "manders_biological_overlap_fraction_pixels": bio_metrics["overlap_fraction_pixels"],
            "manders_biological_red_positive_fraction": bio_metrics["red_positive_fraction"],
            "manders_biological_green_positive_fraction": bio_metrics["green_positive_fraction"],
        })

        if i % 500 == 0:
            log(f"    Processed {i}/{len(labels)} cells...")

    df_manders = pd.DataFrame(rows)
    summary = {
        "n_cells_processed": int(len(df_manders)),
        "n_red_positive": int(df_manders["red_positive"].sum()) if "red_positive" in df_manders.columns else 0,
        "n_green_positive": int(df_manders["green_positive"].sum()) if "green_positive" in df_manders.columns else 0,
        "n_double_positive": int(df_manders["double_positive"].sum()) if "double_positive" in df_manders.columns else 0,
        "global_red_threshold": float(global_red_thr),
        "global_green_threshold": float(global_green_thr),
        "biological_red_threshold": float(params.biological_red_threshold),
        "biological_green_threshold": float(params.biological_green_threshold),
    }
    return df_manders, summary


def compute_and_save_manders(instance_labels, rgb_stack_hwc, df_features, output_folder, params: PipelineParams, log=print):
    output_folder = Path(output_folder)
    rgb_hwc = np.asarray(rgb_stack_hwc)[:instance_labels.shape[0], :instance_labels.shape[1], :]

    if df_features is None or df_features.empty:
        log("Skipping Manders: no valid cells in cell_features.csv")
        empty = pd.DataFrame()
        empty.to_csv(output_folder / "manders_features.csv", index=False)
        if df_features is None:
            df_features = pd.DataFrame()
        df_features.to_csv(output_folder / "cell_features_with_manders.csv", index=False)
        summary = {
            "n_cells_processed": 0,
            "global_red_threshold": None,
            "global_green_threshold": None,
            "biological_red_threshold": float(params.biological_red_threshold),
            "biological_green_threshold": float(params.biological_green_threshold),
        }
        safe_json_dump(summary, output_folder / "manders_summary.json", indent=2)
        return empty, df_features, summary

    df_manders, summary = compute_manders_for_all_cells(
        instance_labels,
        rgb_hwc,
        df_features,
        params,
        log=log,
    )

    manders_csv_path = output_folder / "manders_features.csv"
    df_manders.to_csv(manders_csv_path, index=False)

    df_merged = df_features.merge(df_manders, on="label", how="left")
    merged_csv_path = output_folder / "cell_features_with_manders.csv"
    df_merged.to_csv(merged_csv_path, index=False)

    summary_json_path = output_folder / "manders_summary.json"
    safe_json_dump(summary, summary_json_path, indent=2)

    log("Manders outputs saved:")
    log(f"  - {manders_csv_path.name}")
    log(f"  - {merged_csv_path.name}")
    log(f"  - {summary_json_path.name}")

    return df_manders, df_merged, summary


# ============================================================
# GeoJSON + preview
# ============================================================


def _signed_polygon_area(polygon):
    """Return signed polygon area from [[x, y], ...] coordinates."""
    if polygon is None or len(polygon) < 4:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(polygon[:-1], polygon[1:]):
        area += (float(x0) * float(y1)) - (float(x1) * float(y0))
    return area / 2.0


def _qupath_original_style_feature_from_contour(contour):
    """
    Build a QuPath-compatible feature using the SAME polygon style as your
    original working script.

    Important:
    The previous fast versions used padding + contour level 0.5 + extra
    geometry cleaning. That produced valid-looking GeoJSON, but QuPath can
    still reject some polygons with:
        "Reduction failed, possible invalid input"

    This function intentionally mimics your original working exporter:
      - skimage contour in [row, col]
      - append the first point if contour is not closed
      - round to integer pixels
      - convert to GeoJSON [x, y]
      - if needed, replace the last point with the first point

    The speed improvement now comes only from how the contour is found:
    a local bounding-box crop with a 1-pixel margin is used instead of scanning
    the whole image for every label. The polygon construction itself remains
    original-style for QuPath compatibility.
    """
    if contour is None or len(contour) <= 2:
        return None

    if not np.allclose(contour[0], contour[-1], atol=1e-3):
        contour = np.vstack([contour, contour[0]])

    contour_px = np.round(contour).astype(int)
    polygon = [[int(col), int(row)] for row, col in contour_px]

    if len(polygon) < 4:
        return None

    if polygon and polygon[0] != polygon[-1]:
        # This matches your original working code behavior.
        polygon[-1] = polygon[0]

    # Minimal sanity check only. Do not aggressively simplify/clean because
    # that was the source of some QuPath import failures.
    unique_points = set((p[0], p[1]) for p in polygon[:-1])
    if len(unique_points) < 3:
        return None

    area = abs(_signed_polygon_area(polygon))
    if area < 0.5:
        return None

    return {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon]
        },
        "properties": {
            "objectType": "annotation",
            "isLocked": False
        },
    }


def _qupath_feature_from_contour(contour):
    """
    Backwards-compatible alias.
    Uses the original-style polygon construction that has been the most
    compatible with QuPath in your earlier scripts.
    """
    return _qupath_original_style_feature_from_contour(contour)


def create_geojson_fast_from_labels(instance_labels, df, output_folder, params: PipelineParams, log=print):
    """
    Fast QuPath-compatible GeoJSON export from the finished instance mask.

    This version is designed to generate the SAME style of GeoJSON as your
    original working full-mask exporter, but faster.

    How it works:
      1. The final segmentation mask is already complete and unchanged.
      2. For each label, get its bounding box with ndimage.find_objects().
      3. Add a 1-pixel margin around the bounding box when possible. This gives
         find_contours() local background context similar to the full image.
      4. Run find_contours(local_mask, 0.0), matching the original working code.
      5. Add the local crop offset back to the contour.
      6. Build the polygon using the original QuPath-compatible construction.

    This does NOT tile segmentation and does NOT cut cells. It only avoids
    scanning the full 100+ million pixel image once for every label.
    """
    output_folder = Path(output_folder)
    geojson_path = output_folder / "qupath_final.geojson"

    if df is not None and not df.empty and "label" in df.columns:
        labels = df["label"].astype(int).to_numpy()
    else:
        labels = np.unique(instance_labels)
        labels = labels[labels > 0].astype(int)

    labels = labels[labels > 0]
    h_img, w_img = instance_labels.shape[:2]

    log(f"Generating fast QuPath-compatible GeoJSON for {len(labels)} labels...")

    object_slices = ndimage.find_objects(instance_labels)

    features = []
    # In this QuPath-compatible mode, tolerance 0.0 is strongly recommended.
    # If tolerance > 0, simplification is applied before original-style polygon
    # conversion, but if QuPath import fails, set it back to 0.0.
    tol = float(getattr(params, "geojson_simplify_tolerance", 0.0) or 0.0)
    log_every = max(1, int(getattr(params, "geojson_log_every", 250) or 250))

    skipped = 0

    for i, cell_id in enumerate(labels, 1):
        cell_id = int(cell_id)

        if cell_id - 1 < 0 or cell_id - 1 >= len(object_slices):
            skipped += 1
            continue

        slc = object_slices[cell_id - 1]
        if slc is None:
            skipped += 1
            continue

        y_slice, x_slice = slc
        y0, y1 = int(y_slice.start), int(y_slice.stop)
        x0, x1 = int(x_slice.start), int(x_slice.stop)

        # Add one-pixel background context around the cell bbox whenever possible.
        # This is the key to matching full-mask contour behavior locally.
        yy0 = max(0, y0 - 1)
        yy1 = min(h_img, y1 + 1)
        xx0 = max(0, x0 - 1)
        xx1 = min(w_img, x1 + 1)

        local_mask = (instance_labels[yy0:yy1, xx0:xx1] == cell_id)

        if not np.any(local_mask):
            skipped += 1
            continue

        # Match the original working code: binary mask + contour level 0.0.
        contours = measure.find_contours(local_mask, 0.0)
        if not contours:
            skipped += 1
            continue

        contour = max(contours, key=len)

        # Convert local contour coordinates back to full-image coordinates.
        # contour[:, 0] is row/y; contour[:, 1] is col/x.
        contour[:, 0] = contour[:, 0] + yy0
        contour[:, 1] = contour[:, 1] + xx0

        if tol > 0:
            contour = measure.approximate_polygon(contour, tolerance=tol)

        feat = _qupath_original_style_feature_from_contour(contour)
        if feat is not None:
            features.append(feat)
        else:
            skipped += 1

        if i % log_every == 0:
            log(f"  GeoJSON polygons processed: {i}/{len(labels)} | exported: {len(features)}")

    safe_json_dump({"type": "FeatureCollection", "features": features}, geojson_path, indent=2)

    log(f"  Fast GeoJSON saved: {geojson_path.name}")
    log(f"  Polygons exported: {len(features)}")
    if skipped:
        log(f"  Polygons skipped by sanity checks: {skipped}")
    return geojson_path


def create_geojson_original_full_mask(instance_labels, df, output_folder, params: PipelineParams, log=print):
    """
    Original GeoJSON export kept as a compatibility/debug option.

    This uses the old method that scans the full image for every label:
        instance_labels == cell_id

    It can be very slow on large images. Prefer Fast bounding-box mode.
    """
    output_folder = Path(output_folder)
    geojson_path = output_folder / "qupath_final.geojson"

    valid_labels = set(df["label"].astype(int).tolist()) if df is not None and not df.empty else set()
    labels = np.unique(instance_labels)
    labels = labels[labels > 0]
    log(f"Generating original full-mask GeoJSON for {len(labels)} labels...")

    features = []
    tol = float(getattr(params, "geojson_simplify_tolerance", 0.0) or 0.0)
    log_every = max(1, int(getattr(params, "geojson_log_every", 250) or 250))

    for i, cell_id in enumerate(labels, 1):
        cell_id = int(cell_id)
        if valid_labels and cell_id not in valid_labels:
            continue

        contours = measure.find_contours(instance_labels == cell_id, 0.5)
        if not contours:
            continue

        contour = max(contours, key=len)
        if tol > 0:
            contour = measure.approximate_polygon(contour, tolerance=tol)

        feat = _qupath_feature_from_contour(contour)
        if feat is not None:
            features.append(feat)

        if i % log_every == 0:
            log(f"  GeoJSON polygons processed: {i}/{len(labels)}")

    safe_json_dump({"type": "FeatureCollection", "features": features}, geojson_path, indent=2)

    log(f"  Original GeoJSON saved: {geojson_path.name}")
    log(f"  Polygons exported: {len(features)}")
    return geojson_path


def create_geojson_and_preview(instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, df, output_folder, params: PipelineParams, log=print):
    output_folder = Path(output_folder)

    if params.save_geojson:
        mode = str(getattr(params, "geojson_mode", "Fast bounding-box"))

        if mode.startswith("Original"):
            create_geojson_original_full_mask(
                instance_labels=instance_labels,
                df=df,
                output_folder=output_folder,
                params=params,
                log=log,
            )
        else:
            create_geojson_fast_from_labels(
                instance_labels=instance_labels,
                df=df,
                output_folder=output_folder,
                params=params,
                log=log,
            )

    if params.save_preview:
        log("Generating preview...")
        try:
            ds = max(1, int(params.preview_downsample_factor))
            rgb_ds = rgb_norm.transpose(1, 2, 0)[::ds, ::ds]
            instance_ds = instance_labels[::ds, ::ds]
            seeds_ds = ndimage.distance_transform_edt(seeds_bool)[::ds, ::ds]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            axes[0, 0].imshow(rgb_ds)
            axes[0, 0].set_title("CellCyto RGB")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(blue_nuclei[::ds, ::ds], cmap="Blues_r")
            axes[0, 1].contour(seeds_ds > 0, colors="red", linewidths=2)
            axes[0, 1].set_title("Nuclei + Seeds")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(green_cyto[::ds, ::ds], cmap="Greens")
            axes[0, 2].contour(instance_ds, levels=[0.5], colors="white", linewidths=1)
            axes[0, 2].set_title("Cyto + Contours")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(instance_ds, cmap="tab20")
            axes[1, 0].set_title(f"Instances / valid features: {len(df)}")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(rgb_ds)
            unique_ds_labels = np.unique(instance_ds)
            unique_ds_labels = unique_ds_labels[unique_ds_labels > 0]
            for label in unique_ds_labels[:20]:
                axes[1, 1].contour(instance_ds == label, colors="C0", linewidths=0.8)
            axes[1, 1].set_title("RGB + Example Contours")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(rgb_ds)
            axes[1, 2].imshow(instance_ds, cmap="tab20", alpha=0.4)
            axes[1, 2].contour(seeds_ds > 0, colors="red", linewidths=1.5)
            axes[1, 2].set_title("Final Result")
            axes[1, 2].axis("off")

            plt.tight_layout()
            preview_path = output_folder / "preview.png"
            plt.savefig(preview_path, dpi=int(params.preview_dpi), bbox_inches="tight", facecolor="white")
            plt.close(fig)
            log(f"  Preview saved: {preview_path.name}")
        finally:
            plt.close("all")
            gc.collect()





# ============================================================
# Validation / DICE helpers
# ============================================================

def find_validation_geojson_for_image(image_path, params: PipelineParams):
    """Return the ground-truth GeoJSON path for this image, or None.

    Modes:
      - Single GeoJSON: use params.validation_geojson_path for the selected image.
      - Match by image name in folder: search for files named like the image stem.
    """
    if not bool(getattr(params, "validation_enabled", False)):
        return None

    mode = str(getattr(params, "validation_mode", "Single GeoJSON"))
    image_path = Path(image_path)

    if mode.startswith("Single"):
        gt = str(getattr(params, "validation_geojson_path", "") or "").strip()
        return Path(gt) if gt and Path(gt).exists() else None

    folder = str(getattr(params, "validation_geojson_folder", "") or "").strip()
    if not folder:
        return None
    folder = Path(folder)
    if not folder.exists():
        return None

    candidates = [
        folder / f"{image_path.stem}.geojson",
        folder / f"{image_path.stem}.json",
        folder / image_path.stem / "qupath_final.geojson",
        folder / image_path.stem / f"{image_path.stem}.geojson",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Flexible fallback: any geojson/json that contains the image stem.
    matches = sorted(list(folder.glob(f"*{image_path.stem}*.geojson")) + list(folder.glob(f"*{image_path.stem}*.json")))
    return matches[0] if matches else None


def _iter_geojson_polygons(geojson_data):
    """Yield polygon rings from GeoJSON data.

    Yields tuples (exterior, holes), where exterior and holes are coordinate
    arrays in GeoJSON order [x, y]. Supports FeatureCollection, Feature,
    Polygon and MultiPolygon.
    """
    if not geojson_data:
        return

    typ = geojson_data.get("type") if isinstance(geojson_data, dict) else None

    if typ == "FeatureCollection":
        for feat in geojson_data.get("features", []):
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            yield from _iter_geojson_polygons(geom)

    elif typ == "Feature":
        yield from _iter_geojson_polygons(geojson_data.get("geometry", {}))

    elif typ == "Polygon":
        coords = geojson_data.get("coordinates", [])
        if coords and len(coords[0]) >= 3:
            exterior = coords[0]
            holes = coords[1:] if len(coords) > 1 else []
            yield exterior, holes

    elif typ == "MultiPolygon":
        for poly in geojson_data.get("coordinates", []):
            if poly and len(poly[0]) >= 3:
                exterior = poly[0]
                holes = poly[1:] if len(poly) > 1 else []
                yield exterior, holes


def rasterize_geojson_to_mask(geojson_path, shape, log=print):
    """Rasterize a GeoJSON annotation file into a binary mask.

    GeoJSON coordinates are [x, y], while NumPy masks are [row, col] = [y, x].
    Holes are supported when present, although QuPath annotations often do not
    contain holes.
    """
    from skimage.draw import polygon as draw_polygon

    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError(f"Validation mask shape must be 2D. Got: {shape}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    n_poly = 0

    for exterior, holes in _iter_geojson_polygons(data):
        try:
            pts = np.asarray(exterior, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            xs = np.clip(pts[:, 0], 0, w - 1)
            ys = np.clip(pts[:, 1], 0, h - 1)
            rr, cc = draw_polygon(ys, xs, shape=(h, w))
            mask[rr, cc] = True
            n_poly += 1

            for hole in holes:
                hpts = np.asarray(hole, dtype=np.float64)
                if hpts.ndim != 2 or hpts.shape[0] < 3 or hpts.shape[1] < 2:
                    continue
                hxs = np.clip(hpts[:, 0], 0, w - 1)
                hys = np.clip(hpts[:, 1], 0, h - 1)
                hrr, hcc = draw_polygon(hys, hxs, shape=(h, w))
                mask[hrr, hcc] = False
        except Exception:
            continue

    if n_poly == 0:
        log(f"WARNING: Validation GeoJSON had no usable polygons: {geojson_path}")

    return mask, int(n_poly)


def rasterize_geojson_to_roi_mask(geojson_path, roi_x, roi_y, roi_w, roi_h, log=print):
    """Rasterize full-image GeoJSON annotations into a selected ROI mask.

    The input GeoJSON must be in the same coordinate system as the original
    image. Coordinates are shifted from full-image space to ROI-local space by
    subtracting roi_x and roi_y. This allows Parameter Exploration to calculate
    DICE/IoU using a complete QuPath GeoJSON without creating a separate ROI
    GeoJSON file.
    """
    from skimage.draw import polygon as draw_polygon

    roi_x = int(roi_x)
    roi_y = int(roi_y)
    roi_w = int(roi_w)
    roi_h = int(roi_h)

    if roi_w <= 0 or roi_h <= 0:
        raise ValueError(f"ROI size must be positive. Got W={roi_w}, H={roi_h}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((roi_h, roi_w), dtype=bool)
    n_poly = 0

    roi_x1 = roi_x + roi_w
    roi_y1 = roi_y + roi_h

    for exterior, holes in _iter_geojson_polygons(data):
        try:
            pts = np.asarray(exterior, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue

            xs_global = pts[:, 0]
            ys_global = pts[:, 1]

            # Skip polygons whose bounding box does not intersect the selected ROI.
            if (np.nanmax(xs_global) < roi_x or np.nanmin(xs_global) > roi_x1 or
                    np.nanmax(ys_global) < roi_y or np.nanmin(ys_global) > roi_y1):
                continue

            xs = np.clip(xs_global - roi_x, 0, roi_w - 1)
            ys = np.clip(ys_global - roi_y, 0, roi_h - 1)

            rr, cc = draw_polygon(ys, xs, shape=(roi_h, roi_w))
            mask[rr, cc] = True
            n_poly += 1

            for hole in holes:
                hpts = np.asarray(hole, dtype=np.float64)
                if hpts.ndim != 2 or hpts.shape[0] < 3 or hpts.shape[1] < 2:
                    continue

                hxs_global = hpts[:, 0]
                hys_global = hpts[:, 1]

                if (np.nanmax(hxs_global) < roi_x or np.nanmin(hxs_global) > roi_x1 or
                        np.nanmax(hys_global) < roi_y or np.nanmin(hys_global) > roi_y1):
                    continue

                hxs = np.clip(hxs_global - roi_x, 0, roi_w - 1)
                hys = np.clip(hys_global - roi_y, 0, roi_h - 1)
                hrr, hcc = draw_polygon(hys, hxs, shape=(roi_h, roi_w))
                mask[hrr, hcc] = False

        except Exception:
            continue

    if n_poly == 0:
        log(f"WARNING: ROI validation GeoJSON had no usable polygons inside the selected ROI: {geojson_path}")

    return mask, int(n_poly)


def make_roi_validation_overlay(base_rgb, pred_mask, gt_mask, alpha=0.70):
    """Create a readable ROI validation overlay on top of the RGB preview.

    Green = overlap / true positive
    Red   = prediction only / false positive
    Blue  = ground truth only / false negative
    """
    base = _to_uint8_rgb(base_rgb).astype(np.float32)
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)

    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ. Prediction={pred.shape}, GT={gt.shape}")

    overlay = make_validation_overlay(pred, gt).astype(np.float32)
    colored = np.any(overlay > 0, axis=-1)

    out = base.copy()
    out[colored] = (1.0 - float(alpha)) * base[colored] + float(alpha) * overlay[colored]

    try:
        boundaries = segmentation.find_boundaries(pred.astype(np.uint8) + gt.astype(np.uint8), mode="outer")
        boundaries = morphology.binary_dilation(boundaries, morphology.disk(1))
        out[boundaries] = [255, 255, 255]
    except Exception:
        pass

    return np.clip(out, 0, 255).astype(np.uint8)


def compute_pixel_validation_metrics(pred_mask, gt_mask):
    """Compute pixel-level DICE/IoU/precision/recall."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ. Prediction={pred.shape}, GT={gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = (2.0 * tp / float(pred_area + gt_area)) if (pred_area + gt_area) > 0 else np.nan
    iou = (tp / float(union)) if union > 0 else np.nan
    precision = (tp / float(tp + fp)) if (tp + fp) > 0 else np.nan
    recall = (tp / float(tp + fn)) if (tp + fn) > 0 else np.nan

    return {
        "dice_pixel": float(dice),
        "iou_pixel": float(iou),
        "precision_pixel": float(precision),
        "recall_pixel": float(recall),
        "intersection_px": tp,
        "union_px": union,
        "prediction_area_px": pred_area,
        "ground_truth_area_px": gt_area,
        "false_positive_px": fp,
        "false_negative_px": fn,
    }


def compute_object_validation_metrics(instance_labels, gt_mask, iou_threshold=0.50):
    """Lightweight object-level metrics using connected GT objects.

    Predicted objects are the existing instance labels. Ground-truth objects are
    connected components from the rasterized GT mask. This reports approximate
    object-level precision/recall/F1 and mean matched IoU.
    """
    pred = np.asarray(instance_labels)
    gt_labels, n_gt = ndimage.label(np.asarray(gt_mask, dtype=bool))
    pred_labels = np.unique(pred)
    pred_labels = pred_labels[pred_labels > 0]

    if len(pred_labels) == 0 and n_gt == 0:
        return {
            "object_iou_threshold": float(iou_threshold),
            "object_true_positives": 0,
            "object_false_positives": 0,
            "object_false_negatives": 0,
            "object_precision": np.nan,
            "object_recall": np.nan,
            "object_f1": np.nan,
            "object_mean_matched_iou": np.nan,
            "n_pred_objects": 0,
            "n_gt_objects": 0,
        }

    # Sparse pairwise overlaps only where pred and GT overlap.
    overlap_mask = (pred > 0) & (gt_labels > 0)
    pairs = np.stack([pred[overlap_mask].ravel(), gt_labels[overlap_mask].ravel()], axis=1) if np.any(overlap_mask) else np.empty((0, 2), dtype=int)
    pair_counts = {}
    for pl, gl in pairs:
        key = (int(pl), int(gl))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    pred_areas = ndimage.sum(np.ones_like(pred, dtype=np.uint8), labels=pred, index=pred_labels).astype(float) if len(pred_labels) else np.array([])
    pred_area_map = {int(l): float(a) for l, a in zip(pred_labels, pred_areas)}
    gt_index = np.arange(1, int(n_gt) + 1)
    gt_areas = ndimage.sum(np.ones_like(gt_labels, dtype=np.uint8), labels=gt_labels, index=gt_index).astype(float) if n_gt else np.array([])
    gt_area_map = {int(l): float(a) for l, a in zip(gt_index, gt_areas)}

    candidates = []
    for (pl, gl), inter in pair_counts.items():
        union = pred_area_map.get(pl, 0.0) + gt_area_map.get(gl, 0.0) - float(inter)
        if union <= 0:
            continue
        iou = float(inter) / union
        if iou >= float(iou_threshold):
            candidates.append((iou, pl, gl))

    candidates.sort(reverse=True)
    matched_pred = set()
    matched_gt = set()
    matched_ious = []
    for iou, pl, gl in candidates:
        if pl in matched_pred or gl in matched_gt:
            continue
        matched_pred.add(pl)
        matched_gt.add(gl)
        matched_ious.append(float(iou))

    tp = len(matched_ious)
    fp = int(len(pred_labels) - tp)
    fn = int(n_gt - tp)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / float(tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / float(precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan

    return {
        "object_iou_threshold": float(iou_threshold),
        "object_true_positives": int(tp),
        "object_false_positives": int(fp),
        "object_false_negatives": int(fn),
        "object_precision": float(precision) if np.isfinite(precision) else np.nan,
        "object_recall": float(recall) if np.isfinite(recall) else np.nan,
        "object_f1": float(f1) if np.isfinite(f1) else np.nan,
        "object_mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else np.nan,
        "n_pred_objects": int(len(pred_labels)),
        "n_gt_objects": int(n_gt),
    }


def make_validation_overlay(pred_mask, gt_mask):
    """Create RGB overlay: green=overlap, red=prediction only, blue=GT only."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    overlay[tp] = [0, 220, 0]
    overlay[fp] = [230, 40, 40]
    overlay[fn] = [40, 90, 230]
    return overlay


def run_validation_if_requested(image_path, instance_labels, output_folder, params: PipelineParams, log=print):
    """Run optional DICE/validation and save reports.

    Saves:
      validation/ground_truth_mask.tif
      validation/prediction_binary_mask.tif
      validation/dice_report.json
      validation/dice_report.csv
      validation/validation_overlay.png
    """
    if not bool(getattr(params, "validation_enabled", False)):
        return None

    gt_path = find_validation_geojson_for_image(image_path, params)
    if gt_path is None:
        msg = "Validation enabled, but no matching ground-truth GeoJSON was found."
        log("WARNING: " + msg)
        return {"status": "missing_ground_truth", "message": msg}

    log(f"Running validation / DICE using GT GeoJSON: {gt_path}")
    output_folder = Path(output_folder)
    val_dir = output_folder / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    pred_mask = np.asarray(instance_labels) > 0
    gt_mask, n_gt_polygons = rasterize_geojson_to_mask(gt_path, pred_mask.shape, log=log)

    pixel_metrics = compute_pixel_validation_metrics(pred_mask, gt_mask)
    object_metrics = compute_object_validation_metrics(
        instance_labels,
        gt_mask,
        iou_threshold=float(getattr(params, "validation_iou_threshold", 0.50) or 0.50),
    )

    report = {
        "status": "success",
        "image": str(image_path),
        "ground_truth_geojson": str(gt_path),
        "n_gt_polygons_rasterized": int(n_gt_polygons),
        **pixel_metrics,
        **object_metrics,
    }

    tifffile.imwrite(str(val_dir / "ground_truth_mask.tif"), gt_mask.astype(np.uint8) * 255)
    tifffile.imwrite(str(val_dir / "prediction_binary_mask.tif"), pred_mask.astype(np.uint8) * 255)
    safe_json_dump(report, val_dir / "dice_report.json", indent=2)
    pd.DataFrame([report]).to_csv(val_dir / "dice_report.csv", index=False)

    if bool(getattr(params, "save_validation_overlay", True)):
        overlay = make_validation_overlay(pred_mask, gt_mask)
        plt.imsave(str(val_dir / "validation_overlay.png"), _downsample_for_preview(overlay, max_side=1800))

    log(f"  Pixel DICE: {report['dice_pixel']:.4f} | IoU: {report['iou_pixel']:.4f} | Object F1: {report['object_f1']:.4f}")
    log(f"  Validation saved in: {val_dir}")
    return report

# ============================================================
# Existing output / resume helpers
# ============================================================

def expected_output_files(output_folder, params: PipelineParams):
    """
    Return the expected output files for deciding whether an image is complete.

    The list respects the user's output options. For example, if Save GeoJSON is
    unchecked, qupath_final.geojson is not required for completion.
    """
    output_folder = Path(output_folder)
    expected = [
        output_folder / "instances.tif",
        output_folder / "cell_features.csv",
        output_folder / "manders_features.csv",
        output_folder / "cell_features_with_manders.csv",
        output_folder / "manders_summary.json",
    ]

    if bool(getattr(params, "save_intermediate_rgb_cellcyto", True)):
        expected.extend([
            output_folder / "RGB.tif",
            output_folder / "CellCyto.tif",
        ])

    if bool(getattr(params, "save_geojson", True)):
        expected.append(output_folder / "qupath_final.geojson")

    if bool(getattr(params, "save_preview", True)):
        expected.append(output_folder / "preview.png")

    if bool(getattr(params, "validation_enabled", False)):
        expected.append(output_folder / "validation" / "dice_report.json")

    return expected


def output_folder_is_complete(output_folder, params: PipelineParams):
    """Check whether all expected outputs exist and are non-empty."""
    expected = expected_output_files(output_folder, params)
    missing = []
    for p in expected:
        try:
            if not p.exists() or p.stat().st_size == 0:
                missing.append(p.name)
        except Exception:
            missing.append(p.name)
    return len(missing) == 0, missing


def count_cells_from_existing_features(output_folder):
    """Return number of rows in an existing cell_features.csv, if available."""
    try:
        path = Path(output_folder) / "cell_features.csv"
        if path.exists():
            return int(len(pd.read_csv(path)))
    except Exception:
        pass
    return 0


def load_rgb_hwc_from_tiff(path):
    """Load RGB.tif as H,W,3 preserving original intensity dtype."""
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"RGB.tif must be 3D. Got shape {arr.shape}")

    if arr.shape[-1] == 3:
        return arr

    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)

    raise ValueError(f"Could not interpret RGB.tif layout: {arr.shape}")


def try_resume_missing_outputs(output_folder, params: PipelineParams, log=print):
    """
    Try to regenerate missing outputs from existing intermediate files.

    This is intentionally conservative:
      - If the folder is already complete, it returns success.
      - If Manders files are missing but instances.tif, RGB.tif and
        cell_features.csv exist, it recomputes Manders.
      - If GeoJSON is missing but instances.tif and cell_features.csv exist,
        it regenerates GeoJSON only.
      - Preview is not regenerated from existing files because the saved
        seed mask is not available. If preview.png is required and missing,
        full reprocessing is still needed.
    """
    output_folder = Path(output_folder)

    complete, missing = output_folder_is_complete(output_folder, params)
    if complete:
        log("Existing output folder is complete. Skipping reprocessing.")
        return True, "complete"

    log(f"Existing output folder is incomplete. Missing: {', '.join(missing)}")

    inst_path = output_folder / "instances.tif"
    features_path = output_folder / "cell_features.csv"
    rgb_path = output_folder / "RGB.tif"

    if not inst_path.exists() or not features_path.exists():
        return False, "missing core files"

    try:
        instance_labels = tifffile.imread(str(inst_path))
        df_features = pd.read_csv(features_path)
    except Exception as e:
        return False, f"could not load existing core files: {e}"

    # Recompute Manders if missing and RGB is available.
    manders_needed = (
        not (output_folder / "manders_features.csv").exists() or
        not (output_folder / "cell_features_with_manders.csv").exists() or
        not (output_folder / "manders_summary.json").exists()
    )

    if manders_needed:
        if not rgb_path.exists():
            return False, "Manders missing and RGB.tif is not available"
        try:
            log("Resuming: recomputing missing Manders outputs from existing instances.tif + RGB.tif...")
            rgb_stack = load_rgb_hwc_from_tiff(rgb_path)
            compute_and_save_manders(
                instance_labels=instance_labels,
                rgb_stack_hwc=rgb_stack,
                df_features=df_features,
                output_folder=output_folder,
                params=params,
                log=log,
            )
        except Exception as e:
            return False, f"could not resume Manders: {e}"

    # Regenerate GeoJSON if requested and missing.
    if bool(getattr(params, "save_geojson", True)) and not (output_folder / "qupath_final.geojson").exists():
        try:
            log("Resuming: regenerating missing GeoJSON from existing instances.tif + cell_features.csv...")
            mode = str(getattr(params, "geojson_mode", "Fast bounding-box"))
            if mode.startswith("Original"):
                create_geojson_original_full_mask(
                    instance_labels=instance_labels,
                    df=df_features,
                    output_folder=output_folder,
                    params=params,
                    log=log,
                )
            else:
                create_geojson_fast_from_labels(
                    instance_labels=instance_labels,
                    df=df_features,
                    output_folder=output_folder,
                    params=params,
                    log=log,
                )
        except Exception as e:
            return False, f"could not resume GeoJSON: {e}"

    # Regenerate validation if requested and missing.
    if bool(getattr(params, "validation_enabled", False)) and not (output_folder / "validation" / "dice_report.json").exists():
        try:
            # We do not know the original image path here with certainty, but the output folder
            # name generally matches the image stem. For single-GT mode this is enough; for
            # match-by-name, the caller should reprocess if exact matching fails.
            pseudo_image_path = output_folder.with_suffix(".tif")
            log("Resuming: regenerating missing validation/DICE from existing instances.tif...")
            run_validation_if_requested(
                image_path=pseudo_image_path,
                instance_labels=instance_labels,
                output_folder=output_folder,
                params=params,
                log=log,
            )
        except Exception as e:
            return False, f"could not resume validation: {e}"

    complete_after, missing_after = output_folder_is_complete(output_folder, params)
    if complete_after:
        return True, "resumed"

    return False, "still missing after resume: " + ", ".join(missing_after)


# ============================================================
# Single file pipeline
# ============================================================

def process_single_file(original_path, params: PipelineParams, output_parent=None, log=print):
    base_path = Path(original_path)
    output_folder = get_output_folder_from_original(base_path, output_parent=output_parent)

    existing_action = str(getattr(params, "existing_output_action", "Reprocess from zero"))

    log("=" * 70)
    log(f"Processing:    {base_path.name}")
    log(f"Output folder: {output_folder}")
    log(f"Existing output behavior: {existing_action}")
    log("=" * 70)

    # ------------------------------------------------------------
    # Existing output handling before opening/reading the full image.
    # This prevents wasting time on images that are already complete.
    # ------------------------------------------------------------
    if output_folder.exists():
        complete, missing = output_folder_is_complete(output_folder, params)

        if existing_action.startswith("Skip"):
            if complete:
                cells = count_cells_from_existing_features(output_folder)
                log("SKIP: existing output folder is complete.")
                return {
                    "status": "skipped_complete",
                    "image": str(base_path),
                    "output_folder": str(output_folder),
                    "cells": int(cells),
                    "message": "Skipped because existing output folder is complete.",
                }
            else:
                log(f"Existing folder is incomplete, so this image will be reprocessed. Missing: {', '.join(missing)}")

        elif existing_action.startswith("Resume"):
            ok_resume, resume_message = try_resume_missing_outputs(output_folder, params, log=log)
            if ok_resume:
                cells = count_cells_from_existing_features(output_folder)
                log(f"RESUME/SKIP: {resume_message}.")
                return {
                    "status": "resumed_or_complete",
                    "image": str(base_path),
                    "output_folder": str(output_folder),
                    "cells": int(cells),
                    "message": resume_message,
                }
            else:
                log(f"Resume was not possible ({resume_message}). Reprocessing from source image.")

        elif existing_action.startswith("Reprocess"):
            log("Existing folder will be overwritten/recomputed from source image.")

    output_folder.mkdir(parents=True, exist_ok=True)

    backend = ImageBackend().load(str(base_path))

    try:
        log(f"Reader: {backend.reader} | Kind: {backend.file_kind} | Size: {backend.slide_dims[0]} x {backend.slide_dims[1]}")
        arr, metadata = backend.read_processing_stack(max_full_read_pixels=params.max_full_read_pixels)
        log(f"Loaded array shape: {arr.shape} | dtype: {arr.dtype}")

        stack_cyx = ensure_cyx_stack_for_pipeline(arr, params)
        log(f"Interpreted stack as C,Y,X: {stack_cyx.shape}")

        cellcyto_path, rgb_path, rgb_stack, cellcyto_stack = create_channel_images(
            stack_cyx,
            output_folder,
            params,
            metadata=metadata,
            log=log,
        )

        instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, n_after = run_segmentation_from_cellcyto(
            cellcyto_stack,
            output_folder,
            params,
            log=log,
        )

        df_features = extract_features(
            instance_labels,
            rgb_stack,
            output_folder,
            params,
            log=log,
        )

        compute_and_save_manders(
            instance_labels,
            rgb_stack,
            df_features,
            output_folder,
            params,
            log=log,
        )

        # ------------------------------------------------------------
        # Optional validation / DICE. This uses the final global mask and a
        # user-provided ground-truth GeoJSON. It does not change segmentation.
        # ------------------------------------------------------------
        try:
            run_validation_if_requested(
                image_path=base_path,
                instance_labels=instance_labels,
                output_folder=output_folder,
                params=params,
                log=log,
            )
        except Exception as val_error:
            log(f"WARNING: Validation/DICE failed but processing outputs were saved: {val_error}")
            log(traceback.format_exc())

        # ------------------------------------------------------------
        # GeoJSON and preview are post-processing outputs.
        # If one of them fails, the quantitative CSV/Manders results should
        # still be considered successfully saved.
        # ------------------------------------------------------------
        postprocess_warning = ""

        try:
            create_geojson_and_preview(
                instance_labels,
                rgb_norm,
                blue_nuclei,
                green_cyto,
                seeds_bool,
                df_features,
                output_folder,
                params,
                log=log,
            )
        except Exception as post_error:
            postprocess_warning = (
                "WARNING: Quantitative outputs were saved, but GeoJSON/preview "
                f"post-processing failed: {post_error}"
            )
            log(postprocess_warning)
            log(traceback.format_exc())

        log(f"DONE: {base_path.name} | cells: {len(df_features)}")
        return {
            "status": "success",
            "image": str(base_path),
            "output_folder": str(output_folder),
            "cells": int(len(df_features)),
            "message": postprocess_warning,
        }

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        log(f"ERROR: {base_path.name}: {e}")
        return {
            "status": "failed",
            "image": str(base_path),
            "output_folder": str(output_folder),
            "cells": 0,
            "message": err,
        }

    finally:
        backend.close()
        cleanup_memory()


def write_processing_log(log_path, rows, params: PipelineParams):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "status", "image", "output_folder", "cells", "message", "params_json"
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        params_json = json.dumps(asdict(params), ensure_ascii=False)
        for row in rows:
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": row.get("status", ""),
                "image": row.get("image", ""),
                "output_folder": row.get("output_folder", ""),
                "cells": row.get("cells", 0),
                "message": row.get("message", ""),
                "params_json": params_json,
            })


# ============================================================
# Worker thread
# ============================================================

class ProcessingWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal = pyqtSignal(list, str)

    def __init__(self, paths, params: PipelineParams, output_parent=None):
        super().__init__()
        self.paths = [str(p) for p in paths]
        self.params = params
        self.output_parent = output_parent

    def _log(self, msg):
        self.log_signal.emit(str(msg))

    def run(self):
        rows = []
        total = len(self.paths)
        for i, path in enumerate(self.paths, 1):
            self.progress_signal.emit(i - 1, total)
            row = process_single_file(
                path,
                params=self.params,
                output_parent=self.output_parent,
                log=self._log,
            )
            rows.append(row)
            self.progress_signal.emit(i, total)

        if self.output_parent:
            log_parent = Path(self.output_parent)
        elif self.paths:
            log_parent = Path(self.paths[0]).parent
        else:
            log_parent = Path.cwd()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_parent / f"CellWellSegmentation_processing_log_{stamp}.csv"
        try:
            write_processing_log(log_path, rows, self.params)
        except Exception as e:
            self._log(f"Could not write processing log: {e}")
            log_path = ""

        self.done_signal.emit(rows, str(log_path))



# ============================================================
# Exploration helpers and dialog
# ============================================================

def make_instance_overlay(rgb_hwc, instance_labels, alpha=0.35, boundary_thickness=3):
    """Create a lightweight RGB overlay for exploration previews.

    boundary_thickness controls how thick the white cell borders appear in the
    Parameter Exploration preview. The previous version used a 1-pixel boundary.
    A value of 3 makes the boundary approximately 3 pixels wide, which is easier
    to see while tuning parameters. This only affects the visual preview and does
    not change the segmentation mask or exported measurements.
    """
    base = _to_uint8_rgb(rgb_hwc)
    labels = np.asarray(instance_labels)
    overlay = base.copy().astype(np.float32)
    if labels.size == 0 or labels.max() == 0:
        return base

    mask = labels > 0
    color_map = np.zeros_like(base, dtype=np.uint8)

    # Deterministic pseudo-random colors from label ids.
    color_map[..., 0] = ((labels * 37) % 255).astype(np.uint8)
    color_map[..., 1] = ((labels * 91) % 255).astype(np.uint8)
    color_map[..., 2] = ((labels * 53) % 255).astype(np.uint8)

    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_map[mask].astype(np.float32)

    try:
        boundaries = segmentation.find_boundaries(labels, mode="outer")

        # Make the preview border thicker.
        # Radius 1 turns a 1-pixel boundary into an approximately 3-pixel boundary.
        if int(boundary_thickness) > 1:
            radius = max(1, int(round((int(boundary_thickness) - 1) / 2)))
            boundaries = morphology.binary_dilation(boundaries, morphology.disk(radius))

        overlay[boundaries] = [255, 255, 255]
    except Exception:
        pass

    return np.clip(overlay, 0, 255).astype(np.uint8)


class CropSelectionLabel(QLabel):
    """Thumbnail widget that allows rectangular ROI selection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap_original = None
        self._thumb_w = None
        self._thumb_h = None
        self._full_w = None
        self._full_h = None
        self._dragging = False
        self._start = QPoint()
        self._end = QPoint()
        self._selection_rect_widget = None
        self.selection_callback = None
        self.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")

    def set_image(self, rgb, full_w, full_h, callback=None):
        rgb = _to_uint8_rgb(rgb)
        self._thumb_h, self._thumb_w = rgb.shape[:2]
        self._full_w = int(full_w)
        self._full_h = int(full_h)
        self.selection_callback = callback
        self._pixmap_original = _numpy_rgb_to_qpixmap(rgb)
        self._selection_rect_widget = None
        self.update()

    def has_image(self):
        return self._pixmap_original is not None and self._full_w is not None and self._full_h is not None

    def _display_rect(self):
        if self._pixmap_original is None:
            return QRect(0, 0, 0, 0)
        label_w = self.width()
        label_h = self.height()
        img_w = self._pixmap_original.width()
        img_h = self._pixmap_original.height()
        if img_w <= 0 or img_h <= 0 or label_w <= 0 or label_h <= 0:
            return QRect(0, 0, 0, 0)
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))
        x0 = int(round((label_w - disp_w) / 2))
        y0 = int(round((label_h - disp_h) / 2))
        return QRect(x0, y0, disp_w, disp_h)

    def _clamp_point_to_display_rect(self, p: QPoint):
        r = self._display_rect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(), min(p.y(), r.bottom()))
        return QPoint(x, y)

    def _widget_rect_to_full_coords(self, rect: QRect):
        if not self.has_image():
            return None
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return None
        inter = rect.normalized().intersected(disp)
        if inter.width() <= 1 or inter.height() <= 1:
            return None

        x0_img = (inter.left() - disp.left()) / disp.width() * self._thumb_w
        y0_img = (inter.top() - disp.top()) / disp.height() * self._thumb_h
        x1_img = (inter.right() - disp.left()) / disp.width() * self._thumb_w
        y1_img = (inter.bottom() - disp.top()) / disp.height() * self._thumb_h

        x0_full = int(round(x0_img / self._thumb_w * self._full_w))
        y0_full = int(round(y0_img / self._thumb_h * self._full_h))
        x1_full = int(round(x1_img / self._thumb_w * self._full_w))
        y1_full = int(round(y1_img / self._thumb_h * self._full_h))

        x0_full = max(0, min(x0_full, self._full_w - 1))
        y0_full = max(0, min(y0_full, self._full_h - 1))
        x1_full = max(1, min(x1_full, self._full_w))
        y1_full = max(1, min(y1_full, self._full_h))

        x = min(x0_full, x1_full)
        y = min(y0_full, y1_full)
        w = abs(x1_full - x0_full)
        h = abs(y1_full - y0_full)
        return x, y, max(1, w), max(1, h)

    def set_selection_from_full_coords(self, x, y, w, h):
        if not self.has_image():
            return
        disp = self._display_rect()
        if disp.width() <= 0 or disp.height() <= 0:
            return
        px0 = disp.left() + (float(x) / self._full_w) * disp.width()
        py0 = disp.top() + (float(y) / self._full_h) * disp.height()
        px1 = disp.left() + (float(x + w) / self._full_w) * disp.width()
        py1 = disp.top() + (float(y + h) / self._full_h) * disp.height()
        self._selection_rect_widget = QRect(
            int(round(px0)), int(round(py0)),
            int(round(px1 - px0)), int(round(py1 - py0))
        ).normalized()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.has_image():
            disp = self._display_rect()
            if disp.contains(event.pos()):
                self._dragging = True
                self._start = self._clamp_point_to_display_rect(event.pos())
                self._end = self._start
                self._selection_rect_widget = QRect(self._start, self._end)
                self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self.has_image():
            self._end = self._clamp_point_to_display_rect(event.pos())
            self._selection_rect_widget = QRect(self._start, self._end).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._end = self._clamp_point_to_display_rect(event.pos())
            self._selection_rect_widget = QRect(self._start, self._end).normalized()
            coords = self._widget_rect_to_full_coords(self._selection_rect_widget)
            if coords is not None and self.selection_callback is not None:
                self.selection_callback(*coords)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if self._pixmap_original is not None:
            disp = self._display_rect()
            scaled = self._pixmap_original.scaled(disp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(disp.topLeft(), scaled)
        if self._selection_rect_widget is not None:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self._selection_rect_widget.normalized())
            painter.fillRect(self._selection_rect_widget.normalized(), QColor(255, 0, 0, 35))
        painter.end()


class ParameterExplorationDialog(QDialog):
    """Interactive ROI-based parameter exploration window."""
    def __init__(self, image_path, start_params: PipelineParams, parent=None):
        super().__init__(parent)
        self.setWindowIcon(get_app_icon())
        self.setWindowTitle("Parameter Exploration")
        self.resize(1180, 900)
        self.image_path = Path(image_path)
        self.params = PipelineParams(**asdict(start_params))
        self.selected_roi = None
        self.accepted_params = None
        self.backend = None
        self.gt_geojson_path = ""
        self.last_roi_validation_report = None

        root = QVBoxLayout(self)

        top_note = QLabel(
            "Select a rectangle on the thumbnail, then test segmentation parameters on that small region. "
            "When the result looks good, click 'Choose these parameters'."
        )
        top_note.setWordWrap(True)
        top_note.setStyleSheet("color: #2c3e50;")
        root.addWidget(top_note)

        top_split = QHBoxLayout()
        self.selector = CropSelectionLabel()
        self.selector.setMinimumSize(560, 330)
        top_split.addWidget(self.selector, 1)

        info_box = QGroupBox("Selected ROI")
        info_layout = QGridLayout(info_box)
        self.roi_x = QSpinBox(); self.roi_x.setRange(0, 2_000_000_000)
        self.roi_y = QSpinBox(); self.roi_y.setRange(0, 2_000_000_000)
        self.roi_w = QSpinBox(); self.roi_w.setRange(1, 2_000_000_000); self.roi_w.setValue(512)
        self.roi_h = QSpinBox(); self.roi_h.setRange(1, 2_000_000_000); self.roi_h.setValue(512)
        info_layout.addWidget(QLabel("X"), 0, 0); info_layout.addWidget(self.roi_x, 0, 1)
        info_layout.addWidget(QLabel("Y"), 1, 0); info_layout.addWidget(self.roi_y, 1, 1)
        info_layout.addWidget(QLabel("Width"), 2, 0); info_layout.addWidget(self.roi_w, 2, 1)
        info_layout.addWidget(QLabel("Height"), 3, 0); info_layout.addWidget(self.roi_h, 3, 1)
        self.use_roi_spin_btn = QPushButton("Update rectangle from values")
        self.use_roi_spin_btn.clicked.connect(self.update_rectangle_from_spinboxes)
        info_layout.addWidget(self.use_roi_spin_btn, 4, 0, 1, 2)
        self.roi_status = QLabel("No ROI selected yet")
        self.roi_status.setWordWrap(True)
        self.roi_status.setStyleSheet("color: #555;")
        info_layout.addWidget(self.roi_status, 5, 0, 1, 2)
        top_split.addWidget(info_box)
        root.addLayout(top_split)

        vis_group = QGroupBox("Exploration visualizer for selected rectangle")
        vis_layout = QHBoxLayout(vis_group)
        self.rgb_label = QLabel("RGB preview")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumSize(520, 260)
        self.rgb_label.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        self.mask_label = QLabel("Mask / overlay preview")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setMinimumSize(520, 260)
        self.mask_label.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        vis_layout.addWidget(self.rgb_label)
        vis_layout.addWidget(self.mask_label)
        root.addWidget(vis_group, 1)

        validation_group = QGroupBox("ROI validation / DICE against full-image GeoJSON")
        validation_layout = QHBoxLayout(validation_group)
        self.roi_validation_enable_chk = QCheckBox("Enable ROI DICE")
        self.roi_validation_enable_chk.setChecked(False)
        self.roi_validation_browse_btn = QPushButton("Select GT GeoJSON")
        self.roi_validation_browse_btn.clicked.connect(self.select_roi_validation_geojson)
        self.roi_validation_status = QLabel("No GT GeoJSON selected")
        self.roi_validation_status.setWordWrap(True)
        self.roi_validation_status.setStyleSheet("color: #555;")
        validation_layout.addWidget(self.roi_validation_enable_chk)
        validation_layout.addWidget(self.roi_validation_browse_btn)
        validation_layout.addWidget(self.roi_validation_status, 1)
        root.addWidget(validation_group)

        params_group = QGroupBox("Parameters for this rectangle")
        params_layout = QGridLayout(params_group)
        self._build_parameter_widgets(params_layout)
        root.addWidget(params_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Process selected rectangle")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 7px;")
        self.run_btn.clicked.connect(self.process_selected_rectangle)
        self.choose_btn = QPushButton("Choose these parameters")
        self.choose_btn.clicked.connect(self.choose_parameters)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.choose_btn)
        btn_row.addWidget(self.cancel_btn)
        root.addLayout(btn_row)

        self.log_label = QLabel("")
        self.log_label.setStyleSheet("color: #555;")
        root.addWidget(self.log_label)

        self._load_image_thumbnail()
        self._set_widgets_from_params(self.params)

    def _spin(self, mn, mx, val):
        s = QSpinBox(); s.setRange(int(mn), int(mx)); s.setValue(int(val)); return s

    def _dspin(self, mn, mx, val, decimals=3, step=None):
        s = QDoubleSpinBox()
        s.setRange(float(mn), float(mx))
        s.setDecimals(decimals)
        s.setValue(float(val))

        # Custom decimal step for the up/down arrows.
        # Useful for threshold parameters where 1.0 increments are too large.
        if step is not None:
            s.setSingleStep(float(step))

        return s

    def _add_row(self, grid, row, label1, widget1, label2, widget2):
        grid.addWidget(QLabel(label1), row, 0); grid.addWidget(widget1, row, 1)
        grid.addWidget(QLabel(label2), row, 2); grid.addWidget(widget2, row, 3)

    def _build_parameter_widgets(self, grid):
        self.nuclei_ch = self._spin(0, 20, 0)
        self.red_ch = self._spin(0, 20, 1)
        self.green_ch = self._spin(0, 20, 2)
        self.cyto_ch = self._spin(0, 20, 3)
        self.rgb_fallback_chk = QCheckBox("Allow RGB fallback if image has only 3 channels")
        self.nuc_sigma = self._dspin(0.1, 20.0, 2.5, 2)
        self.peak_min_dist = self._spin(1, 500, 14)
        self.peak_abs = self._dspin(0.0, 1.0, 0.04, 4, step=0.05)
        self.peak_rel = self._dspin(0.0, 1.0, 0.15, 4, step=0.05)
        self.peak_border = self._spin(0, 1000, 5)
        self.fg_factor = self._dspin(0.01, 5.0, 0.4, 3, step=0.05)
        self.small_holes = self._spin(0, 1000000, 200)
        self.closing_radius = self._spin(0, 100, 2)
        self.marker_radius = self._spin(0, 100, 2)
        self.min_seg_area = self._spin(0, 1000000, 30)
        self.min_cell_area = self._spin(0, 10000000, 100)
        self.max_cell_area = self._spin(1, 100000000, 50000)
        self.bio_red_thr = self._dspin(0.0, 1e9, 7000.0, 2)
        self.bio_green_thr = self._dspin(0.0, 1e9, 3500.0, 2)

        r = 0
        self._add_row(grid, r, "Nuclei channel", self.nuclei_ch, "Red channel", self.red_ch); r += 1
        self._add_row(grid, r, "Green channel", self.green_ch, "Cyto channel", self.cyto_ch); r += 1
        grid.addWidget(self.rgb_fallback_chk, r, 0, 1, 4); r += 1
        self._add_row(grid, r, "Nuclei gaussian sigma", self.nuc_sigma, "Peak min distance", self.peak_min_dist); r += 1
        self._add_row(grid, r, "Peak threshold abs", self.peak_abs, "Peak threshold rel", self.peak_rel); r += 1
        self._add_row(grid, r, "Peak exclude border", self.peak_border, "Foreground Otsu factor", self.fg_factor); r += 1
        self._add_row(grid, r, "Remove small holes", self.small_holes, "Closing radius", self.closing_radius); r += 1
        self._add_row(grid, r, "Marker dilation radius", self.marker_radius, "Min instance area", self.min_seg_area); r += 1
        self._add_row(grid, r, "Min cell area", self.min_cell_area, "Max cell area", self.max_cell_area); r += 1
        self._add_row(grid, r, "Biological red threshold", self.bio_red_thr, "Biological green threshold", self.bio_green_thr)

    def _set_widgets_from_params(self, p):
        self.nuclei_ch.setValue(int(p.nuclei_channel)); self.red_ch.setValue(int(p.red_channel))
        self.green_ch.setValue(int(p.green_channel)); self.cyto_ch.setValue(int(p.cyto_channel))
        self.rgb_fallback_chk.setChecked(bool(p.use_rgb_fallback_if_needed))
        self.nuc_sigma.setValue(float(p.nuclei_gaussian_sigma)); self.peak_min_dist.setValue(int(p.peak_min_distance))
        self.peak_abs.setValue(float(p.peak_threshold_abs)); self.peak_rel.setValue(float(p.peak_threshold_rel))
        self.peak_border.setValue(int(p.peak_exclude_border)); self.fg_factor.setValue(float(p.foreground_otsu_factor))
        self.small_holes.setValue(int(p.remove_small_holes_size)); self.closing_radius.setValue(int(p.foreground_closing_radius))
        self.marker_radius.setValue(int(p.marker_dilation_radius)); self.min_seg_area.setValue(int(p.min_segmented_area_for_instance))
        self.min_cell_area.setValue(int(p.min_cell_area_features)); self.max_cell_area.setValue(int(p.max_cell_area_features))
        self.bio_red_thr.setValue(float(p.biological_red_threshold)); self.bio_green_thr.setValue(float(p.biological_green_threshold))

    def collect_params(self):
        p = PipelineParams(**asdict(self.params))
        p.nuclei_channel = int(self.nuclei_ch.value())
        p.red_channel = int(self.red_ch.value())
        p.green_channel = int(self.green_ch.value())
        p.cyto_channel = int(self.cyto_ch.value())
        p.use_rgb_fallback_if_needed = bool(self.rgb_fallback_chk.isChecked())
        p.nuclei_gaussian_sigma = float(self.nuc_sigma.value())
        p.peak_min_distance = int(self.peak_min_dist.value())
        p.peak_threshold_abs = float(self.peak_abs.value())
        p.peak_threshold_rel = float(self.peak_rel.value())
        p.peak_exclude_border = int(self.peak_border.value())
        p.foreground_otsu_factor = float(self.fg_factor.value())
        p.remove_small_holes_size = int(self.small_holes.value())
        p.foreground_closing_radius = int(self.closing_radius.value())
        p.marker_dilation_radius = int(self.marker_radius.value())
        p.min_segmented_area_for_instance = int(self.min_seg_area.value())
        p.min_cell_area_features = int(self.min_cell_area.value())
        p.max_cell_area_features = int(self.max_cell_area.value())
        p.biological_red_threshold = float(self.bio_red_thr.value())
        p.biological_green_threshold = float(self.bio_green_thr.value())
        p.save_intermediate_rgb_cellcyto = False
        p.save_geojson = False
        p.save_preview = False
        return p

    def _load_image_thumbnail(self):
        self.backend = ImageBackend().load(str(self.image_path))
        w, h = self.backend.slide_dims
        thumb = self.backend.input_thumbnail(max_side=900)
        self.selector.set_image(thumb, full_w=w, full_h=h, callback=self.on_roi_selected)
        default_w = max(64, min(512, w // 4 if w > 0 else 512))
        default_h = max(64, min(512, h // 4 if h > 0 else 512))
        default_x = max(0, (w - default_w) // 2)
        default_y = max(0, (h - default_h) // 2)
        self.on_roi_selected(default_x, default_y, default_w, default_h)
        self.selector.set_selection_from_full_coords(default_x, default_y, default_w, default_h)

    def on_roi_selected(self, x, y, w, h):
        self.selected_roi = (int(x), int(y), int(w), int(h))
        self.roi_x.setValue(int(x)); self.roi_y.setValue(int(y)); self.roi_w.setValue(int(w)); self.roi_h.setValue(int(h))
        self.roi_status.setText(f"Selected: X={x}, Y={y}, W={w}, H={h}")

    def update_rectangle_from_spinboxes(self):
        x, y, w, h = self.roi_x.value(), self.roi_y.value(), self.roi_w.value(), self.roi_h.value()
        self.on_roi_selected(x, y, w, h)
        self.selector.set_selection_from_full_coords(x, y, w, h)

    def _set_label_image(self, label, rgb):
        pm = _numpy_rgb_to_qpixmap(_downsample_for_preview(rgb, max_side=700))
        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def select_roi_validation_geojson(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select full-image ground-truth GeoJSON for ROI DICE",
            "",
            "GeoJSON / JSON (*.geojson *.json);;All Files (*)",
        )
        if not path:
            return
        self.gt_geojson_path = path
        self.roi_validation_enable_chk.setChecked(True)
        self.roi_validation_status.setText(f"GT GeoJSON: {path}")

    def process_selected_rectangle(self):
        if self.selected_roi is None:
            QMessageBox.warning(self, "No ROI", "Select a rectangle first.")
            return
        try:
            params = self.collect_params()
            x, y, w, h = self.selected_roi
            self.log_label.setText(f"Processing ROI X={x}, Y={y}, W={w}, H={h} ...")
            QApplication.processEvents()

            arr, metadata = self.backend.read_processing_stack_region(x, y, w, h)
            stack_cyx = ensure_cyx_stack_for_pipeline(arr, params)
            tmp_output = Path(os.environ.get("TEMP", str(Path.cwd()))) / "cell_well_segmentation_exploration_tmp"
            tmp_output.mkdir(parents=True, exist_ok=True)
            _, _, rgb_stack, cellcyto_stack = create_channel_images(stack_cyx, tmp_output, params, metadata=metadata, log=lambda m: None)
            instance_labels, rgb_norm, blue_nuclei, green_cyto, seeds_bool, n_after = run_segmentation_from_cellcyto(
                cellcyto_stack, tmp_output, params, log=lambda m: None
            )
            df_features = extract_features(instance_labels, rgb_stack, tmp_output, params, log=lambda m: None)

            self._set_label_image(self.rgb_label, rgb_stack)

            validation_text = ""
            if bool(self.roi_validation_enable_chk.isChecked()) and str(self.gt_geojson_path).strip():
                gt_path = Path(self.gt_geojson_path)
                if not gt_path.exists():
                    raise FileNotFoundError(f"Selected GT GeoJSON does not exist: {gt_path}")

                gt_mask, n_gt_polygons = rasterize_geojson_to_roi_mask(
                    gt_path, x, y, w, h, log=lambda m: None
                )
                pred_mask = np.asarray(instance_labels) > 0
                metrics = compute_pixel_validation_metrics(pred_mask, gt_mask)
                object_metrics = compute_object_validation_metrics(
                    instance_labels, gt_mask, iou_threshold=0.50
                )
                self.last_roi_validation_report = {
                    "roi_x": int(x),
                    "roi_y": int(y),
                    "roi_width": int(w),
                    "roi_height": int(h),
                    "ground_truth_geojson": str(gt_path),
                    "n_gt_polygons_rasterized_in_roi": int(n_gt_polygons),
                    **metrics,
                    **object_metrics,
                }
                overlay = make_roi_validation_overlay(rgb_stack, pred_mask, gt_mask, alpha=0.70)
                validation_text = (
                    f" | ROI DICE: {metrics['dice_pixel']:.4f}; "
                    f"IoU: {metrics['iou_pixel']:.4f}; "
                    f"Precision: {metrics['precision_pixel']:.4f}; "
                    f"Recall: {metrics['recall_pixel']:.4f}; "
                    f"GT polygons: {n_gt_polygons}"
                )
                self.roi_validation_status.setText(
                    f"GT GeoJSON: {gt_path} | DICE={metrics['dice_pixel']:.4f}, "
                    f"IoU={metrics['iou_pixel']:.4f}, Precision={metrics['precision_pixel']:.4f}, "
                    f"Recall={metrics['recall_pixel']:.4f}"
                )
            else:
                overlay = make_instance_overlay(rgb_stack, instance_labels, alpha=0.42, boundary_thickness=3)
                self.last_roi_validation_report = None

            self._set_label_image(self.mask_label, overlay)
            self.log_label.setText(
                f"ROI processed. Raw instances: {n_after}; valid feature cells after area filter: {len(df_features)}."
                f"{validation_text}"
            )
            cleanup_memory()
        except Exception as e:
            QMessageBox.critical(self, "Exploration error", str(e))
            self.log_label.setText(f"Error: {e}")

    def choose_parameters(self):
        self.accepted_params = self.collect_params()
        self.accept()

    def closeEvent(self, event):
        try:
            if self.backend is not None:
                self.backend.close()
        except Exception:
            pass
        super().closeEvent(event)

    def get_params(self):
        return self.accepted_params

# ============================================================
# GUI
# ============================================================

class CellWellSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(get_app_icon())
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - {APP_TITLE}")
        self.resize(1320, 840)
        self.paths = []
        self.output_parent = None
        self.worker = None

        self._build_menu_bar()
        self._build_ui()


    def _build_menu_bar(self):
        """Top menu with a Help/About placeholder that you can fill later."""
        help_menu = self.menuBar().addMenu("Help / About")
        about_action = QAction("Help / About", self)
        about_action.triggered.connect(self.show_help_about)
        help_menu.addAction(about_action)

    def show_help_about(self):
        """Show application information, citation, DOI and output summary."""
        msg = (
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "Immunofluorescence Cell Segmentation, Quantification and Validation\n\n"
            "Cell Well Segmentation is a desktop GUI tool for microscopy image "
            "cell segmentation, cell-level feature extraction, Manders colocalization "
            "analysis, QuPath-compatible GeoJSON export, and optional DICE-based "
            "validation against ground-truth GeoJSON annotations.\n\n"
            "Main workflow:\n"
            "1. Select one image or bulk microscopy images.\n"
            "2. Review the image thumbnail and choose Default or Custom parameters.\n"
            "3. Optionally use Parameter Exploration on a selected ROI.\n"
            "4. Run segmentation and quantification.\n"
            "5. Export masks, CSV features, Manders metrics, preview images, GeoJSON, "
            "and optional validation reports.\n\n"
            "Main outputs per image:\n"
            "- instances.tif: instance segmentation mask.\n"
            "- cell_features.csv: cell-level morphology and intensity features.\n"
            "- manders_features.csv: cell-level Manders colocalization metrics.\n"
            "- cell_features_with_manders.csv: merged feature table.\n"
            "- manders_summary.json: threshold and colocalization summary.\n"
            "- qupath_final.geojson: QuPath-compatible cell annotations.\n"
            "- preview.png: visual quality-control summary.\n"
            "- validation/: optional DICE/IoU reports when ground-truth GeoJSON is used.\n\n"
            "Citation:\n"
            "Rodriguez Rojas JJ. Cell Well Segmentation: Immunofluorescence Cell "
            "Segmentation, Quantification and Validation. Version 1.0.0. Zenodo. "
            "2026. doi: 10.5281/zenodo.20387083.\n\n"
            "DOI: 10.5281/zenodo.20387083\n"
            "Zenodo badge: https://zenodo.org/badge/1149252759.svg\n"
            "GitHub: https://github.com/Juaco2r/cell-well-segmentation\n\n"
            f"Author: {APP_AUTHOR}\n"
            f"Year: {APP_YEAR}"
        )
        QMessageBox.information(self, "Help / About", msg)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel(f"{APP_TITLE}")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 8px;")
        root.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([640, 680])

        # Input group
        input_group = QGroupBox("Input images")
        input_layout = QVBoxLayout(input_group)

        btn_row = QHBoxLayout()
        one_btn = QPushButton("Select one image")
        one_btn.clicked.connect(self.select_one_image)
        bulk_btn = QPushButton("Bulk select images")
        bulk_btn.clicked.connect(self.select_bulk_images)
        folder_btn = QPushButton("Select output folder")
        folder_btn.clicked.connect(self.select_output_folder)
        btn_row.addWidget(one_btn)
        btn_row.addWidget(bulk_btn)
        btn_row.addWidget(folder_btn)
        input_layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        self.file_list.currentRowChanged.connect(self.preview_selected_list_item)
        input_layout.addWidget(self.file_list)

        self.output_label = QLabel("Output: beside each image")
        self.output_label.setStyleSheet("color: #555;")
        input_layout.addWidget(self.output_label)

        left_layout.addWidget(input_group)

        # Thumbnail group
        thumb_group = QGroupBox("Loaded image thumbnail")
        thumb_layout = QVBoxLayout(thumb_group)
        self.thumbnail_label = QLabel("No image loaded")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setMinimumSize(560, 300)
        self.thumbnail_label.setStyleSheet("background: white; border: 1px solid #bdc3c7; border-radius: 6px;")
        thumb_layout.addWidget(self.thumbnail_label)
        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet("color: #555;")
        thumb_layout.addWidget(self.image_info_label)
        left_layout.addWidget(thumb_group)

        # Params group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Parameter mode:"))
        self.param_mode_combo = QComboBox()
        self.param_mode_combo.addItems(["Default", "Custom"])
        self.param_mode_combo.currentIndexChanged.connect(self.update_custom_param_visibility)
        preset_row.addWidget(self.param_mode_combo)
        self.explore_btn = QPushButton("Exploration")
        self.explore_btn.setToolTip("Open a ROI-based parameter exploration window on the selected image.")
        self.explore_btn.clicked.connect(self.open_parameter_exploration)
        preset_row.addWidget(self.explore_btn)
        preset_row.addStretch()
        params_layout.addLayout(preset_row)

        self.custom_params_widget = QWidget()
        custom_grid = QGridLayout(self.custom_params_widget)

        self.nuclei_ch = self._spin(0, 20, 0)
        self.red_ch = self._spin(0, 20, 1)
        self.green_ch = self._spin(0, 20, 2)
        self.cyto_ch = self._spin(0, 20, 3)
        self.rgb_fallback_chk = QCheckBox("Allow RGB fallback if image has only 3 channels")
        self.rgb_fallback_chk.setChecked(False)

        self.nuc_sigma = self._dspin(0.1, 20.0, 2.5, 2)
        self.peak_min_dist = self._spin(1, 500, 14)
        self.peak_abs = self._dspin(0.0, 1.0, 0.04, 4, step=0.05)
        self.peak_rel = self._dspin(0.0, 1.0, 0.15, 4, step=0.05)
        self.peak_border = self._spin(0, 1000, 5)

        self.fg_factor = self._dspin(0.01, 5.0, 0.4, 3, step=0.05)
        self.small_holes = self._spin(0, 1000000, 200)
        self.closing_radius = self._spin(0, 100, 2)
        self.marker_radius = self._spin(0, 100, 2)
        self.min_seg_area = self._spin(0, 1000000, 30)

        self.min_cell_area = self._spin(0, 10000000, 100)
        self.max_cell_area = self._spin(1, 100000000, 50000)
        self.bio_red_thr = self._dspin(0.0, 1e9, 7000.0, 2)
        self.bio_green_thr = self._dspin(0.0, 1e9, 3500.0, 2)
        self.max_pixels = self._spin(1, 2_000_000_000, 250_000_000)

        self.save_rgb_chk = QCheckBox("Save RGB.tif and CellCyto.tif")
        self.save_rgb_chk.setChecked(True)
        self.save_geojson_chk = QCheckBox("Save QuPath GeoJSON")
        self.save_geojson_chk.setChecked(True)

        # ------------------------------------------------------------
        # GeoJSON speed controls
        # ------------------------------------------------------------
        # Fast bounding-box mode is the default because it keeps the final
        # segmentation unchanged while avoiding one full-image scan per cell.
        self.geojson_mode_combo = QComboBox()
        self.geojson_mode_combo.addItems(["Fast bounding-box", "Original full-mask"])
        self.geojson_simplify = self._dspin(0.0, 50.0, 0.0, 2)
        self.geojson_simplify.setToolTip(
            "0.0 keeps detailed contours. Try 0.5 or 1.0 for smaller GeoJSON files "
            "and faster QuPath loading, with slight boundary simplification."
        )
        self.geojson_log_every = self._spin(1, 100000, 250)

        self.save_preview_chk = QCheckBox("Save preview.png")
        self.save_preview_chk.setChecked(True)

        # ------------------------------------------------------------
        # Validation / DICE controls
        # ------------------------------------------------------------
        self.validation_enable_chk = QCheckBox("Enable validation / DICE with ground-truth GeoJSON")
        self.validation_enable_chk.setChecked(False)
        self.validation_mode_combo = QComboBox()
        self.validation_mode_combo.addItems(["Single GeoJSON", "Match by image name in folder"])
        self.validation_iou_thr = self._dspin(0.0, 1.0, 0.50, 2, step=0.05)
        self.validation_gt_label = QLabel("No GT GeoJSON/folder selected")
        self.validation_gt_label.setStyleSheet("color: #555;")
        self.validation_browse_file_btn = QPushButton("Select GT GeoJSON")
        self.validation_browse_file_btn.clicked.connect(self.select_validation_geojson)
        self.validation_browse_folder_btn = QPushButton("Select GT folder")
        self.validation_browse_folder_btn.clicked.connect(self.select_validation_folder)
        self.validation_overlay_chk = QCheckBox("Save validation overlay")
        self.validation_overlay_chk.setChecked(True)

        r = 0
        self._add_grid_row(custom_grid, r, "Nuclei channel", self.nuclei_ch, "Red channel", self.red_ch); r += 1
        self._add_grid_row(custom_grid, r, "Green channel", self.green_ch, "Cyto channel", self.cyto_ch); r += 1
        custom_grid.addWidget(self.rgb_fallback_chk, r, 0, 1, 4); r += 1
        self._add_grid_row(custom_grid, r, "Nuclei gaussian sigma", self.nuc_sigma, "Peak min distance", self.peak_min_dist); r += 1
        self._add_grid_row(custom_grid, r, "Peak threshold abs", self.peak_abs, "Peak threshold rel", self.peak_rel); r += 1
        self._add_grid_row(custom_grid, r, "Peak exclude border", self.peak_border, "Foreground Otsu factor", self.fg_factor); r += 1
        self._add_grid_row(custom_grid, r, "Remove small holes", self.small_holes, "Closing radius", self.closing_radius); r += 1
        self._add_grid_row(custom_grid, r, "Marker dilation radius", self.marker_radius, "Min instance area", self.min_seg_area); r += 1
        self._add_grid_row(custom_grid, r, "Min cell area", self.min_cell_area, "Max cell area", self.max_cell_area); r += 1
        self._add_grid_row(custom_grid, r, "Biological red threshold", self.bio_red_thr, "Biological green threshold", self.bio_green_thr); r += 1
        self._add_grid_row(custom_grid, r, "Max full-read pixels", self.max_pixels, "", QLabel("")); r += 1
        custom_grid.addWidget(self.save_rgb_chk, r, 0, 1, 2)
        custom_grid.addWidget(self.save_geojson_chk, r, 2, 1, 2); r += 1

        # GeoJSON options. These affect only polygon export after the full mask
        # already exists; they do not change segmentation results.
        self._add_grid_row(custom_grid, r, "GeoJSON mode", self.geojson_mode_combo, "Simplify tolerance", self.geojson_simplify); r += 1
        self._add_grid_row(custom_grid, r, "GeoJSON log every N", self.geojson_log_every, "", QLabel("")); r += 1

        custom_grid.addWidget(self.save_preview_chk, r, 0, 1, 2); r += 1

        # Validation options. These compare the final predicted mask against a
        # ground-truth GeoJSON and save DICE/IoU reports under validation/.
        custom_grid.addWidget(self.validation_enable_chk, r, 0, 1, 4); r += 1
        self._add_grid_row(custom_grid, r, "Validation mode", self.validation_mode_combo, "Object IoU threshold", self.validation_iou_thr); r += 1
        custom_grid.addWidget(self.validation_browse_file_btn, r, 0, 1, 1)
        custom_grid.addWidget(self.validation_browse_folder_btn, r, 1, 1, 1)
        custom_grid.addWidget(self.validation_overlay_chk, r, 2, 1, 1); r += 1
        custom_grid.addWidget(self.validation_gt_label, r, 0, 1, 4); r += 1

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.custom_params_widget)
        scroll.setMaximumHeight(300)
        params_layout.addWidget(scroll)

        default_note = QLabel(
            "Default parameters match your current segmentation/Manders settings. "
            "Switch to Custom to edit channel mapping, thresholds, and fast GeoJSON export options."
        )
        default_note.setWordWrap(True)
        default_note.setStyleSheet("color: #555;")
        params_layout.addWidget(default_note)

        left_layout.addWidget(params_group)
        self.update_custom_param_visibility()

        # Run controls
        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)

        existing_row = QHBoxLayout()
        existing_row.addWidget(QLabel("If output folder already exists:"))
        self.existing_output_combo = QComboBox()
        self.existing_output_combo.addItems([
            "Ask before run",
            "Reprocess from zero",
            "Skip completed",
            "Resume missing outputs",
        ])
        self.existing_output_combo.setToolTip(
            "Skip completed checks that all expected outputs exist. "
            "Resume can regenerate missing Manders/GeoJSON from existing intermediates when possible."
        )
        existing_row.addWidget(self.existing_output_combo, 1)
        run_layout.addLayout(existing_row)

        run_btn_row = QHBoxLayout()
        self.run_btn = QPushButton("RUN PROCESSING")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.run_btn.clicked.connect(self.run_processing)
        self.clear_log_btn = QPushButton("Clear log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        run_btn_row.addWidget(self.run_btn)
        run_btn_row.addWidget(self.clear_log_btn)
        run_layout.addLayout(run_btn_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        run_layout.addWidget(self.progress)
        left_layout.addWidget(run_group)

        # Right processing log
        log_title = QLabel("Processing log")
        log_title.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #111; color: #eee; font-family: Consolas, monospace;")
        right_layout.addWidget(self.log_text, 1)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #2c3e50; padding: 6px;")
        right_layout.addWidget(self.status_label)

        self.append_log(f"{APP_NAME} v{APP_VERSION}")
        self.append_log("Select one image or bulk images, review thumbnail, choose Default or Custom parameters, then run.")

    def _spin(self, mn, mx, val):
        s = QSpinBox()
        s.setRange(int(mn), int(mx))
        s.setValue(int(val))
        return s

    def _dspin(self, mn, mx, val, decimals=2, step=None):
        s = QDoubleSpinBox()
        s.setRange(float(mn), float(mx))
        s.setDecimals(decimals)
        s.setValue(float(val))

        # Custom decimal step for the up/down arrows.
        # Useful for threshold parameters where 1.0 increments are too large.
        if step is not None:
            s.setSingleStep(float(step))

        return s

    def _add_grid_row(self, grid, row, label1, widget1, label2, widget2):
        grid.addWidget(QLabel(label1), row, 0)
        grid.addWidget(widget1, row, 1)
        if label2:
            grid.addWidget(QLabel(label2), row, 2)
            grid.addWidget(widget2, row, 3)

    def update_custom_param_visibility(self):
        is_custom = self.param_mode_combo.currentText() == "Custom"
        self.custom_params_widget.setVisible(is_custom)

    def set_widgets_from_params(self, p: PipelineParams):
        self.nuclei_ch.setValue(int(p.nuclei_channel))
        self.red_ch.setValue(int(p.red_channel))
        self.green_ch.setValue(int(p.green_channel))
        self.cyto_ch.setValue(int(p.cyto_channel))
        self.rgb_fallback_chk.setChecked(bool(p.use_rgb_fallback_if_needed))
        self.nuc_sigma.setValue(float(p.nuclei_gaussian_sigma))
        self.peak_min_dist.setValue(int(p.peak_min_distance))
        self.peak_abs.setValue(float(p.peak_threshold_abs))
        self.peak_rel.setValue(float(p.peak_threshold_rel))
        self.peak_border.setValue(int(p.peak_exclude_border))
        self.fg_factor.setValue(float(p.foreground_otsu_factor))
        self.small_holes.setValue(int(p.remove_small_holes_size))
        self.closing_radius.setValue(int(p.foreground_closing_radius))
        self.marker_radius.setValue(int(p.marker_dilation_radius))
        self.min_seg_area.setValue(int(p.min_segmented_area_for_instance))
        self.min_cell_area.setValue(int(p.min_cell_area_features))
        self.max_cell_area.setValue(int(p.max_cell_area_features))
        self.bio_red_thr.setValue(float(p.biological_red_threshold))
        self.bio_green_thr.setValue(float(p.biological_green_threshold))
        self.max_pixels.setValue(int(p.max_full_read_pixels))
        self.save_rgb_chk.setChecked(bool(p.save_intermediate_rgb_cellcyto))
        self.save_geojson_chk.setChecked(bool(p.save_geojson))
        self.save_preview_chk.setChecked(bool(p.save_preview))
        self.geojson_mode_combo.setCurrentText(str(getattr(p, "geojson_mode", "Fast bounding-box")))
        self.geojson_simplify.setValue(float(getattr(p, "geojson_simplify_tolerance", 0.0)))
        self.geojson_log_every.setValue(int(getattr(p, "geojson_log_every", 250)))
        if hasattr(self, "validation_enable_chk"):
            self.validation_enable_chk.setChecked(bool(getattr(p, "validation_enabled", False)))
            self.validation_mode_combo.setCurrentText(str(getattr(p, "validation_mode", "Single GeoJSON")))
            self.validation_iou_thr.setValue(float(getattr(p, "validation_iou_threshold", 0.50)))
            self.validation_overlay_chk.setChecked(bool(getattr(p, "save_validation_overlay", True)))
            self.gt_geojson_path = str(getattr(p, "validation_geojson_path", "") or "")
            self.gt_geojson_folder = str(getattr(p, "validation_geojson_folder", "") or "")
            self._update_validation_label()
        if hasattr(self, "existing_output_combo"):
            self.existing_output_combo.setCurrentText(str(getattr(p, "existing_output_action", "Ask before run")))

    def open_parameter_exploration(self):
        if not self.paths:
            QMessageBox.warning(self, "No image", "Select one image or bulk images first. The exploration uses the currently selected image.")
            return
        row = self.file_list.currentRow()
        if row < 0 or row >= len(self.paths):
            row = 0
        image_path = self.paths[row]
        start_params = self.collect_params()
        dlg = ParameterExplorationDialog(image_path, start_params, self)
        if dlg.exec_() == QDialog.Accepted:
            chosen = dlg.get_params()
            if chosen is not None:
                # Keep the save options and full-read limit from the main window.
                current = self.collect_params()
                chosen.save_intermediate_rgb_cellcyto = current.save_intermediate_rgb_cellcyto
                chosen.save_geojson = current.save_geojson
                chosen.save_preview = current.save_preview
                chosen.max_full_read_pixels = current.max_full_read_pixels
                chosen.geojson_mode = current.geojson_mode
                chosen.geojson_simplify_tolerance = current.geojson_simplify_tolerance
                chosen.geojson_log_every = current.geojson_log_every
                self.param_mode_combo.setCurrentText("Custom")
                self.set_widgets_from_params(chosen)
                self.update_custom_param_visibility()
                self.append_log(f"Exploration parameters selected from: {image_path.name}")
                self.status_label.setText("Exploration parameters applied. Ready to run full image/bulk processing.")

    def _update_validation_label(self):
        if not hasattr(self, "validation_gt_label"):
            return
        if self.gt_geojson_path:
            self.validation_gt_label.setText(f"GT GeoJSON: {self.gt_geojson_path}")
        elif self.gt_geojson_folder:
            self.validation_gt_label.setText(f"GT folder: {self.gt_geojson_folder}")
        else:
            self.validation_gt_label.setText("No GT GeoJSON/folder selected")

    def select_validation_geojson(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ground-truth GeoJSON",
            "",
            "GeoJSON / JSON (*.geojson *.json);;All Files (*)",
        )
        if not path:
            return
        self.gt_geojson_path = path
        self.validation_mode_combo.setCurrentText("Single GeoJSON")
        self.validation_enable_chk.setChecked(True)
        self._update_validation_label()
        self.append_log(f"Selected validation GeoJSON: {path}")

    def select_validation_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing ground-truth GeoJSON files")
        if not folder:
            return
        self.gt_geojson_folder = folder
        self.validation_mode_combo.setCurrentText("Match by image name in folder")
        self.validation_enable_chk.setChecked(True)
        self._update_validation_label()
        self.append_log(f"Selected validation folder: {folder}")

    def select_one_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", _image_file_filter())
        if not path:
            return
        self.paths = [Path(path)]
        self.refresh_file_list()
        self.load_thumbnail(Path(path))

    def select_bulk_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select images", "", _image_file_filter())
        if not paths:
            return
        self.paths = [Path(p) for p in paths]
        self.refresh_file_list()
        self.load_thumbnail(self.paths[0])

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output parent folder")
        if not folder:
            return
        self.output_parent = Path(folder)
        self.output_label.setText(f"Output parent: {self.output_parent}")

    def refresh_file_list(self):
        self.file_list.clear()
        for p in self.paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))
            self.file_list.addItem(item)
        if self.paths:
            self.file_list.setCurrentRow(0)
        self.status_label.setText(f"Selected {len(self.paths)} image(s)")

    def preview_selected_list_item(self, row):
        if row < 0 or row >= len(self.paths):
            return
        self.load_thumbnail(self.paths[row])

    def load_thumbnail(self, path):
        try:
            backend = ImageBackend().load(str(path))
            thumb = backend.input_thumbnail(max_side=900)
            pm = _numpy_rgb_to_qpixmap(thumb)
            self.thumbnail_label.setPixmap(pm.scaled(self.thumbnail_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            w, h = backend.slide_dims
            self.image_info_label.setText(
                f"{path.name} | {w} x {h} px | reader: {backend.reader} | kind: {backend.file_kind}"
            )
            backend.close()
        except Exception as e:
            self.thumbnail_label.setText("Thumbnail failed")
            self.image_info_label.setText(str(e))

    def collect_params(self):
        if self.param_mode_combo.currentText() == "Default":
            p = default_params()
            if hasattr(self, "existing_output_combo"):
                p.existing_output_action = str(self.existing_output_combo.currentText())
            if hasattr(self, "validation_enable_chk"):
                p.validation_enabled = bool(self.validation_enable_chk.isChecked())
                p.validation_mode = str(self.validation_mode_combo.currentText())
                p.validation_geojson_path = str(getattr(self, "gt_geojson_path", "") or "")
                p.validation_geojson_folder = str(getattr(self, "gt_geojson_folder", "") or "")
                p.validation_iou_threshold = float(self.validation_iou_thr.value())
                p.save_validation_overlay = bool(self.validation_overlay_chk.isChecked())
            return p

        return PipelineParams(
            nuclei_channel=int(self.nuclei_ch.value()),
            red_channel=int(self.red_ch.value()),
            green_channel=int(self.green_ch.value()),
            cyto_channel=int(self.cyto_ch.value()),
            use_rgb_fallback_if_needed=bool(self.rgb_fallback_chk.isChecked()),
            nuclei_gaussian_sigma=float(self.nuc_sigma.value()),
            peak_min_distance=int(self.peak_min_dist.value()),
            peak_threshold_abs=float(self.peak_abs.value()),
            peak_threshold_rel=float(self.peak_rel.value()),
            peak_exclude_border=int(self.peak_border.value()),
            foreground_otsu_factor=float(self.fg_factor.value()),
            remove_small_holes_size=int(self.small_holes.value()),
            foreground_closing_radius=int(self.closing_radius.value()),
            marker_dilation_radius=int(self.marker_radius.value()),
            min_segmented_area_for_instance=int(self.min_seg_area.value()),
            min_cell_area_features=int(self.min_cell_area.value()),
            max_cell_area_features=int(self.max_cell_area.value()),
            biological_red_threshold=float(self.bio_red_thr.value()),
            biological_green_threshold=float(self.bio_green_thr.value()),
            max_full_read_pixels=int(self.max_pixels.value()),
            save_intermediate_rgb_cellcyto=bool(self.save_rgb_chk.isChecked()),
            save_geojson=bool(self.save_geojson_chk.isChecked()),
            save_preview=bool(self.save_preview_chk.isChecked()),
            geojson_mode=str(self.geojson_mode_combo.currentText()),
            geojson_simplify_tolerance=float(self.geojson_simplify.value()),
            geojson_log_every=int(self.geojson_log_every.value()),
            existing_output_action=str(self.existing_output_combo.currentText()) if hasattr(self, "existing_output_combo") else "Ask before run",
            validation_enabled=bool(self.validation_enable_chk.isChecked()) if hasattr(self, "validation_enable_chk") else False,
            validation_mode=str(self.validation_mode_combo.currentText()) if hasattr(self, "validation_mode_combo") else "Single GeoJSON",
            validation_geojson_path=str(getattr(self, "gt_geojson_path", "") or ""),
            validation_geojson_folder=str(getattr(self, "gt_geojson_folder", "") or ""),
            validation_iou_threshold=float(self.validation_iou_thr.value()) if hasattr(self, "validation_iou_thr") else 0.50,
            save_validation_overlay=bool(self.validation_overlay_chk.isChecked()) if hasattr(self, "validation_overlay_chk") else True,
        )

    def append_log(self, msg):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{stamp}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()

    def run_processing(self):
        if not self.paths:
            QMessageBox.warning(self, "No images", "Select one image or bulk images first.")
            return

        params = self.collect_params()

        # Ask once before starting the worker if output folders already exist.
        if str(getattr(params, "existing_output_action", "")).startswith("Ask"):
            existing = [
                p for p in self.paths
                if get_output_folder_from_original(p, output_parent=self.output_parent).exists()
            ]
            if existing:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Existing output folders found")
                msg.setText(
                    f"{len(existing)} selected image(s) already have an output folder.\n\n"
                    "What do you want to do?"
                )
                reprocess_btn = msg.addButton("Reprocess from zero", QMessageBox.AcceptRole)
                skip_btn = msg.addButton("Skip completed", QMessageBox.ActionRole)
                resume_btn = msg.addButton("Resume missing outputs", QMessageBox.ActionRole)
                cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
                msg.exec_()

                clicked = msg.clickedButton()
                if clicked == cancel_btn:
                    return
                if clicked == reprocess_btn:
                    params.existing_output_action = "Reprocess from zero"
                    self.existing_output_combo.setCurrentText("Reprocess from zero")
                elif clicked == skip_btn:
                    params.existing_output_action = "Skip completed"
                    self.existing_output_combo.setCurrentText("Skip completed")
                elif clicked == resume_btn:
                    params.existing_output_action = "Resume missing outputs"
                    self.existing_output_combo.setCurrentText("Resume missing outputs")
            else:
                params.existing_output_action = "Reprocess from zero"

        self.run_btn.setEnabled(False)
        self.progress.setRange(0, len(self.paths))
        self.progress.setValue(0)
        self.append_log("Starting processing...")
        self.append_log(f"Images: {len(self.paths)}")
        self.append_log(f"Parameter mode: {self.param_mode_combo.currentText()}")
        self.append_log(f"GeoJSON mode: {params.geojson_mode} | simplify tolerance: {params.geojson_simplify_tolerance}")
        self.append_log(f"Existing output behavior: {params.existing_output_action}")
        if bool(getattr(params, "validation_enabled", False)):
            self.append_log(f"Validation: {params.validation_mode} | IoU threshold: {params.validation_iou_threshold}")

        self.worker = ProcessingWorker(self.paths, params, output_parent=str(self.output_parent) if self.output_parent else None)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.done_signal.connect(self.processing_done)
        self.worker.start()

    def update_progress(self, value, total):
        self.progress.setRange(0, total)
        self.progress.setValue(value)
        self.status_label.setText(f"Processing {value}/{total}")

    def processing_done(self, rows, log_path):
        self.run_btn.setEnabled(True)
        ok = sum(1 for r in rows if r.get("status") == "success")
        skipped = sum(1 for r in rows if r.get("status") in ("skipped_complete", "resumed_or_complete"))
        failed = sum(1 for r in rows if r.get("status") == "failed")
        self.append_log("Processing finished.")
        self.append_log(f"Successful: {ok} | Skipped/resumed: {skipped} | Failed: {failed}")
        if log_path:
            self.append_log(f"Batch log saved: {log_path}")
        self.status_label.setText(f"Done. Successful: {ok}; Skipped/resumed: {skipped}; Failed: {failed}")

        if failed:
            QMessageBox.warning(
                self,
                "Done with warnings",
                f"Successful: {ok}\nSkipped/resumed: {skipped}\nFailed: {failed}\n\nLog saved to:\n{log_path}",
            )
        else:
            QMessageBox.information(
                self,
                "Done",
                f"Processed {ok} image(s).\nSkipped/resumed: {skipped}\n\nLog saved to:\n{log_path}",
            )


# ============================================================
# Main
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setWindowIcon(get_app_icon())
    app.setStyle("Fusion")
    window = CellWellSegmentationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
