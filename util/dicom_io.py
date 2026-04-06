"""
dicom_io.py
-----------
Core utilities for loading a DICOM series from disk, assembling a sorted 3D
volume, and writing a modified volume back to DICOM.

Design notes
------------
* Slices are sorted by the SliceLocation tag (0020,1041), which records the
  position of the image plane along the beam axis.  This tag is reliably
  present in volumetric MR acquisitions; files missing it are skipped with a
  warning.

* Pixel values are normalised to [0, 1] float32 for downstream processing.
  The original dtype and value range are returned alongside the volume so the
  transform can be exactly inverted.

* Round-trip fidelity: rescale_volume() uses float64 arithmetic internally and
  rounds before casting, minimising quantisation error so that a
  normalise → denormalise cycle produces bit-identical pixel values.
"""

import copy
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pydicom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class VolumeStats(NamedTuple):
    """Bookkeeping information needed to invert the normalisation."""
    original_dtype: np.dtype
    original_min: float
    original_max: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dicom_series(dicom_dir: str | Path) -> list[pydicom.Dataset]:
    """
    Load every DICOM file in *dicom_dir* and return the datasets sorted by
    ascending SliceLocation.

    Files that cannot be read as DICOM or that lack the SliceLocation tag are
    skipped with a debug/warning log message rather than raising immediately,
    so a directory that contains a few stray non-DICOM files (e.g. a README)
    still works.

    Args:
        dicom_dir: Path to the directory containing DICOM files.

    Returns:
        List of pydicom.Dataset objects sorted by ascending SliceLocation.

    Raises:
        FileNotFoundError: If dicom_dir does not exist.
        ValueError:        If no usable DICOM slices are found.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    datasets: list[pydicom.Dataset] = []

    for fpath in sorted(dicom_dir.iterdir()):
        if fpath.is_dir():
            continue

        try:
            ds = pydicom.dcmread(str(fpath))
        except Exception:
            logger.debug("Skipping non-DICOM file: %s", fpath.name)
            continue

        if not hasattr(ds, "SliceLocation"):
            logger.warning("No SliceLocation tag in %s — skipping.", fpath.name)
            continue

        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No valid DICOM slices with SliceLocation found in: {dicom_dir}")

    datasets.sort(key=lambda ds: float(ds.SliceLocation))
    logger.info("Loaded %d DICOM slices from %s", len(datasets), dicom_dir)
    return datasets


# ---------------------------------------------------------------------------
# Volume assembly
# ---------------------------------------------------------------------------

def build_volume(datasets: list[pydicom.Dataset]) -> tuple[np.ndarray, VolumeStats]:
    """
    Stack pixel arrays from a sorted DICOM series into a normalised 3D volume.

    Steps
    -----
    1. Call pixel_array on each dataset to get a 2D slice (Y, X).
    2. Stack along axis 0 → raw volume of shape (Z, Y, X).
    3. Record the original dtype and value range (needed for inversion).
    4. Normalise to [0, 1] float32.

    Args:
        datasets: Sorted list of pydicom.Dataset objects (output of
                  load_dicom_series).

    Returns:
        volume_f32: float32 array of shape (Z, Y, X), values in [0, 1].
        stats:      VolumeStats named-tuple with original_dtype, original_min,
                    original_max.
    """
    slices = [ds.pixel_array for ds in datasets]
    raw = np.stack(slices, axis=0)  # shape: (Z, Y, X)

    stats = VolumeStats(
        original_dtype=raw.dtype,
        original_min=float(raw.min()),
        original_max=float(raw.max()),
    )

    drange = stats.original_max - stats.original_min
    if drange == 0.0:
        logger.warning("Volume has zero dynamic range; returning all-zeros.")
        volume_f32 = np.zeros(raw.shape, dtype=np.float32)
    else:
        # Use float64 for the arithmetic to preserve precision, then store
        # as float32 (sufficient for image data).
        volume_f32 = (
            (raw.astype(np.float64) - stats.original_min) / drange
        ).astype(np.float32)

    return volume_f32, stats


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------

def rescale_volume(
    volume_f32: np.ndarray,
    target_dtype: np.dtype,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    """
    Invert the normalisation applied by build_volume().

    Maps values from [0, 1] back to [target_min, target_max] and casts to
    *target_dtype*.  Float64 arithmetic + rounding ensures bit-exact
    round-trips for integer dtypes (assuming the original data fits cleanly
    in float32 mantissa, which holds for the 16-bit MRI data used here).

    Args:
        volume_f32:   Normalised float32 array with values in [0, 1].
        target_dtype: Destination numpy dtype (e.g. np.dtype("uint16")).
        target_min:   Minimum value of the original range.
        target_max:   Maximum value of the original range.

    Returns:
        Array of shape matching volume_f32, cast to target_dtype.
    """
    rescaled = volume_f32.astype(np.float64) * (target_max - target_min) + target_min

    # Clip before casting to prevent wrap-around artefacts at the boundaries.
    rescaled = np.clip(rescaled, target_min, target_max)

    # Round to nearest integer when the target is an integer dtype.
    if np.issubdtype(target_dtype, np.integer):
        rescaled = np.round(rescaled)

    return rescaled.astype(target_dtype)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_dicom_series(
    volume: np.ndarray,
    template_datasets: list[pydicom.Dataset],
    output_dir: str | Path,
) -> None:
    """
    Write a 3D volume as a DICOM series using template datasets for metadata.

    For each slice z, the corresponding template dataset is deep-copied, its
    PixelData is replaced with the z-th plane of *volume*, and the file is
    saved to *output_dir* under the same filename as the template.  All DICOM
    metadata (patient demographics, acquisition geometry, etc.) is preserved.

    Args:
        volume:            Integer-typed 3D array of shape (Z, Y, X).  Its
                           dtype must be consistent with the BitsAllocated /
                           PixelRepresentation tags in the template.
        template_datasets: Sorted list of template pydicom.Dataset objects
                           (output of load_dicom_series).
        output_dir:        Destination directory; will be created if absent.

    Raises:
        ValueError: If the number of slices in volume != len(template_datasets).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = volume.shape[0]
    if n_slices != len(template_datasets):
        raise ValueError(
            f"Volume has {n_slices} slices but there are "
            f"{len(template_datasets)} template datasets."
        )

    for idx, template_ds in enumerate(template_datasets):
        # Deep-copy so the caller's list remains unmodified.
        out_ds = copy.deepcopy(template_ds)

        # Replace the pixel payload; keep all other tags intact.
        out_ds.PixelData = volume[idx].tobytes()
        out_ds.Rows = volume.shape[1]
        out_ds.Columns = volume.shape[2]

        # Derive the output filename from the template's source path.
        try:
            fname = Path(template_ds.filename).name
        except AttributeError:
            fname = f"slice_{idx:04d}.dcm"

        out_ds.save_as(str(output_dir / fname))

    logger.info("Saved %d DICOM slices to %s", n_slices, output_dir)
