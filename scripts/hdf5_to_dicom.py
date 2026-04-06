"""
hdf5_to_dicom.py  –  Task I, Script 2
--------------------------------------
Read a normalised HDF5 volume and write it back as a DICOM series, rescaling
pixel values to match the dynamic range and data type of a template DICOM
series.

The script is intentionally the inverse of dicom_to_hdf5.py.  Running both
scripts in sequence on the same data must produce DICOM files that are
pixel-identical to the originals (verified by the challenge requirements).

How rescaling works
-------------------
The template DICOM series is loaded and its min/max pixel values are extracted.
The HDF5 float32 volume (range [0, 1]) is then mapped back to [min, max] with
float64 arithmetic, rounded, and cast to the original pixel dtype.

Usage
-----
    python scripts/hdf5_to_dicom.py \\
        --input-dicom  /path/to/template_dicoms \\
        --input-hdf5   /path/to/input.h5 \\
        --output-dicom /path/to/output_dicoms
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.dicom_io import build_volume, load_dicom_series, rescale_volume, save_dicom_series


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write an HDF5 volume to DICOM, rescaling to match a template series."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-dicom", "--i",
        dest="input_dicom",
        required=True,
        metavar="DIR",
        help="Path to the template DICOM directory (provides metadata and target dtype).",
    )
    parser.add_argument(
        "--input-hdf5", "--h",
        dest="input_hdf5",
        required=True,
        metavar="FILE",
        help="Path to input HDF5 file (float32 volume in [0, 1]).",
    )
    parser.add_argument(
        "--output-dicom", "--o",
        dest="output_dicom",
        required=True,
        metavar="DIR",
        help="Path to output DICOM directory (will be created if absent).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _build_parser().parse_args()

    # 1. Load the template DICOM series to obtain the target dtype and value
    #    range.  We also keep the datasets to use as metadata templates when
    #    writing the output DICOMs.
    template_datasets = load_dicom_series(args.input_dicom)
    _, stats = build_volume(template_datasets)

    logging.info(
        "Template: %d slices  |  dtype: %s  |  range: [%.1f, %.1f]",
        len(template_datasets),
        stats.original_dtype,
        stats.original_min,
        stats.original_max,
    )

    # 2. Read the normalised float32 volume from HDF5.
    with h5py.File(args.input_hdf5, "r") as hf:
        volume_f32: np.ndarray = hf["volume"][:]

    logging.info("HDF5 volume shape: %s", volume_f32.shape)

    # 3. Rescale the float32 [0, 1] volume back to the original dtype and range.
    volume_out = rescale_volume(
        volume_f32,
        target_dtype=stats.original_dtype,
        target_min=stats.original_min,
        target_max=stats.original_max,
    )

    # 4. Write each slice into a new DICOM file, preserving template metadata.
    save_dicom_series(volume_out, template_datasets, args.output_dicom)

    logging.info("Done. Output DICOMs written to: %s", args.output_dicom)


if __name__ == "__main__":
    main()
