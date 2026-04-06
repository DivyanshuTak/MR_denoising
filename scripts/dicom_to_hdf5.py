"""
dicom_to_hdf5.py  –  Task I, Script 1
--------------------------------------
Convert a directory of DICOM slices into a single HDF5 file containing a
normalised float32 3-D volume.

The HDF5 file layout is:

    /volume                 – float32 array (Z, Y, X), values in [0, 1]
    /metadata (group)
        .original_dtype     – string, e.g. "uint16"
        .original_min       – float, minimum raw pixel value
        .original_max       – float, maximum raw pixel value

Storing the original range in the HDF5 lets the companion hdf5_to_dicom.py
script reconstruct bit-exact DICOM files without needing to re-read the
template series just for its value range.

Usage
-----
    python scripts/dicom_to_hdf5.py \\
        --input-dicom  /path/to/dicoms \\
        --output-hdf5  /path/to/output.h5
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py

# Allow running this script directly from the repo root without installing the
# package (e.g. `python scripts/dicom_to_hdf5.py ...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.dicom_io import build_volume, load_dicom_series


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False prevents '--h' being interpreted as a prefix of
    # '--help', since the challenge spec requires '--h' as a short form for
    # '--output-hdf5'.
    parser = argparse.ArgumentParser(
        description="Convert a DICOM series to a normalised HDF5 volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-dicom", "--i",
        dest="input_dicom",
        required=True,
        metavar="DIR",
        help="Path to the directory containing DICOM files.",
    )
    parser.add_argument(
        "--output-hdf5", "--h",
        dest="output_hdf5",
        required=True,
        metavar="FILE",
        help="Path to the output HDF5 file (created or overwritten).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _build_parser().parse_args()

    # 1. Load and sort DICOM slices by SliceLocation.
    datasets = load_dicom_series(args.input_dicom)

    # 2. Stack into a 3-D volume and normalise to [0, 1] float32.
    volume_f32, stats = build_volume(datasets)

    logging.info(
        "Volume shape: %s  |  dtype: %s  |  original range: [%.1f, %.1f]",
        volume_f32.shape,
        stats.original_dtype,
        stats.original_min,
        stats.original_max,
    )

    # 3. Write to HDF5.
    out_path = Path(args.output_hdf5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), "w") as hf:
        # Store the volume with gzip compression (lossless, ~2–4× smaller).
        hf.create_dataset("volume", data=volume_f32, compression="gzip")

        # Attach normalisation metadata so the inverse script can reconstruct
        # bit-exact pixel values without re-reading the template DICOMs.
        meta = hf.create_group("metadata")
        meta.attrs["original_dtype"] = str(stats.original_dtype)
        meta.attrs["original_min"]   = stats.original_min
        meta.attrs["original_max"]   = stats.original_max

    logging.info("HDF5 written to: %s", out_path)


if __name__ == "__main__":
    main()
