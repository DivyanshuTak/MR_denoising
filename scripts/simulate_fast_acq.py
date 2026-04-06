"""
simulate_fast_acq.py  –  Task II
---------------------------------
Simulate a fast MRI acquisition by applying slice-wise 2-D Gaussian blurring
to a normalised HDF5 volume, then save the blurred result to both HDF5 and
DICOM.

Optionally generates a side-by-side grayscale comparison of the central axial
slice (sharp vs. blurry) and saves it to a PNG file.

Pipeline
--------
    sharp HDF5  →  blurring3d(sigma)  →  blurry HDF5
                                       →  blurry DICOMs  (via hdf5_to_dicom)
                                       →  central-slice figure (optional)

Usage
-----
    python scripts/simulate_fast_acq.py \\
        --input-hdf5    /path/to/sharp.h5 \\
        --input-dicom   /path/to/template_dicoms \\
        --output-hdf5   /path/to/blurry.h5 \\
        --output-dicom  /path/to/blurry_dicoms \\
        --sigma         5.0 \\
        --save-figure   /path/to/comparison.png
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.dicom_io import build_volume, load_dicom_series, rescale_volume, save_dicom_series
from util.simulation import blurring3d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a fast MRI acquisition via slice-wise Gaussian blurring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input-hdf5", "--ih",
        dest="input_hdf5",
        required=True,
        metavar="FILE",
        help="Input HDF5 file (normalised float32 volume from Task I).",
    )
    parser.add_argument(
        "--input-dicom", "--i",
        dest="input_dicom",
        required=True,
        metavar="DIR",
        help="Template DICOM directory (provides metadata for output DICOMs).",
    )
    parser.add_argument(
        "--output-hdf5", "--oh",
        dest="output_hdf5",
        required=True,
        metavar="FILE",
        help="Output HDF5 file for the blurred volume.",
    )
    parser.add_argument(
        "--output-dicom", "--o",
        dest="output_dicom",
        required=True,
        metavar="DIR",
        help="Output DICOM directory for the blurred images.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Standard deviation of the Gaussian kernel in pixels.",
    )
    parser.add_argument(
        "--save-figure",
        dest="save_figure",
        metavar="FILE",
        default=None,
        help="If provided, save the central-slice comparison PNG to this path.",
    )
    return parser


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_central_slice(
    sharp: np.ndarray,
    blurry: np.ndarray,
    sigma: float,
    save_path: str | None = None,
) -> None:
    """
    Show the central axial slice of the sharp and blurred volumes side by side.

    Window/level is derived from the sharp image statistics:
        centre = mean, width = 4 × std
    This reveals the anatomy without clipping aggressively on either end.

    Args:
        sharp:     Normalised float32 3-D volume (sharp).
        blurry:    Normalised float32 3-D volume (blurred).
        sigma:     Blur sigma used (for the figure title).
        save_path: If given, save the figure here instead of displaying it.
    """
    z_centre = sharp.shape[0] // 2
    sl_sharp  = sharp[z_centre]
    sl_blurry = blurry[z_centre]

    # Window/level: centre ± 2σ of the sharp slice, clamped to [0, 1].
    mean, std = float(sl_sharp.mean()), float(sl_sharp.std())
    vmin = max(0.0, mean - 2.0 * std)
    vmax = min(1.0, mean + 2.0 * std)

    imshow_kw = dict(cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(sl_sharp, **imshow_kw)
    axes[0].set_title("Original (fully sampled)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(sl_blurry, **imshow_kw)
    axes[1].set_title(f"Simulated fast acquisition  (σ = {sigma})", fontsize=13)
    axes[1].axis("off")

    fig.suptitle(
        f"Central axial slice  (z = {z_centre} / {sharp.shape[0] - 1})",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info("Comparison figure saved to: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _build_parser().parse_args()

    # 1. Load the normalised sharp volume.
    with h5py.File(args.input_hdf5, "r") as hf:
        volume_sharp: np.ndarray = hf["volume"][:]
    logging.info("Loaded sharp volume: shape=%s", volume_sharp.shape)

    # 2. Apply slice-wise 2-D Gaussian blurring.
    volume_blurry = blurring3d(volume_sharp, sigma=args.sigma)
    logging.info("Applied Gaussian blur with sigma=%.1f", args.sigma)

    # 3. Save the blurred volume to HDF5.
    out_hdf5 = Path(args.output_hdf5)
    out_hdf5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_hdf5), "w") as hf:
        ds = hf.create_dataset("volume", data=volume_blurry, compression="gzip")
        ds.attrs["sigma"]       = args.sigma
        ds.attrs["description"] = (
            "Slice-wise 2-D Gaussian-blurred volume simulating a fast acquisition."
        )
    logging.info("Blurred HDF5 written to: %s", out_hdf5)

    # 4. Write blurred volume to DICOM using the template for metadata.
    template_datasets = load_dicom_series(args.input_dicom)
    _, stats = build_volume(template_datasets)

    volume_out = rescale_volume(
        volume_blurry,
        target_dtype=stats.original_dtype,
        target_min=stats.original_min,
        target_max=stats.original_max,
    )
    save_dicom_series(volume_out, template_datasets, args.output_dicom)

    # 5. Optional: central-slice comparison figure.
    _plot_central_slice(
        volume_sharp,
        volume_blurry,
        sigma=args.sigma,
        save_path=args.save_figure,
    )


if __name__ == "__main__":
    main()
