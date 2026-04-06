"""
simulation.py
-------------
Functions for simulating fast MRI acquisitions by degrading image quality.

Background
----------
In accelerated MRI, scan time is reduced by collecting fewer k-space lines
(phase-encode steps).  In image space this manifests as reduced effective
resolution, which can be approximated by convolving the fully-sampled image
with a Gaussian point-spread function.

Applying the blur independently per axial slice (rather than with a 3-D
kernel) reflects the clinical reality that acceleration is typically applied
in the in-plane phase-encode direction only.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def blurring3d(input3d: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a 2-D Gaussian blur independently to each axial slice of a 3-D volume.

    The filter is applied slice-by-slice so that information is not mixed
    across the slice dimension.  This models in-plane resolution reduction
    (e.g. from k-space undersampling) rather than through-plane blurring.

    Args:
        input3d: Float array of shape (Z, Y, X).  Expected range [0, 1] but
                 any finite float values are accepted.
        sigma:   Standard deviation of the Gaussian kernel in pixels.
                 Must be strictly positive.  sigma = 5 is used in the
                 challenge to simulate a substantially accelerated acquisition.

    Returns:
        Float32 array of the same shape as input3d, with each axial slice
        independently convolved with a 2-D Gaussian of the given sigma.

    Raises:
        ValueError: If input3d is not 3-D or sigma is not positive.

    Example
    -------
    >>> sharp   = load_volume(...)          # shape (Z, Y, X), float32, [0,1]
    >>> blurry  = blurring3d(sharp, sigma=5.0)
    """
    if input3d.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array (Z, Y, X), got shape {input3d.shape}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}.")

    output = np.empty_like(input3d, dtype=np.float32)

    for z in range(input3d.shape[0]):
        # gaussian_filter on a 2-D slice applies the same sigma to both axes,
        # which is what we want for isotropic in-plane blurring.
        output[z] = gaussian_filter(input3d[z].astype(np.float32), sigma=sigma)

    return output
