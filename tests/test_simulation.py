"""
tests/test_simulation.py
-------------------------
Unit tests for subtle.simulation.

Tests are self-contained and do not require external data.  They verify
correctness of the blurring3d() function using synthetic volumes with
analytically known properties.
"""

import numpy as np
import pytest

from util.simulation import blurring3d


class TestBlurring3d:
    """Tests for the slice-wise 2-D Gaussian blur."""

    # ------------------------------------------------------------------
    # Basic contract
    # ------------------------------------------------------------------

    def test_output_shape_matches_input(self):
        volume = np.random.rand(10, 64, 64).astype(np.float32)
        result = blurring3d(volume, sigma=2.0)
        assert result.shape == volume.shape

    def test_output_dtype_is_float32(self):
        volume = np.ones((5, 8, 8), dtype=np.float32)
        result = blurring3d(volume, sigma=1.0)
        assert result.dtype == np.float32

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_uniform_volume_is_unchanged(self):
        """Blurring a constant field must return the same constant value."""
        constant = 0.7
        volume = np.full((4, 16, 16), constant, dtype=np.float32)
        result = blurring3d(volume, sigma=3.0)
        np.testing.assert_allclose(result, volume, atol=1e-6)

    def test_larger_sigma_spreads_impulse_more(self):
        """A centred impulse should have a lower peak after larger-sigma blur."""
        volume = np.zeros((3, 32, 32), dtype=np.float32)
        volume[:, 16, 16] = 1.0

        peak_small = blurring3d(volume, sigma=1.0).max()
        peak_large = blurring3d(volume, sigma=5.0).max()

        assert peak_large < peak_small

    def test_blurring_preserves_total_energy_per_slice(self):
        """
        Gaussian convolution conserves signal energy (sum of pixel values).
        We allow a 0.1% tolerance to account for boundary effects.
        """
        rng = np.random.default_rng(42)
        volume = rng.random((5, 64, 64)).astype(np.float32)
        result = blurring3d(volume, sigma=2.0)

        for z in range(volume.shape[0]):
            np.testing.assert_allclose(
                result[z].sum(), volume[z].sum(), rtol=1e-3,
                err_msg=f"Energy not conserved at slice {z}.",
            )

    def test_slice_independence(self):
        """Blur must not mix information across the slice dimension."""
        volume = np.zeros((3, 16, 16), dtype=np.float32)
        volume[0, 8, 8] = 1.0  # impulse in slice 0 only

        result = blurring3d(volume, sigma=2.0)

        # Slices 1 and 2 had no signal and must remain zero.
        np.testing.assert_array_equal(result[1], 0.0)
        np.testing.assert_array_equal(result[2], 0.0)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_raises_on_wrong_ndim(self):
        with pytest.raises(ValueError, match="3-D"):
            blurring3d(np.ones((4, 4), dtype=np.float32), sigma=1.0)

    def test_raises_on_4d_input(self):
        with pytest.raises(ValueError, match="3-D"):
            blurring3d(np.ones((2, 4, 4, 4), dtype=np.float32), sigma=1.0)

    def test_raises_on_zero_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            blurring3d(np.ones((3, 4, 4), dtype=np.float32), sigma=0.0)

    def test_raises_on_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            blurring3d(np.ones((3, 4, 4), dtype=np.float32), sigma=-2.0)
