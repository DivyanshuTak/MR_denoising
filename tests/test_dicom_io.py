"""
tests/test_dicom_io.py
-----------------------
Unit tests for subtle.dicom_io.

These tests use synthetic numpy arrays and lightweight mock objects so that no
real DICOM data is required.  The goal is to verify the correctness of the
normalisation / denormalisation pipeline independently of pydicom internals.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from subtle.dicom_io import VolumeStats, build_volume, load_dicom_series, rescale_volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_datasets(arrays: list[np.ndarray]) -> list[MagicMock]:
    """Create minimal mock pydicom.Dataset objects from a list of 2-D arrays."""
    datasets = []
    for i, arr in enumerate(arrays):
        ds = MagicMock()
        ds.pixel_array = arr
        ds.SliceLocation = float(i)  # ascending slice order
        ds.filename = f"/fake/dir/slice_{i:04d}.dcm"
        datasets.append(ds)
    return datasets


# ---------------------------------------------------------------------------
# build_volume
# ---------------------------------------------------------------------------

class TestBuildVolume:
    """Verify the normalisation step of the pipeline."""

    def test_output_dtype_is_float32(self):
        arrays = [np.zeros((4, 4), dtype=np.uint16) for _ in range(3)]
        volume_f32, _ = build_volume(_make_mock_datasets(arrays))
        assert volume_f32.dtype == np.float32

    def test_normalised_range_is_0_to_1(self):
        # Two slices: one all-zeros, one with a hot pixel → range [0, 1].
        arrays = [
            np.zeros((4, 4), dtype=np.uint16),
            np.full((4, 4), 1000, dtype=np.uint16),
        ]
        volume_f32, _ = build_volume(_make_mock_datasets(arrays))
        assert volume_f32.min() == pytest.approx(0.0)
        assert volume_f32.max() == pytest.approx(1.0)

    def test_output_shape_is_z_y_x(self):
        arrays = [np.ones((10, 12), dtype=np.int16) * i for i in range(5)]
        volume_f32, _ = build_volume(_make_mock_datasets(arrays))
        assert volume_f32.shape == (5, 10, 12)

    def test_returns_correct_original_dtype(self):
        arrays = [np.zeros((4, 4), dtype=np.uint16)]
        _, stats = build_volume(_make_mock_datasets(arrays))
        assert stats.original_dtype == np.dtype("uint16")

    def test_returns_correct_original_min_max(self):
        arrays = [
            np.array([[10, 20]], dtype=np.uint16),
            np.array([[30, 40]], dtype=np.uint16),
        ]
        _, stats = build_volume(_make_mock_datasets(arrays))
        assert stats.original_min == pytest.approx(10.0)
        assert stats.original_max == pytest.approx(40.0)

    def test_zero_range_volume_returns_all_zeros(self):
        # A flat volume (single unique value) should not produce NaN/inf.
        arrays = [np.full((3, 3), 42, dtype=np.uint16)]
        volume_f32, _ = build_volume(_make_mock_datasets(arrays))
        np.testing.assert_array_equal(volume_f32, 0.0)

    def test_slices_stacked_in_input_order(self):
        # After normalisation, the relative ordering of slices must be preserved.
        arrays = [
            np.full((2, 2), v, dtype=np.uint16) for v in [0, 500, 1000]
        ]
        volume_f32, _ = build_volume(_make_mock_datasets(arrays))
        assert volume_f32[0].mean() < volume_f32[1].mean() < volume_f32[2].mean()


# ---------------------------------------------------------------------------
# rescale_volume
# ---------------------------------------------------------------------------

class TestRescaleVolume:
    """Verify the inverse-normalisation step of the pipeline."""

    def test_round_trip_uint16(self):
        """Normalising then denormalising must recover the original integers."""
        orig = np.array([[[0, 100, 500, 1000, 4095]]], dtype=np.uint16)
        orig_min = float(orig.min())
        orig_max = float(orig.max())

        normalised = (orig.astype(np.float64) - orig_min) / (orig_max - orig_min)
        recovered  = rescale_volume(
            normalised.astype(np.float32),
            target_dtype=np.dtype("uint16"),
            target_min=orig_min,
            target_max=orig_max,
        )
        np.testing.assert_array_equal(recovered, orig)

    def test_output_dtype_matches_target(self):
        volume_f32 = np.linspace(0, 1, 6, dtype=np.float32).reshape(1, 2, 3)
        result = rescale_volume(volume_f32, np.dtype("int16"), -100.0, 100.0)
        assert result.dtype == np.dtype("int16")

    def test_values_clipped_to_target_range(self):
        # Values slightly outside [0, 1] must be clamped, not wrapped.
        out_of_range = np.array([[[-0.1, 0.5, 1.1]]], dtype=np.float32)
        result = rescale_volume(out_of_range, np.dtype("uint8"), 0.0, 255.0)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_midpoint_maps_to_midpoint(self):
        """0.5 → (max + min) / 2."""
        volume_f32 = np.array([[[0.5]]], dtype=np.float32)
        result = rescale_volume(volume_f32, np.dtype("float32"), 0.0, 200.0)
        assert float(result[0, 0, 0]) == pytest.approx(100.0, abs=1e-4)


# ---------------------------------------------------------------------------
# load_dicom_series
# ---------------------------------------------------------------------------

class TestLoadDicomSeries:
    """Smoke-tests for load_dicom_series (filesystem interaction is mocked)."""

    def test_raises_if_directory_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_dicom_series(missing)

    def test_raises_if_no_dicom_files(self, tmp_path):
        # An empty directory should raise ValueError.
        with pytest.raises(ValueError, match="No valid DICOM"):
            load_dicom_series(tmp_path)

    def test_slices_sorted_ascending_by_location(self, tmp_path):
        """Datasets should come back in ascending SliceLocation order."""

        # Write three tiny fake DICOM files (just placeholders; we mock dcmread).
        file_a = tmp_path / "a.dcm"
        file_b = tmp_path / "b.dcm"
        file_c = tmp_path / "c.dcm"
        for f in (file_a, file_b, file_c):
            f.write_bytes(b"\x00")

        # Create mock datasets with deliberately shuffled slice locations.
        locations = {"a.dcm": 30.0, "b.dcm": 10.0, "c.dcm": 20.0}

        def _mock_dcmread(path, **kwargs):
            ds = MagicMock()
            ds.SliceLocation = locations[Path(path).name]
            ds.filename = path
            return ds

        with patch("subtle.dicom_io.pydicom.dcmread", side_effect=_mock_dcmread):
            result = load_dicom_series(tmp_path)

        locs = [float(ds.SliceLocation) for ds in result]
        assert locs == sorted(locs)
