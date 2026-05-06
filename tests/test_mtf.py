"""
Unit tests for phycam_eval.eval.mtf

Tests verify the MTF pipeline on synthetic slanted-edge images where the
true MTF is analytically known.
"""

import numpy as np
import pytest
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")

from phycam_eval.eval.mtf import (
    esf_to_mtf,
    measure_esf,
    measure_mtf,
    mtf50,
)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def make_slanted_edge_image(
    height: int = 128,
    width: int = 128,
    angle_deg: float = 5.0,
    blur_sigma: float = 0.0,
):
    """
    Create a synthetic slanted-edge image (3, H, W) as a numpy array or torch tensor.

    The edge is placed at x = W/2 + y * tan(angle_deg), giving a slight tilt.
    Optional Gaussian blur of known sigma simulates a defocused system.
    """
    import math
    from scipy.ndimage import gaussian_filter

    H, W = height, width
    slope = math.tan(math.radians(angle_deg))

    img = np.zeros((H, W), dtype=np.float32)
    for row in range(H):
        edge_col = W // 2 + row * slope
        img[row, :] = np.where(np.arange(W) >= edge_col, 1.0, 0.0)

    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma).astype(np.float32)

    # Stack to 3-channel array (H, W, 3) — works without torch
    img3 = np.stack([img, img, img], axis=-1)
    if _TORCH_AVAILABLE and torch is not None:
        return torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
    return img3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestESFToMTF:

    def test_ideal_step_edge_dc_is_one(self):
        """A perfect step edge → ESF is a Heaviside → MTF(DC) ≈ 1."""
        n = 256
        esf = np.where(np.arange(n) >= n // 2, 1.0, 0.0).astype(float)
        freqs, mtf = esf_to_mtf(esf)
        assert abs(mtf[0] - 1.0) < 0.05, f"DC MTF = {mtf[0]:.3f}"

    def test_output_shape(self):
        esf = np.linspace(0, 1, 256)
        freqs, mtf = esf_to_mtf(esf)
        assert len(freqs) == len(mtf)
        assert len(freqs) == 128  # half of 256

    def test_frequencies_in_valid_range(self):
        esf = np.linspace(0, 1, 256)
        freqs, mtf = esf_to_mtf(esf)
        assert freqs[0] >= 0.0
        assert freqs[-1] <= 0.5

    def test_mtf_nonneg(self):
        esf = np.where(np.arange(512) >= 256, 1.0, 0.0).astype(float)
        _, mtf = esf_to_mtf(esf)
        assert (mtf >= 0).all()


class TestMTF50:

    def test_known_mtf50(self):
        """Construct a synthetic MTF that crosses 0.5 at a known frequency."""
        freqs = np.linspace(0, 0.5, 100)
        # MTF = 1 - 2f → crosses 0.5 at f = 0.25
        mtf = np.clip(1.0 - 2.0 * freqs, 0.0, 1.0)
        f50 = mtf50(freqs, mtf)
        assert abs(f50 - 0.25) < 0.01, f"MTF50 = {f50:.4f}, expected 0.25"

    def test_all_above_half(self):
        """If MTF never drops below 0.5, return last frequency."""
        freqs = np.linspace(0, 0.5, 100)
        mtf = np.ones(100) * 0.8
        f50 = mtf50(freqs, mtf)
        assert f50 == freqs[-1]


class TestMeasureMTF:

    def test_output_types(self):
        img = make_slanted_edge_image()
        roi = (0, 128, 32, 96)
        freqs, mtf = measure_mtf(img, roi=roi)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mtf, np.ndarray)

    def test_clean_edge_high_mtf50(self):
        """A perfectly sharp (unblurred) slanted edge should have high MTF50."""
        img = make_slanted_edge_image(blur_sigma=0.0)
        roi = (0, 128, 32, 96)
        freqs, mtf = measure_mtf(img, roi=roi)
        f50 = mtf50(freqs, mtf)
        assert f50 > 0.2, f"Sharp edge MTF50 = {f50:.4f}, expected > 0.2"

    def test_blur_reduces_mtf50(self):
        """More blur → lower MTF50."""
        roi = (0, 128, 32, 96)
        f50_sharp = mtf50(*measure_mtf(make_slanted_edge_image(blur_sigma=0.5), roi=roi))
        f50_blurry = mtf50(*measure_mtf(make_slanted_edge_image(blur_sigma=3.0), roi=roi))
        assert f50_sharp > f50_blurry, \
            f"Sharp MTF50={f50_sharp:.4f}, blurry MTF50={f50_blurry:.4f}"

    def test_accepts_numpy_input(self):
        # make_slanted_edge_image returns numpy when torch is unavailable
        img = make_slanted_edge_image()
        if _TORCH_AVAILABLE and torch is not None and hasattr(img, "permute"):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img  # already (H, W, C) numpy
        roi = (0, 128, 32, 96)
        freqs, mtf = measure_mtf(img_np, roi=roi)
        assert len(freqs) > 0

    def test_accepts_numpy_chw_input(self):
        img = make_slanted_edge_image()
        if _TORCH_AVAILABLE and torch is not None and hasattr(img, "numpy"):
            img_chw = img.numpy()
        else:
            img_chw = img.transpose(2, 0, 1)

        roi = (0, 128, 32, 96)
        f50_chw = mtf50(*measure_mtf(img_chw, roi=roi))
        f50_hwc = mtf50(*measure_mtf(img_chw.transpose(1, 2, 0), roi=roi))
        assert abs(f50_chw - f50_hwc) < 1e-6

    def test_no_roi_raises_on_flat_image(self):
        """A flat (no-edge) image should fail auto-detection gracefully."""
        flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="slanted edge"):
            measure_mtf(flat, roi=None)
