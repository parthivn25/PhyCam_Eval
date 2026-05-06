"""
Unit tests for phycam_eval.degradations.optical

Tests verify:
- Output shape and dtype are preserved
- Output is in [0, 1] (clamped)
- Identity behaviour: alpha=0 → output ≈ input
- Monotonicity: increasing alpha increases degradation (PSNR decreases)
- Transfer function has correct shape and is unit-magnitude (phase-only)
- OTF magnitude validates against known limits
"""

import math

import numpy as np
import pytest
import torch

from phycam_eval.degradations.optical import (
    DefocusOperator,
    _frequency_grid,
    _radial_freq,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_image():
    """A random (3, 128, 128) float32 tensor in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(3, 128, 128)


@pytest.fixture
def edge_image():
    """A synthetic image with a sharp vertical edge — useful for MTF checks."""
    img = torch.zeros(3, 128, 128)
    img[:, :, 64:] = 1.0
    return img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# DefocusOperator tests
# ---------------------------------------------------------------------------

class TestDefocusOperator:

    def test_output_shape(self, random_image):
        op = DefocusOperator(alpha=1.0)
        out = op(random_image)
        assert out.shape == random_image.shape

    def test_output_dtype(self, random_image):
        op = DefocusOperator(alpha=1.0)
        out = op(random_image)
        assert out.dtype == torch.float32

    def test_output_range(self, random_image):
        op = DefocusOperator(alpha=2.0)
        out = op(random_image)
        assert out.min() >= 0.0, f"min={out.min()}"
        assert out.max() <= 1.0, f"max={out.max()}"

    def test_identity_at_zero_alpha(self, random_image):
        """alpha=0 → phase mask is exp(0) = 1 everywhere → output = input."""
        op = DefocusOperator(alpha=0.0)
        out = op(random_image)
        assert torch.allclose(out, random_image, atol=1e-5), \
            f"Max diff at alpha=0: {(out - random_image).abs().max():.2e}"

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            DefocusOperator(alpha=-1.0)

    def test_monotonic_degradation(self, random_image):
        """Higher alpha → lower PSNR (more degradation)."""
        psnrs = [psnr(DefocusOperator(alpha=a)(random_image), random_image) for a in [0.5, 1.0, 2.0, 3.0]]
        for i in range(len(psnrs) - 1):
            assert psnrs[i] > psnrs[i + 1], \
                f"PSNR not monotonically decreasing: {psnrs}"

    def test_transfer_function_shape(self):
        op = DefocusOperator(alpha=1.0)
        H, W = 64, 64
        tf = op.transfer_function(H, W, torch.device("cpu"))
        assert tf.shape == (H, W)
        assert tf.dtype == torch.complex64

    def test_transfer_function_unit_magnitude(self):
        """Phase-only filter → |H(ω)| = 1 everywhere."""
        op = DefocusOperator(alpha=1.5)
        tf = op.transfer_function(64, 64, torch.device("cpu"))
        mag = tf.abs()
        assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5), \
            f"Max |H| deviation from 1: {(mag - 1).abs().max():.2e}"

    def test_otf_dc_is_one(self):
        """OTF(0,0) must equal 1 by definition (normalised MTF)."""
        op = DefocusOperator(alpha=2.0)
        otf = op.otf_magnitude(64, 64, torch.device("cpu"))
        dc = otf[0, 0].item()
        assert abs(dc - 1.0) < 1e-4, f"OTF DC = {dc:.6f}, expected 1.0"

    def test_repr(self):
        op = DefocusOperator(alpha=1.23)
        assert "1.230" in repr(op)






# ---------------------------------------------------------------------------
# Frequency grid helpers
# ---------------------------------------------------------------------------

class TestFrequencyHelpers:

    def test_frequency_grid_shape(self):
        H, W = 32, 64
        fy, fx = _frequency_grid(H, W, torch.device("cpu"))
        assert fy.shape == (H, W)
        assert fx.shape == (H, W)

    def test_radial_freq_nonneg(self):
        H, W = 32, 32
        fy, fx = _frequency_grid(H, W, torch.device("cpu"))
        rho = _radial_freq(fy, fx)
        assert (rho >= 0).all()

    def test_dc_is_zero(self):
        H, W = 32, 32
        fy, fx = _frequency_grid(H, W, torch.device("cpu"))
        rho = _radial_freq(fy, fx)
        assert rho[0, 0].item() == pytest.approx(0.0, abs=1e-6)
