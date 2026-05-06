"""
Integration tests for the full PhyCam-Eval degradation pipeline.

Verifies that operators chain correctly end-to-end, that shapes and
value ranges are preserved, and that identity/near-identity parameter
settings produce minimal change.
"""

import numpy as np
import pytest
import torch

from phycam_eval.degradations import (
    DefocusOperator,
    HDRCompressionOperator,
    SensorNoiseOperator,
)


def _np(x) -> np.ndarray:
    """Convert torch tensor or ndarray to numpy without deprecation warnings."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'detach'):
        return x.detach().numpy()
    return np.asarray(x)


@pytest.fixture
def clean_image():
    torch.manual_seed(0)
    return torch.rand(3, 256, 256)


@pytest.fixture
def clean_image_np():
    rng = np.random.default_rng(0)
    return rng.random((3, 256, 256)).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline: A_φ → Q_β → N_σ
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_shape_preserved(self, clean_image):
        out = SensorNoiseOperator.from_iso(800)(
              HDRCompressionOperator(beta=0.9)(
              DefocusOperator(alpha=1.0)(clean_image)))
        assert out.shape == (3, 256, 256)

    def test_range_preserved(self, clean_image):
        out = SensorNoiseOperator.from_iso(800)(
              HDRCompressionOperator(beta=0.9)(
              DefocusOperator(alpha=1.0)(clean_image)))
        arr = _np(out)
        assert arr.min() >= -1e-6
        assert arr.max() <= 1.0 + 1e-6

    def test_numpy_input_accepted(self, clean_image_np):
        out = SensorNoiseOperator.from_iso(800)(
              HDRCompressionOperator(beta=0.9)(
              DefocusOperator(alpha=1.0)(clean_image_np)))
        assert out.shape == (3, 256, 256)

    def test_grayscale_input_accepted(self):
        img = torch.rand(1, 128, 128)
        out = SensorNoiseOperator.from_iso(400)(
              HDRCompressionOperator(beta=0.9)(
              DefocusOperator(alpha=0.5)(img)))
        assert out.shape == (1, 128, 128)


# ---------------------------------------------------------------------------
# Identity / near-identity at neutral parameter values
# ---------------------------------------------------------------------------

class TestIdentityParams:
    def test_defocus_identity_at_alpha0(self, clean_image):
        out = DefocusOperator(alpha=0.0)(clean_image)
        arr = out if isinstance(out, np.ndarray) else out.numpy()
        np.testing.assert_allclose(arr, clean_image.numpy(), atol=1e-4)

    def test_hdr_identity_at_beta1(self, clean_image):
        out = HDRCompressionOperator(beta=1.0)(clean_image)
        arr = _np(out)
        np.testing.assert_allclose(arr, clean_image.numpy(), atol=1e-3)

    def test_noise_low_gain_small_perturbation(self, clean_image):
        op = SensorNoiseOperator(gain=1e-7, read_noise_std=0.0, seed=42)
        out = op(clean_image)
        arr = _np(out)
        np.testing.assert_allclose(arr, clean_image.numpy(), atol=0.01)


# ---------------------------------------------------------------------------
# Degradation is monotone in severity for HDR and noise
# ---------------------------------------------------------------------------

class TestMonotoneSeverity:
    def _psnr(self, ref, deg):
        ref = _np(ref).astype(np.float64)
        deg = _np(deg).astype(np.float64)
        mse = np.mean((ref - deg) ** 2)
        if mse < 1e-12:
            return 100.0
        return 10.0 * np.log10(1.0 / mse)

    def test_hdr_psnr_decreases_with_beta(self, clean_image):
        psnrs = [
            self._psnr(clean_image, HDRCompressionOperator(beta=b)(clean_image))
            for b in [1.0, 0.9, 0.8, 0.7, 0.5]
        ]
        # PSNR should be non-increasing as beta falls
        for i in range(len(psnrs) - 1):
            assert psnrs[i] >= psnrs[i + 1] - 0.5  # 0.5 dB tolerance

    def test_noise_psnr_decreases_with_iso(self, clean_image):
        psnrs = [
            self._psnr(clean_image, SensorNoiseOperator.from_iso(iso, seed=42)(clean_image))
            for iso in [100, 400, 1600, 6400]
        ]
        for i in range(len(psnrs) - 1):
            assert psnrs[i] >= psnrs[i + 1] - 1.0  # 1 dB tolerance for noise stochasticity


# ---------------------------------------------------------------------------
# Pipeline ordering changes output (non-commutativity)
# ---------------------------------------------------------------------------

class TestPipelineOrdering:
    def test_hdr_noise_order_matters(self, clean_image):
        hdr_then_noise = SensorNoiseOperator.from_iso(6400, seed=7)(
                         HDRCompressionOperator(beta=0.8)(clean_image))
        noise_then_hdr = HDRCompressionOperator(beta=0.8)(
                         SensorNoiseOperator.from_iso(6400, seed=7)(clean_image))
        a = _np(hdr_then_noise)
        b = _np(noise_then_hdr)
        # The two orderings should differ — if they're identical something is wrong
        assert not np.allclose(a, b, atol=1e-3)
