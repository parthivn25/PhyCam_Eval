"""
Unit tests for phycam_eval.degradations.noise
"""

import math

import pytest
import torch

from phycam_eval.degradations.noise import SensorNoiseOperator


@pytest.fixture
def flat_image():
    """Uniform-grey image — convenient for variance analysis."""
    return torch.full((3, 256, 256), 0.5)


@pytest.fixture
def black_image():
    return torch.zeros(3, 64, 64)


class TestSensorNoiseOperator:

    def test_output_shape(self, flat_image):
        op = SensorNoiseOperator(gain=4.0)
        assert op(flat_image).shape == flat_image.shape

    def test_output_dtype(self, flat_image):
        op = SensorNoiseOperator()
        assert op(flat_image).dtype == torch.float32

    def test_output_range_with_clip(self, flat_image):
        op = SensorNoiseOperator(gain=32.0, clip=True)
        out = op(flat_image)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_invalid_gain(self):
        with pytest.raises(ValueError):
            SensorNoiseOperator(gain=0.0)
        with pytest.raises(ValueError):
            SensorNoiseOperator(gain=-1.0)

    def test_invalid_read_noise(self):
        with pytest.raises(ValueError):
            SensorNoiseOperator(read_noise_std=-0.01)

    def test_noise_increases_with_gain(self, flat_image):
        """Higher ISO (gain) → more noise → higher variance in output."""
        variances = []
        for gain in [1.0, 4.0, 16.0, 64.0]:
            op = SensorNoiseOperator(gain=gain, seed=99)
            out = op(flat_image)
            variances.append(out.var().item())
        for i in range(len(variances) - 1):
            assert variances[i] < variances[i + 1], \
                f"Variance not monotone at gain steps: {variances}"

    def test_zero_noise_at_gain_zero_read_noise(self, flat_image):
        """
        With gain → 0 (very small) and read_noise=0, output should be very close
        to input.  With gain=1e-6 and signal=0.5, σ_shot = sqrt(1e-6 × 0.5) ≈ 7e-4.
        With 196k samples, extreme quantiles can exceed 4σ ≈ 3e-3, so atol=5e-3.
        """
        op = SensorNoiseOperator(gain=1e-6, read_noise_std=0.0, seed=0)
        out = op(flat_image)
        max_diff = (out - flat_image).abs().max().item()
        assert max_diff < 5e-3, f"Max diff with near-zero noise: {max_diff:.4f}"

    def test_shot_noise_scales_with_signal(self):
        """
        Poisson shot noise: Var ∝ signal level.
        At higher pixel values, variance should be larger.
        """
        bright = torch.full((1, 256, 256), 0.8)
        dark   = torch.full((1, 256, 256), 0.1)

        op_b = SensorNoiseOperator(gain=10.0, read_noise_std=0.0, clip=False, seed=7)
        op_d = SensorNoiseOperator(gain=10.0, read_noise_std=0.0, clip=False, seed=7)

        var_bright = (op_b(bright) - bright).var().item()
        var_dark   = (op_d(dark) - dark).var().item()
        assert var_bright > var_dark, \
            f"Shot noise variance: bright={var_bright:.4f}, dark={var_dark:.4f}"

    def test_from_iso_constructor(self, flat_image):
        # gain = (iso/base_iso) * base_gain = 32.0 * 5e-5 = 1.6e-3
        op = SensorNoiseOperator.from_iso(iso=3200, base_iso=100)
        assert op.gain == pytest.approx(32.0 * 5e-5)
        out = op(flat_image)
        assert out.shape == flat_image.shape

    def test_snr_db_decreases_with_gain(self):
        gains = [1.0, 4.0, 16.0]
        snrs = [SensorNoiseOperator(gain=g).snr_db(0.5) for g in gains]
        for i in range(len(snrs) - 1):
            assert snrs[i] > snrs[i + 1], f"SNR not decreasing: {snrs}"

    def test_repr(self):
        op = SensorNoiseOperator(gain=8.0, read_noise_std=0.003)
        assert "8.00" in repr(op)
        assert "0.0030" in repr(op)

    def test_black_image_read_noise_only(self, black_image):
        """On a zero-signal image, only read noise contributes.

        clip=True clips negative values to 0 (half-normal distribution), so the
        measured std is roughly σ × sqrt(1 - 2/π) ≈ 0.60σ.  We use clip=False to
        measure the unclipped distribution where std ≈ read_noise_std.
        """
        op = SensorNoiseOperator(gain=1e-8, read_noise_std=0.05, clip=False, seed=42)
        out = op(black_image)
        std = out.std().item()
        assert abs(std - 0.05) < 0.02, f"std={std:.4f}, expected ≈ 0.05"

    def test_shot_and_read_noise_add_in_quadrature(self, flat_image):
        """The total variance should be gain*signal + read_noise_std^2."""
        op = SensorNoiseOperator(gain=0.04, read_noise_std=0.05, clip=False, seed=7)
        out = op(flat_image[:1])
        noise = out - flat_image[:1]
        measured_var = noise.var().item()
        expected_var = 0.04 * 0.5 + 0.05**2
        assert abs(measured_var - expected_var) < 0.003, (
            f"var={measured_var:.4f}, expected ≈ {expected_var:.4f}"
        )
