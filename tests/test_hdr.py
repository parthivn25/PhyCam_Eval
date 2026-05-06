"""
Unit tests for phycam_eval.degradations.hdr
"""

import pytest
import torch

from phycam_eval.degradations.hdr import HDRCompressionOperator


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.rand(3, 128, 128)


class TestHDRCompressionOperator:

    def test_output_shape(self, image):
        op = HDRCompressionOperator(beta=0.5)
        assert op(image).shape == image.shape

    def test_output_range(self, image):
        op = HDRCompressionOperator(beta=0.3)
        out = op(image)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_identity_at_beta_one(self, image):
        """
        beta=1.0: |F(I)|^1 = |F(I)|  → Q_1(I) = F⁻¹{F(I)} = I (up to normalisation).
        The output is normalised to [0,1], so we check that the structure is preserved
        rather than exact equality.
        """
        op = HDRCompressionOperator(beta=1.0)
        out = op(image)
        # Pearson correlation between flattened tensors should be near 1
        x = image.flatten()
        y = out.flatten()
        corr = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        assert corr > 0.99, f"beta=1 correlation = {corr:.4f}, expected > 0.99"

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            HDRCompressionOperator(beta=0.0)
        with pytest.raises(ValueError):
            HDRCompressionOperator(beta=-0.5)

    def test_compression_reduces_dynamic_range(self, image):
        """beta < 1 compresses dynamic range: std of output < std of input."""
        # Use a high-contrast image
        high_contrast = torch.zeros(3, 64, 64)
        high_contrast[:, :32, :] = 1.0

        op = HDRCompressionOperator(beta=0.3)
        out = op(high_contrast)
        assert out.std() < high_contrast.std(), \
            "beta=0.3 should reduce dynamic range (std)"

    def test_per_channel_vs_luminance(self, image):
        """The two modes should give different results."""
        op_per = HDRCompressionOperator(beta=0.5, per_channel=True)
        op_lum = HDRCompressionOperator(beta=0.5, per_channel=False)
        out_per = op_per(image)
        out_lum = op_lum(image)
        assert not torch.allclose(out_per, out_lum, atol=1e-3)

    def test_compression_ratio_sign(self):
        op = HDRCompressionOperator(beta=0.5)
        assert op.compression_ratio() < 0, "Compression should give negative dB"

    def test_repr(self):
        op = HDRCompressionOperator(beta=0.42)
        assert "0.420" in repr(op)
