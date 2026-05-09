"""
HDR dynamic range compression operator — C++ backend with numpy fallback.
"""
from __future__ import annotations
import numpy as np

try:
    from . import phycam_cpp as _cpp
    _CPP = True
except ImportError:
    _cpp = None
    _CPP = False

try:
    import torch as _torch
    _TORCH = True
except ImportError:
    _torch = None
    _TORCH = False


def _to_chw_float32(image) -> np.ndarray:
    if _TORCH and isinstance(image, _torch.Tensor):
        arr = image.detach().cpu().to(_torch.float32).numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB values in [0,1] to linear light."""
    x = np.asarray(x, dtype=np.float32).clip(0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Convert linear-light values in [0,1] to gamma-encoded sRGB."""
    x = np.asarray(x, dtype=np.float32).clip(0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * (x ** (1.0 / 2.4)) - 0.055,
    ).astype(np.float32)


class HDRCompressionOperator:
    r"""
    ODRC-inspired HDR dynamic range compression (C++ / numpy).

        L = sRGB⁻¹(I)
        Q_β(I) = sRGB(clip(F⁻¹ { sign(F(L)) · |F(L)|^β }, 0, 1))

    Parameters
    ----------
    beta        : float > 0  — 1.0=identity, <1=compress, >1=expand
    per_channel : bool       — apply per colour channel (default True)
    eps         : float      — avoid log(0) (default 1e-10)
    """

    def __init__(self, beta: float = 0.5, per_channel: bool = True,
                 eps: float = 1e-10) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.per_channel = per_channel
        self.eps = eps
        if _CPP:
            self._op = _cpp.HDRCompressionOperator(beta, per_channel, eps)

    def __call__(self, image) -> np.ndarray:
        was_tensor = _TORCH and isinstance(image, _torch.Tensor)
        arr = _to_chw_float32(image)
        if _CPP:
            result = self._op.apply(arr)
        else:
            linear = _srgb_to_linear(arr)

            def _compress_ch(ch):
                F = np.fft.fft2(ch)
                mag = np.abs(F).clip(self.eps)
                compressed = (F / mag) * mag**self.beta
                out = np.real(np.fft.ifft2(compressed))
                return out.clip(0.0, 1.0)

            if self.per_channel:
                linear_result = np.stack(
                    [_compress_ch(linear[c]) for c in range(linear.shape[0])],
                    axis=0,
                )
            else:
                if linear.shape[0] < 3:
                    linear_result = np.stack(
                        [_compress_ch(linear[c]) for c in range(linear.shape[0])],
                        axis=0,
                    )
                else:
                    Y = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
                    Y_c = _compress_ch(Y)
                    ratio = np.where(Y > 1e-6, (Y_c / Y).clip(0, 2), 1.0)
                    linear_result = linear.copy()
                    linear_result[:3] = (linear[:3] * ratio[np.newaxis]).clip(0.0, 1.0)

            result = _linear_to_srgb(linear_result)
        if was_tensor:
            return _torch.from_numpy(result).to(device=image.device)
        return result

    def compression_ratio_db(self) -> float:
        import math
        return 20.0 * math.log10(self.beta) if self.beta != 1.0 else 0.0

    def compression_ratio(self) -> float:
        """Alias for compression_ratio_db() — returns dB value (negative for β < 1)."""
        return self.compression_ratio_db()

    def __repr__(self) -> str:
        backend = "C++" if _CPP else "numpy"
        return f"HDRCompressionOperator(beta={self.beta:.3f}, backend={backend})"
