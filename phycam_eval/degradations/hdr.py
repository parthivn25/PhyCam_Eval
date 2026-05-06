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


class HDRCompressionOperator:
    r"""
    ODRC-inspired HDR dynamic range compression (C++ / numpy).

        Q_β(I) = F⁻¹ { sign(F(I)) · |F(I)|^β }

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
            def _compress_ch(ch):
                F = np.fft.fft2(ch)
                mag = np.abs(F).clip(self.eps)
                compressed = (F / mag) * mag**self.beta
                out = np.real(np.fft.ifft2(compressed))
                out -= out.min()
                mx = out.max()
                if mx > 1e-8:
                    out /= mx
                return out.clip(0.0, 1.0)

            if self.per_channel:
                result = np.stack([_compress_ch(arr[c]) for c in range(arr.shape[0])], axis=0)
            else:
                if arr.shape[0] < 3:
                    result = np.stack([_compress_ch(arr[c]) for c in range(arr.shape[0])], axis=0)
                else:
                    Y = 0.299*arr[0] + 0.587*arr[1] + 0.114*arr[2]
                    Y_c = _compress_ch(Y)
                    ratio = np.where(Y > 1e-6, (Y_c / Y).clip(0, 2), 1.0)
                    result = arr.copy()
                    result[:3] = (arr[:3] * ratio[np.newaxis]).clip(0.0, 1.0)
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
