"""
Sensor noise operator — C++ backend with numpy fallback.
"""
from __future__ import annotations
import math
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


class SensorNoiseOperator:
    r"""
    Mixed Poisson-Gaussian sensor noise (C++ / numpy).

        N(I) = Poisson(I/g)·g + Normal(0, σ_r²)
    Shot noise approximated as Normal(0, g·I).

    Parameters
    ----------
    gain           : float > 0  — ISO / base_ISO
    read_noise_std : float ≥ 0  — read noise std in normalised units
    clip           : bool       — clamp output to [0,1] (default True)
    seed           : int        — RNG seed
    """

    def __init__(self, gain: float = 1.0, read_noise_std: float = 0.005,
                 clip: bool = True, seed: int = 12345) -> None:
        if gain <= 0:
            raise ValueError("gain must be positive")
        if read_noise_std < 0:
            raise ValueError("read_noise_std must be non-negative")
        self.gain = gain
        self.read_noise_std = read_noise_std
        self.clip = clip
        self.seed = seed
        if _CPP:
            self._op = _cpp.SensorNoiseOperator(gain, read_noise_std, clip, seed)

    def __call__(self, image) -> np.ndarray:
        was_tensor = _TORCH and isinstance(image, _torch.Tensor)
        arr = _to_chw_float32(image)
        if _CPP:
            result = self._op.apply(arr)
        else:
            sigma_shot = np.sqrt(self.gain * arr.clip(min=0.0)).astype(np.float32)
            rng = np.random.default_rng(self.seed)
            shot  = (rng.standard_normal(arr.shape) * sigma_shot).astype(np.float32)
            read  = rng.normal(0.0, self.read_noise_std, arr.shape).astype(np.float32)
            noisy = arr + shot + read
            result = noisy.clip(0.0, 1.0) if self.clip else noisy
        if was_tensor:
            return _torch.from_numpy(result).to(device=image.device)
        return result

    def snr_db(self, signal: float = 0.5) -> float:
        if _CPP:
            return self._op.snr_db(signal)
        variance = self.gain * signal + self.read_noise_std**2
        snr = signal / math.sqrt(max(variance, 1e-12))
        return 20.0 * math.log10(max(snr, 1e-12))

    @classmethod
    def from_iso(cls, iso: int, base_iso: int = 100,
                 base_read_noise: float = 0.002,
                 base_gain: float = 5e-5, **kwargs) -> "SensorNoiseOperator":
        """Construct from ISO value.

        base_gain calibrates the shot-noise scale so that ISO 100 at mid-gray
        (signal=0.5) gives SNR ≈ 40 dB, matching a well-exposed camera sensor.
        ISO doubles per stop: ISO 1600 → SNR ≈ 31 dB, ISO 6400 → SNR ≈ 25 dB.
        """
        return cls(gain=(iso / base_iso) * base_gain,
                   read_noise_std=base_read_noise, **kwargs)

    def __repr__(self) -> str:
        backend = "C++" if _CPP else "numpy"
        gain_str = f"{self.gain:.2e}" if self.gain < 0.01 else f"{self.gain:.2f}"
        return (f"SensorNoiseOperator(gain={gain_str}, "
                f"read_noise_std={self.read_noise_std:.4f}, backend={backend})")
