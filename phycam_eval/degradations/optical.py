"""
Optical degradation operators — thin Python wrappers over the C++ backend.

The C++ extension (phycam_cpp) is the authoritative implementation.
If it is not built, a pure-Python fallback (numpy/scipy FFT) is used automatically.

All operators accept:
    image : (C, H, W) float32 array-like (numpy or torch.Tensor)
and return a numpy ndarray of the same shape.

If you have PyTorch, wrap the result with torch.from_numpy() as needed.
"""

from __future__ import annotations
import math
import numpy as np

# --- try C++ backend --------------------------------------------------------
try:
    from . import phycam_cpp as _cpp
    _CPP = True
except ImportError:
    _cpp = None
    _CPP = False

# --- optional torch support -------------------------------------------------
try:
    import torch as _torch
    _TORCH = True
except ImportError:
    _torch = None
    _TORCH = False


def _to_chw_float32(image) -> np.ndarray:
    """Normalise input to (C, H, W) float32 C-contiguous numpy array."""
    if _TORCH and isinstance(image, _torch.Tensor):
        arr = image.detach().cpu().to(_torch.float32).numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]          # (H,W) → (1,H,W)
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


# ============================================================================
# Pure-Python fallback (numpy FFT — same math as C++)
# ============================================================================

def _freq_grid(H: int, W: int):
    fy = np.fft.fftfreq(H)[:, np.newaxis]   # (H,1)
    fx = np.fft.fftfreq(W)[np.newaxis, :]   # (1,W)
    return fy + np.zeros((H, W)), fx + np.zeros((H, W))

def _radial(fy, fx):
    return np.sqrt(fy**2 + fx**2)

# Public aliases used by tests and evaluation code
def _frequency_grid(H: int, W: int, device=None):
    """Return (fy, fx) frequency grids as torch tensors (or numpy if torch unavailable)."""
    fy_np, fx_np = _freq_grid(H, W)
    if _TORCH and device is not None:
        fy = _torch.from_numpy(fy_np.astype(np.float32)).to(device)
        fx = _torch.from_numpy(fx_np.astype(np.float32)).to(device)
        return fy, fx
    return fy_np, fx_np

def _radial_freq(fy, fx):
    """Compute radial frequency grid from fy, fx (numpy or torch)."""
    if _TORCH and isinstance(fy, _torch.Tensor):
        return _torch.sqrt(fy**2 + fx**2)
    return _radial(fy, fx)

def _apply_phase_np(image: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Apply phase mask exp(j phi) to each channel of (C,H,W) array."""
    H = np.exp(1j * phi)
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        F = np.fft.fft2(image[c])
        out[c] = np.real(np.fft.ifft2(F * H)).clip(0.0, 1.0)
    return out

def _apply_amp_np(image: np.ndarray, amp: np.ndarray) -> np.ndarray:
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        F = np.fft.fft2(image[c])
        out[c] = np.real(np.fft.ifft2(F * amp)).clip(0.0, 1.0)
    return out


# ============================================================================
# DefocusOperator
# ============================================================================

class DefocusOperator:
    r"""
    Defocus blur via quadratic pupil phase (Fourier optics / PST formalism).

        A_φ(I)[c] = Re{ F⁻¹{ F(I[c]) · exp(j α ρ²) } }

    Parameters
    ----------
    alpha : float ≥ 0   Phase strength.  0 = identity.  ~π = strong blur.
    normalize_freq : bool  Normalise ρ to [0,1] (default True, PhyCV convention).
    """

    def __init__(self, alpha: float = 1.0, normalize_freq: bool = True) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.normalize_freq = normalize_freq
        if _CPP:
            self._op = _cpp.DefocusOperator(alpha, normalize_freq)

    def __call__(self, image) -> np.ndarray:
        was_tensor = _TORCH and isinstance(image, _torch.Tensor)
        arr = _to_chw_float32(image)
        if _CPP:
            result = self._op.apply(arr)
        else:
            fy, fx = _freq_grid(*arr.shape[1:])
            rho = _radial(fy, fx)
            if self.normalize_freq:
                rho = rho / max(rho.max(), 1e-10)
            phi = self.alpha * rho**2
            result = _apply_phase_np(arr, phi)
        if was_tensor:
            return _torch.from_numpy(result).to(device=image.device)
        return result

    def transfer_function(self, H: int, W: int, device=None):
        """Return the complex OTF H(ω) = exp(j α ρ²) as a (H,W) complex64 tensor."""
        fy, fx = _freq_grid(H, W)
        rho = _radial(fy, fx)
        if self.normalize_freq:
            rho = rho / max(rho.max(), 1e-10)
        phi = (self.alpha * rho**2).astype(np.float32)
        real = np.cos(phi).astype(np.float32)
        imag = np.sin(phi).astype(np.float32)
        if _TORCH:
            t = _torch.complex(_torch.from_numpy(real), _torch.from_numpy(imag))
            if device is not None:
                t = t.to(device)
            return t
        return (real + 1j * imag).astype(np.complex64)

    def otf_magnitude(self, H: int, W: int, device=None):
        """Return |OTF(f)| = 1 everywhere (phase-only filter) as (H,W) array/tensor."""
        if _CPP and device is None:
            return self._op.otf_magnitude(H, W)
        ones_np = np.ones((H, W), dtype=np.float32)
        if _TORCH and device is not None:
            t = _torch.from_numpy(ones_np).to(device)
            return t
        return ones_np

    def phase_mask(self, H: int, W: int) -> np.ndarray:
        """Return the (H,W) phase mask φ(ρ) as a numpy array."""
        if _CPP:
            return self._op.phase_mask(H, W)
        fy, fx = _freq_grid(H, W)
        rho = _radial(fy, fx)
        if self.normalize_freq:
            rho = rho / max(rho.max(), 1e-10)
        return self.alpha * rho**2

    def __repr__(self) -> str:
        backend = "C++" if _CPP else "numpy"
        return f"DefocusOperator(alpha={self.alpha:.3f}, backend={backend})"

# ============================================================================
# AstigmatismOperator
# ============================================================================

class AstigmatismOperator:
    r"""
    Astigmatism via PAGE-derived axis-dependent quadratic phase.

        φ(ρ, θ) = α · cos²(θ - θ_axis) · ρ²

    Equivalent to Zernike Z₂⁺² oblique astigmatism.

    Parameters
    ----------
    alpha      : float  Maximum phase strength.
    theta_axis : float  Blur axis angle in radians (0 = horizontal).
    """

    def __init__(self, alpha: float = 1.0, theta_axis: float = 0.0) -> None:
        self.alpha = alpha
        self.theta_axis = theta_axis
        if _CPP:
            self._op = _cpp.AstigmatismOperator(alpha, theta_axis)

    def __call__(self, image) -> np.ndarray:
        was_tensor = _TORCH and isinstance(image, _torch.Tensor)
        arr = _to_chw_float32(image)
        if _CPP:
            result = self._op.apply(arr)
        else:
            fy, fx = _freq_grid(*arr.shape[1:])
            rho = _radial(fy, fx)
            rho = rho / max(rho.max(), 1e-10)
            theta = np.arctan2(fy, fx)
            phi = self.alpha * np.cos(theta - self.theta_axis)**2 * rho**2
            result = _apply_phase_np(arr, phi)
        if was_tensor:
            return _torch.from_numpy(result).to(device=image.device)
        return result

    def transfer_function(self, H: int, W: int, device=None):
        """Return the complex OTF as a (H,W) complex64 tensor."""
        fy, fx = _freq_grid(H, W)
        rho = _radial(fy, fx)
        rho = rho / max(rho.max(), 1e-10)
        theta = np.arctan2(fy, fx)
        phi = (self.alpha * np.cos(theta - self.theta_axis)**2 * rho**2).astype(np.float32)
        real = np.cos(phi).astype(np.float32)
        imag = np.sin(phi).astype(np.float32)
        if _TORCH:
            t = _torch.complex(_torch.from_numpy(real), _torch.from_numpy(imag))
            if device is not None:
                t = t.to(device)
            return t
        return (real + 1j * imag).astype(np.complex64)

    def __repr__(self) -> str:
        backend = "C++" if _CPP else "numpy"
        return (f"AstigmatismOperator(alpha={self.alpha:.3f}, "
                f"theta_axis={math.degrees(self.theta_axis):.1f}°, backend={backend})")


# ============================================================================
# LowLightOperator
# ============================================================================

class LowLightOperator:
    r"""
    Low-light degradation: VEViD-inspired photon-starvation model.

    Butterworth low-pass on amplitude + shot noise:
        H_ll(ρ) = 1 / (1 + (ρ/ρ_c)^{2n}),  ρ_c = sqrt(light_level)
        σ_shot  = 0.05 / sqrt(light_level)

    Parameters
    ----------
    light_level : float in (0,1]  — 1.0=bright, 0.05=very dark
    order       : int             — Butterworth order (default 2)
    seed        : int             — RNG seed for shot noise
    """

    def __init__(self, light_level: float = 0.5, order: int = 2,
                 seed: int = 42) -> None:
        if not (0.0 < light_level <= 1.0):
            raise ValueError("light_level must be in (0, 1]")
        self.light_level = light_level
        self.order = order
        self.seed = seed
        if _CPP:
            self._op = _cpp.LowLightOperator(light_level, order, seed)

    def __call__(self, image) -> np.ndarray:
        was_tensor = _TORCH and isinstance(image, _torch.Tensor)
        arr = _to_chw_float32(image)
        if _CPP:
            result = self._op.apply(arr)
        else:
            fy, fx = _freq_grid(*arr.shape[1:])
            rho = _radial(fy, fx)
            rho = rho / max(rho.max(), 1e-10)
            rho_c = math.sqrt(self.light_level)
            amp = 1.0 / (1.0 + (rho / rho_c) ** (2 * self.order))
            degraded = _apply_amp_np(arr, amp)
            noise_std = 0.05 / math.sqrt(self.light_level)
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0.0, noise_std, degraded.shape).astype(np.float32)
            result = (degraded + noise).clip(0.0, 1.0)
        if was_tensor:
            return _torch.from_numpy(result).to(device=image.device)
        return result

    def __repr__(self) -> str:
        backend = "C++" if _CPP else "numpy"
        return (f"LowLightOperator(light_level={self.light_level:.2f}, "
                f"order={self.order}, backend={backend})")
