"""
Modulation Transfer Function (MTF) measurement via the slanted-edge method.

Standard reference: ISO 12233:2023 (Photography — Electronic still picture imaging
— Resolution and spatial frequency responses).

Overview
--------
The slanted-edge method works as follows:
    1. Find a sharp, high-contrast edge in the image (tilted ~5° from vertical).
    2. Extract the Edge Spread Function (ESF) by projecting pixel values onto the
       axis perpendicular to the edge.
    3. Differentiate the ESF to get the Line Spread Function (LSF).
    4. Fourier-transform the LSF to get the MTF.

In PhyCam-Eval, this is used to:
    - Validate each degradation operator against its theoretically predicted OTF.
    - Generate Figure 2: MTF vs. physical parameter (e.g. f-number sweep).
    - Provide a physically interpretable quality measure that ISP teams can read
      natively (alongside mAP).

All frequencies are expressed in cycles/pixel (normalised units, range [0, 0.5]).
"""

from __future__ import annotations

import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def _to_hwc_image(image) -> np.ndarray:
    """Normalise torch or NumPy image inputs to an (H, W[, C]) NumPy array."""
    if _TORCH_AVAILABLE and torch is not None and isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image, dtype=float)

    if (
        img_np.ndim == 3
        and img_np.shape[0] in {1, 3, 4}
        and (img_np.shape[1] > 4 or img_np.shape[2] > 4)
        and img_np.shape[-1] not in {1, 3, 4}
    ):
        img_np = img_np.transpose(1, 2, 0)

    return img_np


def find_slanted_edge_roi(
    image: np.ndarray,
    edge_threshold: float = 0.5,
    min_roi_size: int = 64,
) -> tuple[int, int, int, int] | None:
    """
    Automatically detect a region of interest containing a slanted edge.

    Uses a simple gradient-based search: finds the row/column band with the
    highest gradient magnitude that also has a non-zero angle (slant).

    Parameters
    ----------
    image    : (H, W) or (H, W, C) grayscale/colour numpy array in [0,1]
    ...

    Returns
    -------
    (row_start, row_end, col_start, col_end) or None if no edge found
    """
    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image

    # Sobel gradients
    from scipy.ndimage import sobel
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Require a minimum gradient energy — reject flat/textureless images
    min_grad_energy = 0.01 * gray.size
    if grad_mag.sum() < min_grad_energy:
        return None

    # Find column with maximum integrated gradient
    col_sums = grad_mag.sum(axis=0)
    peak_col = int(col_sums.argmax())

    # Require the peak column to have meaningful gradient
    if col_sums[peak_col] < 1e-4 * gray.shape[0]:
        return None

    # Extract ROI around the peak column
    half = min_roi_size // 2
    col_start = max(0, peak_col - half)
    col_end = min(gray.shape[1], peak_col + half)
    row_start = 0
    row_end = gray.shape[0]

    if (col_end - col_start) < 16 or (row_end - row_start) < min_roi_size:
        return None
    return (row_start, row_end, col_start, col_end)


def measure_esf(
    roi: np.ndarray,
    n_bins: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Edge Spread Function (ESF) from an ROI containing a slanted edge.

    Uses the ISO 12233 oversampling approach: the ~5° tilt gives ~1/tan(5°) ≈ 11x
    oversampling across the edge, yielding sub-pixel ESF resolution.

    Parameters
    ----------
    roi    : (H, W) grayscale array
    n_bins : number of oversampled ESF bins

    Returns
    -------
    x_bins  : (n_bins,) array — sub-pixel position from edge centre
    esf     : (n_bins,) array — mean intensity at each position
    """
    H, W = roi.shape
    # Detect edge angle via linear fit on edge locations
    threshold = (roi.max() + roi.min()) / 2.0
    edge_cols = np.array([
        np.argmax(row > threshold) for row in roi
    ], dtype=float)

    # Fit a line to get the edge slope
    rows = np.arange(H, dtype=float)
    if np.std(edge_cols) < 1e-3:
        slope = 0.0
        intercept = edge_cols.mean()
    else:
        slope, intercept = np.polyfit(rows, edge_cols, 1)

    # Project each pixel onto the axis perpendicular to the edge
    positions = []
    values = []
    for r in range(H):
        edge_x = slope * r + intercept
        for c in range(W):
            dist = c - edge_x  # signed distance from edge
            positions.append(dist)
            values.append(roi[r, c])

    positions = np.array(positions)
    values = np.array(values)

    # Bin into oversampled ESF
    x_min, x_max = positions.min(), positions.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    x_bins = 0.5 * (bins[:-1] + bins[1:])

    esf = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    indices = np.digitize(positions, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    for idx, val in zip(indices, values):
        esf[idx] += val
        counts[idx] += 1
    mask = counts > 0
    esf[mask] /= counts[mask]

    # Fill empty bins by interpolation
    if not mask.all():
        from scipy.interpolate import interp1d
        fill_fn = interp1d(
            x_bins[mask], esf[mask], kind="linear", fill_value="extrapolate"
        )
        esf[~mask] = fill_fn(x_bins[~mask])

    return x_bins, esf


def esf_to_mtf(
    esf: np.ndarray,
    pixel_pitch: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Differentiate ESF → LSF → FFT → MTF.

    Parameters
    ----------
    esf         : (N,) array — Edge Spread Function
    pixel_pitch : physical pixel size in consistent units (default 1.0 = normalised)

    Returns
    -------
    frequencies : (N//2,) array — spatial frequencies in cycles/pixel
    mtf         : (N//2,) array — MTF values in [0, 1]
    """
    # 1. Differentiate to get LSF
    lsf = np.gradient(esf)

    # 2. Window to suppress edge ringing (Hann window)
    window = np.hanning(len(lsf))
    lsf_windowed = lsf * window

    # 3. FFT and take magnitude
    N = len(lsf_windowed)
    fft_lsf = np.fft.fft(lsf_windowed)
    mtf_raw = np.abs(fft_lsf)

    # 4. Normalise: MTF(0) = 1 by convention
    dc = mtf_raw[0]
    if dc > 1e-10:
        mtf_raw /= dc

    # 5. Return positive-frequency half
    half = N // 2
    frequencies = np.fft.fftfreq(N, d=pixel_pitch)[:half]
    mtf = mtf_raw[:half]

    return frequencies, mtf


def measure_mtf(
    image,  # torch.Tensor | np.ndarray
    roi: tuple[int, int, int, int] | None = None,
    n_bins: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end MTF measurement on a (C, H, W) tensor or (H, W[, C]) array.

    If roi is None, attempts automatic edge detection.

    Returns
    -------
    frequencies : (N,) cycles/pixel
    mtf         : (N,) in [0, 1]
    """
    img_np = _to_hwc_image(image)

    # Convert to grayscale for MTF measurement
    if img_np.ndim == 3:
        gray = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
    else:
        gray = img_np.copy()

    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)

    if roi is None:
        roi = find_slanted_edge_roi(gray)
        if roi is None:
            raise ValueError(
                "Could not automatically detect a slanted edge. "
                "Please provide an explicit roi=(row_start, row_end, col_start, col_end)."
            )

    r0, r1, c0, c1 = roi
    roi_gray = gray[r0:r1, c0:c1]

    x_bins, esf = measure_esf(roi_gray, n_bins=n_bins)
    frequencies, mtf = esf_to_mtf(esf)

    return frequencies, mtf


def make_slanted_edge_chart(
    height: int = 256,
    width: int = 256,
    angle_deg: float = 5.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Generate a synthetic slanted-edge chart for operator-level MTF calibration.

    The edge runs near the centre of the image, tilted by `angle_deg` from
    vertical.  Using a synthetic chart (rather than a natural-image ROI)
    guarantees a clean, reproducible edge across all sweep scripts.

    Returns
    -------
    chart_chw : (3, H, W) float32 numpy array in [0, 1]
    roi       : (row_start, row_end, col_start, col_end) centred on the edge
    """
    import math
    slope = math.tan(math.radians(angle_deg))
    img = np.zeros((height, width), dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    for row in range(height):
        edge_col = width / 2.0 + row * slope
        img[row] = (cols >= edge_col).astype(np.float32)
    chart_chw = np.stack([img, img, img], axis=0)   # (3, H, W)
    half = min(64, width // 4)
    roi = (0, height, width // 2 - half, width // 2 + half)
    return chart_chw, roi


def mtf50(frequencies: np.ndarray, mtf: np.ndarray) -> float:
    """
    Return the frequency at which MTF drops to 50% (MTF50).
    Industry-standard sharpness metric.
    """
    # Find first crossing below 0.5
    below = np.where(mtf <= 0.5)[0]
    if len(below) == 0:
        return frequencies[-1]
    idx = below[0]
    if idx == 0:
        return frequencies[0]
    # Linear interpolation between idx-1 and idx
    f0, f1 = frequencies[idx - 1], frequencies[idx]
    m0, m1 = mtf[idx - 1], mtf[idx]
    t = (0.5 - m0) / (m1 - m0)
    return float(f0 + t * (f1 - f0))
