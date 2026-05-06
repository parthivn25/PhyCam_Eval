"""
PhyCam-Eval degradation operators.

Each operator is physics-inspired and parameterized by a camera-relevant variable:

    optical.py       — A_φ  : spectral-phase optical operator
    hdr.py           — Q_β  : HDR dynamic range compression (ODRC formalism)
    noise.py         — N_σ  : mixed Poisson-Gaussian sensor noise
"""

from .hdr import HDRCompressionOperator
from .noise import SensorNoiseOperator
from .optical import (
    AstigmatismOperator,
    DefocusOperator,
    LowLightOperator,
)

__all__ = [
    "DefocusOperator",
    "AstigmatismOperator",
    "LowLightOperator",
    "HDRCompressionOperator",
    "SensorNoiseOperator",
]
