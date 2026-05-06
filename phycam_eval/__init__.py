"""
PhyCam-Eval: Physics-Inspired Camera Degradation Benchmarks for Object Detection Robustness.

The degraded image is modeled as the cascade:
    I_d = N_σ( Q_β( A_φ( I_ideal ) ) )

where:
    A_φ  — optical operator (spectral-phase, Fourier-optics inspired)
    Q_β  — dynamic range compression / quantization operator (ODRC formalism)
    N_σ  — sensor noise operator (Poisson + Gaussian)
"""

__version__ = "0.1.0"
__author__ = "Parthiv Nair"
