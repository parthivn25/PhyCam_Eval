"""
Evaluation harness for PhyCam-Eval.

    harness.py      — runs any detection model over a degraded dataset
    metrics.py      — mAP, IoU, per-class sensitivity
    mtf.py          — MTF measurement via slanted-edge (ISO 12233)
    sensitivity.py  — sensitivity landscape S(θ) = M(f(I_d(θ))) / M(f(I_ideal))
"""
