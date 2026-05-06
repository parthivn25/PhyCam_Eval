# PhyCam-Eval

**Physics-Inspired Camera Degradation Benchmarks for Object Detection Robustness**

---

## Overview

Existing robustness benchmarks (ImageNet-C and its derivatives) apply heuristic corruptions — Gaussian blur, uniform noise, JPEG compression — with abstract severity levels that are difficult to relate to camera or ISP parameters.

PhyCam-Eval expresses each simplified degradation in camera-native units: wavefront phase strength α (quadratic pupil-phase defocus), spectral amplitude exponent β (HDR compression via the ODRC formalism), and ISO-calibrated mixed Poisson-Gaussian sensor noise. The operators are intended for reproducible sensitivity screening, not as a complete camera simulation.

---

## Image Formation Model

```
I_d = N_σ( Q_β( A_φ( I_ideal ) ) )
```

| Operator | Symbol | Physical parameter | Module |
|---|---|---|---|
| Optical (quadratic pupil phase) | A_φ | Defocus α | `degradations/optical.py` |
| HDR compression (ODRC) | Q_β | Compression ratio β | `degradations/hdr.py` |
| Sensor noise | N_σ | Gain g (ISO), read noise σ_r | `degradations/noise.py` |

---

## Project Structure

```
phycam-eval/
├── README.md
├── pyproject.toml
├── cpp/                    # C++ core + pybind bindings + C++ tests
├── phycam-eval/            # Python package
│   ├── degradations/       # Optical / HDR / noise operators
│   ├── eval/               # YOLO harness, COCO helpers, MTF, sensitivity
│   └── benchmarks/         # Dataset helpers and ImageNet-C comparison utilities
├── report/                 # (Optional) LaTeX manuscript
├── scripts/                # Reproducible sweeps, figures, and benchmarks
└── tests/                  # Python unit tests
```

Generated figures/results are written to `outputs/` when you run the scripts.

---

## Installation

```bash
git clone https://github.com/parthiv-nair/phycam-eval.git
cd phycam-eval
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, PyTorch ≥ 2.1.

To run detector experiment scripts, install optional extras:

```bash
pip install -e ".[dev,experiments]"
```

---

## Quick Start

```python
from phycam-eval.degradations import DefocusOperator, SensorNoiseOperator, HDRCompressionOperator

# Load any (C, H, W) float32 image tensor in [0, 1]
image = ...

# Apply camera-parameterized degradations
defocused = DefocusOperator(alpha=1.5)(image)
noisy     = SensorNoiseOperator.from_iso(iso=3200)(image)
hdr_comp  = HDRCompressionOperator(beta=0.4)(image)

# Chain the full pipeline: I_d = N_σ( Q_β( A_φ( I_ideal ) ) )
pipeline = lambda img: (
    SensorNoiseOperator(gain=8.0)(
        HDRCompressionOperator(beta=0.5)(
            DefocusOperator(alpha=1.0)(img)
        )
    )
)
degraded = pipeline(image)
```

### Sensitivity Sweep

```python
from phycam-eval.eval.sensitivity import SensitivitySweep

sweep = SensitivitySweep(
    param_name="hdr_beta",
    param_values=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    baseline_map=0.377,
)

for beta in sweep.param_values:
    op = HDRCompressionOperator(beta=beta)
    map50 = ...  # run harness, get mAP
    sweep.add(beta, map50=map50)

print(f"10% mAP drop at beta = {sweep.find_threshold_param('map50', 0.10):.3f}")
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dataset Layout

The harness expects MS COCO val2017 at `--coco-root` (default `data/coco`):

```
data/coco/
├── images/val2017/                       # JPEG images
└── annotations/instances_val2017.json    # COCO ground truth
```

Download from <https://cocodataset.org/#download>. The 500-image subset is selected deterministically by sorted `image_id` (no auxiliary split file needed); the bootstrap seed is `42`.

### C++ Build and Benchmark

```bash
./scripts/build.sh
python3 -c "from phycam_eval.degradations import DefocusOperator; print(DefocusOperator(1.5))"
./cpp/build/phycam_benchmark
```
