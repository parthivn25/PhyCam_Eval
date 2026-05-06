#!/usr/bin/env python3
"""
report/rerun_all.py — master reconfirmation script for PhyCam-Eval.

Reruns every experiment used in the paper, validates key results against
reported values, and regenerates all figures.  Gitignored (report/* in
.gitignore) — local reconfirmation use only.

Usage (from repo root):
    .venv/bin/python report/rerun_all.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        [--device mps|cpu|cuda] \\
        [--jobs 2] \\
        [--skip-existing] \\
        [--dry-run] \\
        [--max-images 500] \\
        [--only STAGE [STAGE ...]]

Stage catalogue (GROUP: name — description — est. wall time at --jobs 2):
  PRIMARY (YOLOv8n):
    1  defocus_sweep        — α sweep (500 imgs)                ~25 min
    2  hdr_sweep            — β sweep (500 imgs)                ~50 min
    3  noise_sweep          — ISO sweep (500 imgs)              ~45 min
    4  chained_sweep        — β×ISO grid (500 imgs)             ~90 min
    5  swap_order           — reversed pipeline (200 imgs)      ~40 min
    6  gaussian_blur        — ImageNet-C control (500 imgs)     ~20 min
  CROSS-ARCH (Faster R-CNN):
    7  defocus_frcnn        — Faster R-CNN defocus              ~30 min
    8  hdr_frcnn            — Faster R-CNN HDR                  ~60 min
    9  noise_frcnn          — Faster R-CNN noise                ~55 min
  CROSS-ARCH (DETR — requires transformers):
   10  detr_sweep           — DETR-ResNet-50 all operators      ~90 min
  SUPPLEMENTARY:
   11  incoherent_defocus   — Incoherent-OTF defocus (500 imgs) ~30 min
   12  rolling_shutter      — RS warped-GT sweep (500 imgs)     ~30 min
  VALIDATION / STATISTICS:
   13  gamma_bootstrap      — Paired bootstrap CI on γ (200 imgs, 500 iters) ~45 min
   14  tonecurve_valid      — OOS tone-curve proxy check        ~60 min
   15  defocus_slice2       — Replication on IDs 500-999        ~25 min
   16  hdr_slice2           — Replication on IDs 500-999        ~50 min
   17  noise_slice2         — Replication on IDs 500-999        ~45 min
  FIGURES:
   18  figure1              — Qualitative comparison panel       ~5  min
   19  figure2              — MTF validation figure              <1  min
   20  figure_chained       — Chained pipeline comparison        <1  min

Run only a subset:  --only defocus_sweep hdr_sweep figure1
Skip done stages:   --skip-existing
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO = Path(__file__).parent.parent.resolve()
PYTHON = str(REPO / ".venv" / "bin" / "python")
SCRIPTS = REPO / "scripts"
REPORT = REPO / "report"
OUTPUTS = REPO / "outputs"

# ─── Paper values to validate against ────────────────────────────────────────

PAPER = {
    "defocus_delta_s_at_alpha3":  0.004,   # |ΔS| at α=3 (both detectors)
    "hdr_beta_10pct":             0.877,   # β̂₁₀% for YOLOv8n
    "hdr_beta_10pct_tol":         0.04,    # ± tolerance
    "noise_iso_10pct":            2854,    # ISO at 10% mAP drop (YOLOv8n)
    "noise_iso_10pct_tol":        800,     # ± tolerance
    "chained_delta_s_0p8_6400":  -0.106,  # ΔS at (β=0.8, ISO=6400)
    "chained_delta_s_tol":        0.04,   # ± tolerance
    "swap_gamma_0p8_6400":        1.012,  # γ_swap at (β=0.8, ISO=6400)
    "swap_gamma_tol":             0.06,
}


# ─── Stage definition ────────────────────────────────────────────────────────

@dataclass
class Stage:
    name: str
    cmd: list[str]               # full argv (PYTHON already expanded)
    cwd: Path                    # working directory for the subprocess
    output_dir: Path             # where results land
    skip_file: str               # file that indicates this stage is done
    est_min: float               # rough runtime estimate in minutes
    validate: Callable[[Path], list[str]] | None = None  # returns list of warnings

    @property
    def done(self) -> bool:
        return (self.output_dir / self.skip_file).exists()


# ─── Validation helpers ───────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _validate_defocus(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    sweep = data.get("sweep", [])
    for pt in sweep:
        alpha = pt.get("alpha", pt.get("param_value", None))
        if alpha is None:
            continue
        s = pt.get("sensitivity", pt.get("map50", 0)) / max(data.get("baseline_map50", 1), 1e-6)
        if abs(alpha - 3.0) < 0.01:
            delta_s = abs(s - 1.0)
            if delta_s > PAPER["defocus_delta_s_at_alpha3"] + 0.03:
                warns.append(
                    f"Defocus ΔS at α=3: {delta_s:.4f}  "
                    f"(paper: <{PAPER['defocus_delta_s_at_alpha3']:.3f})"
                )
    return warns


def _validate_hdr(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    thr = data.get("threshold_map_10pct_beta")
    if thr is None:
        warns.append("threshold_map_10pct_beta missing from results.json")
    else:
        paper_thr = PAPER["hdr_beta_10pct"]
        tol = PAPER["hdr_beta_10pct_tol"]
        if abs(thr - paper_thr) > tol:
            warns.append(
                f"HDR β̂₁₀%: {thr:.4f}  (paper: {paper_thr:.3f} ± {tol:.3f})"
            )
    return warns


def _validate_noise(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    thr = data.get("threshold_map_10pct_iso") or data.get("threshold_map_10pct_param")
    if thr is None:
        # try to derive from sweep
        sweep = data.get("sweep", [])
        baseline = data.get("baseline_map50", None)
        if baseline and sweep:
            target = 0.9 * baseline
            for i in range(1, len(sweep)):
                if sweep[i].get("map50", baseline) <= target:
                    prev, curr = sweep[i - 1], sweep[i]
                    iso_prev = prev.get("iso", prev.get("param_value", 0))
                    iso_curr = curr.get("iso", curr.get("param_value", 0))
                    m_prev = prev.get("map50", baseline)
                    m_curr = curr.get("map50", baseline)
                    frac = (target - m_prev) / (m_curr - m_prev + 1e-9)
                    thr = iso_prev + frac * (iso_curr - iso_prev)
                    break
    if thr is None:
        warns.append("Cannot derive ISO 10% threshold from results.json")
    else:
        paper_thr = PAPER["noise_iso_10pct"]
        tol = PAPER["noise_iso_10pct_tol"]
        if abs(thr - paper_thr) > tol:
            warns.append(
                f"Noise ISO₁₀%: {thr:.0f}  (paper: {paper_thr:.0f} ± {tol:.0f})"
            )
    return warns


def _validate_chained(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    baseline = data.get("baseline_map50", None)
    grid = data.get("grid", data.get("results", []))
    if not baseline or not grid:
        return ["Cannot find baseline_map50 or grid in results.json"]

    # Find (β=0.8, ISO=6400) measured S
    target_beta, target_iso = 0.8, 6400
    for row in grid:
        b = row.get("beta", None)
        iso = row.get("iso", None)
        if b is None or iso is None:
            continue
        if abs(b - target_beta) < 0.01 and abs(iso - target_iso) < 1:
            s_meas = row.get("map50", 0) / max(baseline, 1e-6)
            # marginal S values for independence prediction
            s_beta = row.get("s_beta_marginal", None)
            s_iso = row.get("s_iso_marginal", None)
            if s_beta and s_iso:
                s_pred = s_beta * s_iso
                delta_s = s_meas - s_pred
                paper_ds = PAPER["chained_delta_s_0p8_6400"]
                tol = PAPER["chained_delta_s_tol"]
                if abs(delta_s - paper_ds) > tol:
                    warns.append(
                        f"Chained ΔS at (β=0.8, ISO=6400): {delta_s:.4f}  "
                        f"(paper: {paper_ds:.3f} ± {tol:.3f})"
                    )
            break
    return warns


def _validate_swap(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    results = data.get("results", data.get("ordering_comparison", []))
    for row in results:
        b = row.get("beta", None)
        iso = row.get("iso", None)
        if b and iso and abs(b - 0.8) < 0.01 and abs(iso - 6400) < 1:
            gamma_swap = row.get("gamma_swap", None)
            if gamma_swap is not None:
                paper_g = PAPER["swap_gamma_0p8_6400"]
                tol = PAPER["swap_gamma_tol"]
                if abs(gamma_swap - paper_g) > tol:
                    warns.append(
                        f"Swap γ at (β=0.8, ISO=6400): {gamma_swap:.4f}  "
                        f"(paper: {paper_g:.3f} ± {tol:.3f})"
                    )
            break
    return warns


def _validate_figures(out: Path) -> list[str]:
    warns = []
    for name in ["figure1_degradation_comparison.png", "figure1_degradation_comparison.pdf"]:
        p = out / name
        if not p.exists():
            warns.append(f"Missing: {name}")
        elif p.stat().st_size < 50_000:
            warns.append(f"Suspiciously small: {name} ({p.stat().st_size} bytes)")
    return warns


def _validate_figure2(out: Path) -> list[str]:
    warns = []
    for name in ["figure2_mtf_validation.png", "figure2_mtf_validation.pdf"]:
        p = out / name
        if not p.exists():
            warns.append(f"Missing: {name}")
    return warns


def _validate_detr(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    defocus = data.get("defocus", {}).get("sweep", [])
    for pt in defocus:
        if abs(pt.get("param", -1) - 3.0) < 0.01:
            s = pt.get("S", 1.0)
            if abs(s - 1.0) > 0.05:
                warns.append(f"DETR defocus S at α=3: {s:.4f}  (expected ≈1.0)")
    hdr = data.get("hdr", {}).get("sweep", [])
    for pt in hdr:
        if abs(pt.get("param", -1) - 0.3) < 0.01:
            s = pt.get("S", 1.0)
            if s > 0.50:
                warns.append(f"DETR HDR S at β=0.3: {s:.4f}  (expected <0.50)")
    return warns


def _validate_incoherent(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    for pt in data.get("sweep", []):
        if abs(pt.get("alpha", -1) - 3.0) < 0.01:
            s = pt.get("sensitivity", 0)
            otf = pt.get("otf_mean_abs", 1.0)
            if abs(s - 1.0) > 0.05:
                warns.append(f"Incoherent defocus S at α=3: {s:.4f}  (expected ≈1.0)")
            if not (0.55 < otf < 0.80):
                warns.append(f"Incoherent OTF mean at α=3: {otf:.4f}  (expected 0.55–0.80)")
    return warns


def _validate_rolling_shutter(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    baseline = data.get("baseline_map50", None)
    sweep = data.get("sweep", [])
    if baseline and sweep:
        for pt in sweep:
            disp = pt.get("max_displacement_px", 0)
            if abs(disp - 30.0) < 1:
                s = pt.get("map50", baseline) / max(baseline, 1e-6)
                if abs(s - 1.0) > 0.08:
                    warns.append(
                        f"Rolling-shutter S at 30px: {s:.4f}  (expected within ±0.08 of 1.0)"
                    )
    return warns


def _validate_tonecurve(out: Path) -> list[str]:
    try:
        data = _load_json(out / "results.json")
    except FileNotFoundError:
        return ["results.json not found"]
    warns = []
    r = data.get("pearson_r_oos", 0.0)
    if r < 0.90:
        warns.append(f"Tone-curve Pearson r: {r:.4f}  (paper: 0.953, expected >0.90)")
    n_within = data.get("n_within_ci", 0)
    if n_within < 6:
        warns.append(f"Tone-curve curves within CI: {n_within}  (expected ≥6)")
    return warns


def _validate_figure_chained(out: Path) -> list[str]:
    warns = []
    for name in ["figure_chained_comparison.png", "figure_chained_comparison.pdf"]:
        p = out / name
        if not p.exists():
            warns.append(f"Missing: {name}")
    return warns


# ─── Stage runner ─────────────────────────────────────────────────────────────

class Runner:
    """Thread-safe subprocess runner that tees output to a shared log."""

    def __init__(self, log_path: Path) -> None:
        self._log = open(log_path, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._start = time.monotonic()

    def _emit(self, text: str) -> None:
        with self._lock:
            sys.stdout.write(text)
            sys.stdout.flush()
            self._log.write(text)

    def _header(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"\n{'─'*70}\n[{ts}]  {msg}\n{'─'*70}\n"
        self._emit(line)

    def run_stage(self, stage: Stage, dry_run: bool, skip_existing: bool) -> tuple[bool, float]:
        """Run one stage. Returns (success, elapsed_seconds)."""
        elapsed = 0.0

        if skip_existing and stage.done:
            self._emit(f"  ↷ SKIP  {stage.name}  (output exists)\n")
            return True, 0.0

        self._header(f"STAGE: {stage.name}  [est. {stage.est_min:.0f} min]")

        if dry_run:
            self._emit(f"  [dry-run]  {' '.join(stage.cmd)}\n")
            return True, 0.0

        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                stage.cmd,
                cwd=stage.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            prefix = f"[{stage.name}] "
            for line in proc.stdout:
                self._emit(prefix + line)
            proc.wait()
            elapsed = time.monotonic() - t0
            success = proc.returncode == 0
        except Exception as exc:
            self._emit(f"  ERROR launching {stage.name}: {exc}\n")
            elapsed = time.monotonic() - t0
            success = False

        status = "✓ OK" if success else "✗ FAILED"
        self._emit(f"\n  {status}  {stage.name}  ({elapsed/60:.1f} min)\n")
        return success, elapsed

    def close(self) -> None:
        self._log.close()


# ─── Stage catalogue ──────────────────────────────────────────────────────────

def build_stages(args: argparse.Namespace) -> list[Stage]:
    """Construct all stage objects from CLI args."""
    # out_base is the root for all experiment outputs.
    # Changing --output-base keeps all outputs in a separate directory so
    # reruns never silently overwrite existing paper data.
    out_base = REPO / args.output_base

    def _od(name: str) -> str:
        """--output-dir flag value for a named subdirectory."""
        return f"--output-dir={args.output_base}/{name}"

    def _fig_od() -> str:
        """--output-dir flag for figure scripts (cwd=REPORT, so relative)."""
        return f"--output-dir=../{args.output_base}"

    common = [
        f"--coco-root={args.coco_root}",
        f"--model={args.model}",
        f"--device={args.device}",
        f"--max-images={args.max_images}",
    ]
    common_nomodel = [
        f"--coco-root={args.coco_root}",
        f"--device={args.device}",
        f"--max-images={args.max_images}",
    ]

    stages: list[Stage] = [

        # ── Primary (YOLOv8n) ─────────────────────────────────────────────────
        Stage(
            name="defocus_sweep",
            cmd=[PYTHON, str(SCRIPTS / "run_defocus_sweep.py")] + common + [
                _od("defocus_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "defocus_sweep",
            skip_file="results.json",
            est_min=25,
            validate=_validate_defocus,
        ),
        Stage(
            name="hdr_sweep",
            cmd=[PYTHON, str(SCRIPTS / "run_hdr_sweep.py")] + common + [
                _od("hdr_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "hdr_sweep",
            skip_file="results.json",
            est_min=50,
            validate=_validate_hdr,
        ),
        Stage(
            name="noise_sweep",
            cmd=[PYTHON, str(SCRIPTS / "run_noise_sweep.py")] + common + [
                _od("noise_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "noise_sweep",
            skip_file="results.json",
            est_min=45,
            validate=_validate_noise,
        ),
        Stage(
            name="chained_sweep",
            cmd=[PYTHON, str(SCRIPTS / "run_chained_sweep.py")] + common + [
                _od("chained_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "chained_sweep",
            skip_file="results.json",
            est_min=90,
            validate=_validate_chained,
        ),
        Stage(
            name="swap_order",
            cmd=[PYTHON, str(SCRIPTS / "run_swap_order_experiment.py")] + [
                f"--coco-root={args.coco_root}",
                f"--device={args.device}",
                # swap_order runs 2×19 conditions; cap at 500 for tractability
                f"--max-images={min(args.max_images, 500)}",
                _od("swap_order_experiment"),
            ],
            cwd=REPO,
            output_dir=out_base / "swap_order_experiment",
            skip_file="results.json",
            est_min=40,
            validate=_validate_swap,
        ),
        Stage(
            name="gaussian_blur",
            cmd=[PYTHON, str(SCRIPTS / "run_gaussian_blur_sweep.py")] + common + [
                _od("gaussian_blur_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "gaussian_blur_sweep",
            skip_file="results.json",
            est_min=20,
            validate=None,
        ),

        # ── Cross-architecture (Faster R-CNN) ─────────────────────────────────
        Stage(
            name="defocus_frcnn",
            cmd=[PYTHON, str(SCRIPTS / "run_defocus_sweep.py")] + common + [
                "--detector=fasterrcnn",
                _od("defocus_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "defocus_sweep",
            skip_file="results_fasterrcnn.json",
            est_min=30,
            validate=None,
        ),
        Stage(
            name="hdr_frcnn",
            cmd=[PYTHON, str(SCRIPTS / "run_hdr_sweep.py")] + common + [
                "--detector=fasterrcnn",
                _od("hdr_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "hdr_sweep",
            skip_file="results_fasterrcnn.json",
            est_min=60,
            validate=None,
        ),
        Stage(
            name="noise_frcnn",
            cmd=[PYTHON, str(SCRIPTS / "run_noise_sweep.py")] + common + [
                "--detector=fasterrcnn",
                _od("noise_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "noise_sweep",
            skip_file="results_fasterrcnn.json",
            est_min=55,
            validate=None,
        ),

        # ── Cross-architecture (DETR) ─────────────────────────────────────────
        Stage(
            name="detr_sweep",
            cmd=[PYTHON, str(SCRIPTS / "run_detr_sweep.py")] + common_nomodel + [
                _od("detr_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "detr_sweep",
            skip_file="results.json",
            est_min=90,
            validate=_validate_detr,
        ),

        # ── Supplementary sweeps ──────────────────────────────────────────────
        Stage(
            name="incoherent_defocus",
            cmd=[PYTHON, str(SCRIPTS / "run_incoherent_defocus_sweep.py")] + common + [
                _od("incoherent_defocus_sweep"),
            ],
            cwd=REPO,
            output_dir=out_base / "incoherent_defocus_sweep",
            skip_file="results.json",
            est_min=30,
            validate=_validate_incoherent,
        ),
        Stage(
            name="rolling_shutter",
            cmd=[PYTHON, str(SCRIPTS / "run_rolling_shutter_sweep.py")] + common + [
                _od("rolling_shutter_sweep_warped"),
            ],
            cwd=REPO,
            output_dir=out_base / "rolling_shutter_sweep_warped",
            skip_file="results.json",
            est_min=30,
            validate=_validate_rolling_shutter,
        ),

        # ── Gamma bootstrap (paired CI on super-additivity γ) ────────────────
        Stage(
            name="gamma_bootstrap",
            cmd=[PYTHON, str(SCRIPTS / "run_gamma_bootstrap.py")] + [
                f"--coco-root={args.coco_root}",
                f"--model={args.model}",
                f"--device={args.device}",
                # 200 images is the paper protocol; cap at max_images for consistency
                f"--max-images={min(args.max_images, 500)}",
                "--bootstrap-iters=500",
                _od("gamma_bootstrap"),
            ],
            cwd=REPO,
            output_dir=out_base / "gamma_bootstrap",
            skip_file="results.json",
            est_min=45,
            validate=None,
        ),

        # ── Validation ────────────────────────────────────────────────────────
        Stage(
            name="tonecurve_valid",
            cmd=[PYTHON, str(SCRIPTS / "run_tonecurve_validation.py")] + common + [
                _od("hdr_tonecurve_validation_oos"),
            ],
            cwd=REPO,
            output_dir=out_base / "hdr_tonecurve_validation_oos",
            skip_file="results.json",
            est_min=60,
            validate=_validate_tonecurve,
        ),

        # ── Second-slice replication (IDs 500–999) ────────────────────────────
        Stage(
            name="defocus_slice2",
            cmd=[PYTHON, str(SCRIPTS / "run_defocus_sweep.py")] + common + [
                "--image-offset=500",
                _od("defocus_sweep_slice2"),
            ],
            cwd=REPO,
            output_dir=out_base / "defocus_sweep_slice2",
            skip_file="results.json",
            est_min=25,
            validate=None,
        ),
        Stage(
            name="hdr_slice2",
            cmd=[PYTHON, str(SCRIPTS / "run_hdr_sweep.py")] + common + [
                "--image-offset=500",
                _od("hdr_sweep_slice2"),
            ],
            cwd=REPO,
            output_dir=out_base / "hdr_sweep_slice2",
            skip_file="results.json",
            est_min=50,
            validate=None,
        ),
        Stage(
            name="noise_slice2",
            cmd=[PYTHON, str(SCRIPTS / "run_noise_sweep.py")] + common + [
                "--image-offset=500",
                _od("noise_sweep_slice2"),
            ],
            cwd=REPO,
            output_dir=out_base / "noise_sweep_slice2",
            skip_file="results.json",
            est_min=45,
            validate=None,
        ),

        # ── Figures ───────────────────────────────────────────────────────────
        Stage(
            name="figure1",
            cmd=[PYTHON, "generate_figure1.py",
                 "--coco-root=../data/coco",
                 f"--model=../yolov8n.pt",
                 f"--device={args.device}",
                 _fig_od(),
            ],
            cwd=REPORT,
            output_dir=out_base,
            skip_file="figure1_degradation_comparison.png",
            est_min=5,
            validate=_validate_figures,
        ),
        Stage(
            name="figure2",
            cmd=[PYTHON, "generate_figure2.py",
                 _fig_od(),
            ],
            cwd=REPORT,
            output_dir=out_base,
            skip_file="figure2_mtf_validation.png",
            est_min=1,
            validate=_validate_figure2,
        ),
        Stage(
            name="figure_chained",
            cmd=[PYTHON, "generate_figure_chained.py",
                 _fig_od(),
            ],
            cwd=REPORT,
            output_dir=out_base,
            skip_file="figure_chained_comparison.png",
            est_min=1,
            validate=_validate_figure_chained,
        ),
    ]
    return stages


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="PhyCam-Eval master reconfirmation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--coco-root",    default="data/coco",  help="COCO data root")
    p.add_argument("--model",        default="yolov8n.pt", help="YOLO weights path")
    p.add_argument("--device",       default="cpu",        help="Inference device (cpu/mps/cuda)")
    p.add_argument("--max-images",   type=int, default=500, help="Images per sweep (paper: 500)")
    p.add_argument("--jobs",         type=int, default=2,   help="Parallel stages (default 2)")
    p.add_argument("--output-base",  default="outputs",
                   help="Root directory for all outputs.  Change this to avoid "
                        "overwriting existing paper data — e.g. "
                        "outputs/run_20260506.  Default: outputs (paper canonical).")
    p.add_argument("--skip-existing", action="store_true", help="Skip stages whose output exists")
    p.add_argument("--dry-run",      action="store_true",   help="Print commands, do not execute")
    p.add_argument("--only",         nargs="+", metavar="STAGE",
                   help="Run only these stage names (e.g. --only hdr_sweep figure2)")
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = REPORT / f"rerun_{timestamp}.log"

    out_base_path = REPO / args.output_base
    out_base_path.mkdir(parents=True, exist_ok=True)

    stages = build_stages(args)
    if args.only:
        stages = [s for s in stages if s.name in args.only]
        if not stages:
            print(f"No stages matched: {args.only}")
            return 1

    total_est = sum(s.est_min for s in stages) / args.jobs
    print(f"\nPhyCam-Eval reconfirmation  —  {timestamp}")
    print(f"Repo:      {REPO}")
    print(f"Python:    {PYTHON}")
    print(f"Outputs:   {out_base_path}")
    print(f"Log:       {log_path}")
    print(f"Stages:    {len(stages)}  ({', '.join(s.name for s in stages)})")
    print(f"Jobs:      {args.jobs}  parallel")
    print(f"Est. time: ~{total_est:.0f} min total wall-clock")
    if args.output_base == "outputs":
        print("⚠  Output base is 'outputs' (paper canonical) — "
              "use --output-base outputs/run_YYYYMMDD to avoid overwriting.")
    if args.skip_existing:
        print("Mode:      skip existing outputs")
    if args.dry_run:
        print("Mode:      DRY RUN")
    print()

    # ── Write run manifest ────────────────────────────────────────────────────
    if not args.dry_run:
        import subprocess as _sp
        try:
            git_hash = _sp.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
            ).strip()
        except Exception:
            git_hash = "unknown"
        manifest = {
            "timestamp":   timestamp,
            "git_hash":    git_hash,
            "output_base": args.output_base,
            "device":      args.device,
            "max_images":  args.max_images,
            "model":       args.model,
            "stages":      [s.name for s in stages],
        }
        (out_base_path / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    runner = Runner(log_path)
    results: dict[str, tuple[bool, float]] = {}
    warnings: dict[str, list[str]] = {}

    wall_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(runner.run_stage, stage, args.dry_run, args.skip_existing): stage
            for stage in stages
        }
        done_count = 0
        for future in as_completed(futures):
            stage = futures[future]
            ok, elapsed = future.result()
            done_count += 1
            results[stage.name] = (ok, elapsed)

            remaining = len(stages) - done_count
            runner._emit(
                f"\n  Progress: {done_count}/{len(stages)} stages done"
                f"  |  {remaining} remaining\n"
            )

            # Validate results
            if ok and not args.dry_run and stage.validate:
                try:
                    warns = stage.validate(stage.output_dir)
                except Exception as exc:
                    warns = [f"Validation error: {exc}"]
                warnings[stage.name] = warns
                if warns:
                    runner._emit(
                        f"\n  ⚠ Validation warnings for {stage.name}:\n"
                        + "".join(f"    • {w}\n" for w in warns)
                    )
                else:
                    runner._emit(f"  ✓ Validation passed for {stage.name}\n")

    wall_elapsed = time.monotonic() - wall_start

    # ── Summary ───────────────────────────────────────────────────────────────
    runner._emit(f"\n{'═'*70}\n  RERUN SUMMARY  ({wall_elapsed/60:.1f} min wall-clock)\n{'═'*70}\n")
    all_ok = True
    for stage in stages:
        ok, elapsed = results.get(stage.name, (False, 0.0))
        status = "✓" if ok else "✗"
        skip_note = " [skipped]" if elapsed == 0.0 and ok else f"  ({elapsed/60:.1f} min)"
        warns = warnings.get(stage.name, [])
        warn_note = f"  ⚠ {len(warns)} warning(s)" if warns else ""
        runner._emit(f"  {status}  {stage.name:<22}{skip_note}{warn_note}\n")
        if not ok:
            all_ok = False

    if all_ok:
        runner._emit("\n  All stages completed successfully.\n")
    else:
        runner._emit("\n  ✗ Some stages failed — check log for details.\n")

    # ── Paper value comparison table ──────────────────────────────────────────
    runner._emit(f"\n{'─'*70}\n  PAPER VALUE COMPARISON\n{'─'*70}\n")
    all_warns: list[str] = []
    for wlist in warnings.values():
        all_warns.extend(wlist)
    if not all_warns:
        runner._emit("  All validated results match paper values.\n")
    else:
        runner._emit(f"  {len(all_warns)} discrepancy/discrepancies found:\n")
        for w in all_warns:
            runner._emit(f"    ⚠ {w}\n")

    runner._emit(f"\n  Log written to: {log_path}\n")
    runner.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
