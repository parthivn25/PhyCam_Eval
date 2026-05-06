"""
scripts/run_detr_sweep.py — DETR-ResNet-50 cross-architecture sensitivity sweep.

Runs facebook/detr-resnet-50 (HuggingFace) over the same COCO subset used by
the primary YOLOv8n sweeps, with identical protocol: conf=0.25, 200 bootstrap
resamples, same 500 sorted image IDs.  Sweeps defocus, HDR compression, and
sensor noise.

Requires:
    pip install transformers

Usage:
    PYTHONPATH=. python3 scripts/run_detr_sweep.py \\
        --coco-root data/coco \\
        --device mps \\
        --output-dir outputs/detr_sweep

Outputs:
    outputs/detr_sweep/
        results.json   — sensitivity ratios for defocus, hdr, noise
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phycam_eval.degradations import DefocusOperator, HDRCompressionOperator, SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_detr
from phycam_eval.eval.metrics import compute_map, compute_map_ci


ALPHAS      = [0.0, 1.0, 2.0, 3.0]
BETAS       = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
ISO_VALUES  = [100, 800, 3200, 6400]

BASE_ISO        = 100
BASE_GAIN       = 5e-5
BASE_READ_NOISE = 0.002


def _load_detr(device: str):
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError:
        raise ImportError(
            "transformers is required for DETR. "
            "Install with: pip install transformers"
        )
    print("Loading facebook/detr-resnet-50 ...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    if device not in ("cpu",):
        import torch
        model = model.to(device)
    return model, processor


def _run(model, processor, imgs, image_ids, conf, device):
    preds = run_detr(model, processor, imgs, conf=conf, device=device)
    return [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]


def _sweep_point(run_fn, degraded, image_ids, targets, baseline):
    tagged = run_fn(degraded)
    res = compute_map_ci(tagged, targets)
    s = res["map50"] / max(baseline, 1e-6)
    return res["map50"], res["map50_ci"], s


def main():
    p = argparse.ArgumentParser(description="DETR cross-architecture sensitivity sweep")
    p.add_argument("--coco-root",   default="data/coco")
    p.add_argument("--max-images",  type=int, default=500)
    p.add_argument("--image-size",  type=int, default=640)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--conf",        type=float, default=0.25)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--output-dir",  default="outputs/detr_sweep")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.max_images} COCO val images ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root,
        max_images=args.max_images,
        image_size=args.image_size,
        image_offset=args.image_offset,
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    model, processor = _load_detr(args.device)
    run_fn = lambda imgs: _run(model, processor, imgs, image_ids, args.conf, args.device)

    print("\n=== Baseline (clean) ===")
    baseline_map50 = compute_map(run_fn(images), targets)["map50"]
    print(f"  mAP@50 = {baseline_map50:.4f}")

    results = {
        "model": "facebook/detr-resnet-50",
        "max_images": args.max_images,
        "bootstrap_iters": 200,
        "score_thresh": args.conf,
        "baseline_map50": baseline_map50,
    }

    # ── Defocus sweep ──────────────────────────────────────────────────────────
    print("\n── Defocus sweep ─────────────────────────────────────────────────────")
    defocus_sweep = []
    for alpha in ALPHAS:
        op = DefocusOperator(alpha=alpha)
        degraded = [op(img) for img in images]
        map50, ci, s = _sweep_point(run_fn, degraded, image_ids, targets, baseline_map50)
        print(f"  α={alpha:.1f}  mAP={map50:.4f} ±{ci:.4f}  S={s:.4f}")
        defocus_sweep.append({"param": alpha, "map50": map50, "map50_ci": ci, "S": s})
    results["defocus"] = {"baseline_map50": baseline_map50, "sweep": defocus_sweep}

    # ── HDR sweep ──────────────────────────────────────────────────────────────
    print("\n── HDR sweep ─────────────────────────────────────────────────────────")
    hdr_sweep = []
    for beta in BETAS:
        op = HDRCompressionOperator(beta=beta)
        degraded = [op(img) for img in images]
        map50, ci, s = _sweep_point(run_fn, degraded, image_ids, targets, baseline_map50)
        print(f"  β={beta:.1f}  mAP={map50:.4f} ±{ci:.4f}  S={s:.4f}")
        hdr_sweep.append({"param": beta, "map50": map50, "map50_ci": ci, "S": s})
    results["hdr"] = {"baseline_map50": baseline_map50, "sweep": hdr_sweep}

    # ── Noise sweep ────────────────────────────────────────────────────────────
    print("\n── Noise sweep ───────────────────────────────────────────────────────")
    noise_sweep = []
    # Use separate baseline run for noise (matches main sweep protocol)
    noise_baseline = compute_map(run_fn(images), targets)["map50"]
    for iso in ISO_VALUES:
        op = SensorNoiseOperator.from_iso(
            iso,
            base_iso=BASE_ISO,
            base_gain=BASE_GAIN,
            base_read_noise=BASE_READ_NOISE,
            seed=42,
        )
        degraded = [op(img) for img in images]
        map50, ci, s = _sweep_point(run_fn, degraded, image_ids, targets, noise_baseline)
        print(f"  ISO={iso}  mAP={map50:.4f} ±{ci:.4f}  S={s:.4f}")
        noise_sweep.append({"param": iso, "map50": map50, "map50_ci": ci, "S": s})
    results["noise"] = {"baseline_map50": noise_baseline, "sweep": noise_sweep}

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
