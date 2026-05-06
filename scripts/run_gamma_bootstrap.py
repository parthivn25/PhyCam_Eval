"""
scripts/run_gamma_bootstrap.py — Direct paired bootstrap CIs on super-additivity γ.

Instead of propagating marginal CIs (which gives conservative ±0.14), this script
runs YOLO once per image per condition, stores per-image predictions, then for each
bootstrap resample draws the SAME image indices for all three conditions (chained,
HDR-marginal, noise-marginal) and computes γ = S_meas/(S_beta × S_iso) directly.
This gives properly correlated CIs that are substantially tighter.

Conditions evaluated for each key (β, ISO=6400) cell:
  - chained(β, 6400)  = N_6400(Q_β(I))
  - marginal_beta(β)  = Q_β(I) with no noise      [ISO=100, gain≈0]
  - marginal_iso(6400)= N_6400(clean I)

Usage:
    PYTHONPATH=. python3 scripts/run_gamma_bootstrap.py \
        --coco-root data/coco --model yolov8n.pt \
        --max-images 200 --bootstrap-iters 500 \
        --output-dir outputs/gamma_bootstrap
"""

import argparse
import json
import numpy as np
from pathlib import Path

from phycam_eval.degradations import HDRCompressionOperator, SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo
from phycam_eval.eval.metrics import compute_map


KEY_BETAS = [0.9, 0.8, 0.7]
BASE_ISO  = 100
BASE_GAIN = 5e-5
BASE_READ_NOISE = 0.002
ISO_NOISY = 6400
RNG_SEED  = 42


def make_noise(iso, seed=RNG_SEED):
    return SensorNoiseOperator.from_iso(
        iso, base_iso=BASE_ISO, base_gain=BASE_GAIN,
        base_read_noise=BASE_READ_NOISE, seed=seed,
    )


def bootstrap_map(preds, targets, image_ids, n_iters, rng):
    """Bootstrap mAP@50 over image_ids using stored predictions."""
    n = len(image_ids)
    maps = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        sel_preds   = [{**preds[i], "image_id": image_ids[i]} for i in idx]
        sel_targets = [targets[i] for i in idx]
        try:
            maps.append(compute_map(sel_preds, sel_targets)["map50"])
        except Exception:
            pass
    return np.array(maps)


def main():
    p = argparse.ArgumentParser(description="Direct paired bootstrap CIs on γ")
    p.add_argument("--coco-root",       default="data/coco")
    p.add_argument("--model",           default="yolov8n.pt")
    p.add_argument("--max-images",      type=int, default=200)
    p.add_argument("--bootstrap-iters", type=int, default=500)
    p.add_argument("--image-size",      type=int, default=640)
    p.add_argument("--device",          default="cpu")
    p.add_argument("--output-dir",      default="outputs/gamma_bootstrap")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print(f"Loading {args.max_images} COCO images ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=args.image_size
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    from ultralytics import YOLO
    model = YOLO(args.model)
    run_fn = lambda imgs: run_yolo(model, imgs, device=args.device)

    # Build all degraded image sets we need
    noise_hi = make_noise(ISO_NOISY)
    conditions = {"clean": images}
    for beta in KEY_BETAS:
        hdr_op = HDRCompressionOperator(beta=beta)
        conditions[f"hdr_{beta}"]     = [hdr_op(img) for img in images]
        conditions[f"chained_{beta}"] = [noise_hi(hdr_op(img)) for img in images]
    conditions["noise_only"] = [noise_hi(img) for img in images]

    # Run YOLO once per condition — store raw predictions
    print("Running YOLO inference on all conditions ...")
    preds = {}
    for name, imgs in conditions.items():
        print(f"  {name} ...")
        preds[name] = run_fn(imgs)

    # Point-estimate baseline
    base_maps = bootstrap_map(preds["clean"], targets, image_ids, 1,
                              np.random.default_rng(0))
    baseline = float(base_maps[0]) if len(base_maps) else 0.0
    print(f"\nBaseline mAP: {baseline:.4f}")

    results = []
    for beta in KEY_BETAS:
        print(f"\n=== β={beta}, ISO={ISO_NOISY} ===")

        # Paired bootstrap: same image indices for all three arms
        n = len(image_ids)
        gamma_boot = []
        for _ in range(args.bootstrap_iters):
            idx = rng.integers(0, n, size=n)

            def _map(name):
                sel_p = [{**preds[name][i], "image_id": image_ids[i]} for i in idx]
                sel_t = [targets[i] for i in idx]
                try:
                    return compute_map(sel_p, sel_t)["map50"]
                except Exception:
                    return np.nan

            s_meas = _map(f"chained_{beta}")
            s_beta = _map(f"hdr_{beta}")          # HDR marginal (no noise)
            s_iso  = _map("noise_only")            # noise marginal (no HDR)

            # Use absolute mAP for marginals; normalise by clean-resample baseline
            clean_b = _map("clean")
            if clean_b > 0 and s_beta > 0 and s_iso > 0:
                g = (s_meas / clean_b) / ((s_beta / clean_b) * (s_iso / clean_b))
                gamma_boot.append(g)

        gamma_boot = np.array(gamma_boot)
        gamma_boot = gamma_boot[np.isfinite(gamma_boot)]

        lo, hi   = np.percentile(gamma_boot, [2.5, 97.5])
        gamma_pt = np.median(gamma_boot)
        print(f"  γ median={gamma_pt:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  "
              f"n_valid={len(gamma_boot)}/{args.bootstrap_iters}")

        results.append({
            "beta": beta, "iso": ISO_NOISY,
            "gamma_median": float(gamma_pt),
            "gamma_ci_lo":  float(lo),
            "gamma_ci_hi":  float(hi),
            "n_bootstrap":  len(gamma_boot),
        })

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "max_images": args.max_images,
            "bootstrap_iters": args.bootstrap_iters,
            "baseline_map50": baseline,
            "cells": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
