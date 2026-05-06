"""
scripts/run_gaussian_blur_sweep.py — ImageNet-C style Gaussian blur sensitivity sweep.

Runs YOLOv8n (and optionally Faster R-CNN) over COCO images degraded with
Gaussian blur across a range of σ values, using an identical protocol to the
PhyCam-Eval operator sweeps.  Provides a direct numerical comparison against
the phase-only defocus sweep (run_defocus_sweep.py):

    - Phase-only defocus (A_φ):  |H(f)| ≡ 1  →  ΔS ≤ 0.014
    - Gaussian blur (ImageNet-C): |H(f)| → 0  →  drops 10% threshold at σ ≈ ?

The Gaussian OTF is H(f) = exp(-2π²σ²|f|²), which attenuates amplitude.
This is the key contrast the paper draws in Section 5.1 and Figure 1.

Usage:
    PYTHONPATH=. python3 scripts/run_gaussian_blur_sweep.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --max-images 500 \\
        --output-dir outputs/gaussian_blur_sweep

Outputs:
    outputs/gaussian_blur_sweep/
        results.json                 — mAP@50 + CI + MTF50 for each sigma
        figure_gaussian_sensitivity.png/.pdf
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo, run_fasterrcnn
from phycam_eval.eval.metrics import compute_map, compute_map_ci
from phycam_eval.eval.mtf import make_slanted_edge_chart, measure_mtf, mtf50
from phycam_eval.eval.sensitivity import SensitivitySweep


def gaussian_blur_chw(img_chw: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur independently per channel to a CHW float32 image."""
    if sigma == 0.0:
        return img_chw.copy()
    out = np.empty_like(img_chw)
    for c in range(img_chw.shape[0]):
        out[c] = gaussian_filter(img_chw[c], sigma=sigma)
    return np.clip(out, 0.0, 1.0)


def _load_detector(args):
    if args.detector == "fasterrcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
        )
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights["COCO_V1"]
        )
        model.eval()
        model.to(args.device)
        run_fn = lambda imgs: run_fasterrcnn(model, imgs, conf=0.25, device=args.device)
        return model, run_fn, "fasterrcnn"
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        run_fn = lambda imgs: run_yolo(model, imgs, device=args.device)
        return model, run_fn, "yolo"


def main():
    p = argparse.ArgumentParser(description="Gaussian blur sensitivity sweep (ImageNet-C style)")
    p.add_argument("--coco-root",   default="data/coco")
    p.add_argument("--model",       default="yolov8n.pt")
    p.add_argument("--detector",    default="yolo", choices=["yolo", "fasterrcnn"])
    p.add_argument("--max-images",    type=int, default=500)
    p.add_argument("--image-offset",  type=int, default=0,
                   help="Skip first N sorted COCO image IDs (use 500 for second slice)")
    p.add_argument("--image-size",  type=int, default=640)
    p.add_argument("--output-dir",  default="outputs/gaussian_blur_sweep")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--sigmas",      default="0,0.5,1.0,1.5,2.0,2.5,3.0",
                   help="Comma-separated Gaussian σ values (pixels, post-resize)")
    p.add_argument("--bootstrap-iters", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [float(s) for s in args.sigmas.split(",")]

    print(f"Loading COCO val images from {args.coco_root} ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=args.image_size,
        image_offset=args.image_offset,
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)
    print(f"  Loaded {len(images)} images (image_size={args.image_size})")

    print(f"\nLoading detector: {args.detector}")
    model, run_fn, det_tag = _load_detector(args)

    mtf_chart, mtf_roi = make_slanted_edge_chart()

    # Baseline
    print("\n=== Baseline (clean images) ===")
    clean_preds = run_fn(images)
    map_res = compute_map_ci(
        [{**pr, "image_id": iid} for pr, iid in zip(clean_preds, image_ids)],
        targets,
        n_bootstrap=args.bootstrap_iters,
    )
    baseline_map50 = map_res["map50"]
    baseline_map50_ci = map_res["map50_ci"]
    print(f"  mAP@50 = {baseline_map50:.4f} ±{baseline_map50_ci:.4f}")

    try:
        freqs, mtf_vals = measure_mtf(mtf_chart, roi=mtf_roi)
        baseline_mtf50 = mtf50(freqs, mtf_vals)
    except Exception:
        baseline_mtf50 = 0.0
    print(f"  MTF50  = {baseline_mtf50:.4f} cy/px (synthetic slanted-edge chart)")

    sweep = SensitivitySweep(
        param_name="Gaussian σ (px)",
        param_values=sigmas,
        baseline_map=max(baseline_map50, 1e-6),
        baseline_mtf=max(baseline_mtf50, 1e-6),
    )
    all_data = []

    for sigma in sigmas:
        print(f"\n=== σ = {sigma:.2f} ===")

        t0 = time.perf_counter()
        degraded = [gaussian_blur_chw(img, sigma) for img in images]
        print(f"  Blur: {(time.perf_counter()-t0)*1000:.0f} ms")

        preds = run_fn(degraded)
        tagged = [{**pr, "image_id": iid} for pr, iid in zip(preds, image_ids)]
        res = compute_map_ci(tagged, targets, n_bootstrap=args.bootstrap_iters)
        map50_val = res["map50"]
        map50_ci  = res["map50_ci"]
        s_val = map50_val / max(baseline_map50, 1e-6)
        print(f"  mAP@50 = {map50_val:.4f} ±{map50_ci:.4f}  (S = {s_val:.3f})")

        mtf50_val = 0.0
        try:
            deg_chart = gaussian_blur_chw(mtf_chart, sigma)
            freqs, mtf_vals = measure_mtf(deg_chart, roi=mtf_roi)
            mtf50_val = mtf50(freqs, mtf_vals)
            print(f"  MTF50  = {mtf50_val:.4f} cy/px  "
                  f"(S = {mtf50_val/max(baseline_mtf50,1e-6):.3f})")
        except Exception as e:
            print(f"  MTF50 failed: {e}")

        sweep.add(sigma, map50=map50_val, mtf50=mtf50_val, map50_ci=map50_ci)
        all_data.append({
            "sigma": sigma,
            "map50": map50_val,
            "map50_ci": map50_ci,
            "mtf50": mtf50_val,
            "S": s_val,
        })

    thr_map = sweep.find_threshold_param("map50", 0.10)
    thr_mtf = sweep.find_threshold_param("mtf50", 0.10)
    print(f"\n10% mAP  drop threshold: σ ≈ {thr_map:.2f} px" if thr_map
          else "\nNo 10% mAP drop within tested range")
    print(f"10% MTF50 drop threshold: σ ≈ {thr_mtf:.2f} px" if thr_mtf else "")

    suffix = f"_{det_tag}" if det_tag != "yolo" else ""
    results_path = out_dir / f"results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump({
            "detector": det_tag,
            "max_images": args.max_images,
            "bootstrap_iters": args.bootstrap_iters,
            "baseline_map50": baseline_map50,
            "baseline_map50_ci": baseline_map50_ci,
            "baseline_mtf50": baseline_mtf50,
            "sweep": all_data,
            "threshold_map_10pct_sigma": thr_map,
            "threshold_mtf_10pct_sigma": thr_mtf,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    fig_path = str(out_dir / f"figure_gaussian_sensitivity{suffix}")
    sweep.plot(
        save_path=fig_path + ".png",
        title=f"Gaussian Blur Sensitivity ({det_tag}, n={args.max_images})",
        close=True,
    )
    print(f"Figure saved to {fig_path}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
