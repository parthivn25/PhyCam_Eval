"""
scripts/run_rolling_shutter_sweep.py — Rolling-shutter sensitivity sweep (warped GT).

Applies a horizontal rolling-shutter warp to each image (row y shifted by
velocity_x * readout_time * y/H pixels) and simultaneously warps the ground-truth
bounding boxes to match, so that mAP measures detection quality on the warped
image rather than GT-alignment error.

Physical model:
    x_shifted(y) = x - velocity_x * readout_time * (y / H)
    Box shift at midpoint y_mid:  dx = velocity_x * readout_time * (y_mid / H)

Usage:
    PYTHONPATH=. python3 scripts/run_rolling_shutter_sweep.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --output-dir outputs/rolling_shutter_sweep_warped

Outputs:
    outputs/rolling_shutter_sweep_warped/
        results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phycam_eval.eval.coco import (
    build_coco_targets,
    load_coco_images,
    run_yolo,
    warp_image_rolling_shutter,
    warp_targets_rolling_shutter,
)
from phycam_eval.eval.metrics import compute_map, compute_map_ci


# readout_time values; max_displacement_px = velocity_x * readout_time
DEFAULT_READOUT_TIMES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_VELOCITY_X    = 100.0   # pixels per second


def main():
    p = argparse.ArgumentParser(description="Rolling-shutter sensitivity sweep (warped GT)")
    p.add_argument("--coco-root",      default="data/coco")
    p.add_argument("--model",          default="yolov8n.pt")
    p.add_argument("--max-images",     type=int, default=500)
    p.add_argument("--image-size",     type=int, default=640)
    p.add_argument("--image-offset",   type=int, default=0)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--bootstrap-iters", type=int, default=200)
    p.add_argument("--bootstrap-seed",  type=int, default=42)
    p.add_argument("--velocity-x",     type=float, default=DEFAULT_VELOCITY_X,
                   help="Horizontal readout velocity in px/s")
    p.add_argument("--readout-times",
                   default=",".join(str(t) for t in DEFAULT_READOUT_TIMES),
                   help="Comma-separated readout_time values")
    p.add_argument("--output-dir",     default="outputs/rolling_shutter_sweep_warped")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readout_times = [float(t) for t in args.readout_times.split(",")]

    print(f"Loading {args.max_images} COCO images ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root,
        max_images=args.max_images,
        image_size=args.image_size,
        image_offset=args.image_offset,
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    from ultralytics import YOLO
    model = YOLO(args.model)
    run_fn = lambda imgs, tgts: (
        run_yolo(model, imgs, conf=args.conf, iou=args.iou, device=args.device),
        tgts,
    )

    # Baseline: readout_time = 0 (no warp)
    print("\n=== Baseline (readout_time=0) ===")
    clean_preds = run_yolo(model, images, conf=args.conf, iou=args.iou, device=args.device)
    baseline_map50 = compute_map(
        [{**p, "image_id": iid} for p, iid in zip(clean_preds, image_ids)], targets
    )["map50"]
    print(f"  mAP@50 = {baseline_map50:.4f}")

    sweep = []
    for rt in readout_times:
        disp_px = args.velocity_x * rt
        print(f"\n=== readout_time={rt:.3f}  max_disp={disp_px:.1f}px ===")

        if rt == 0.0:
            warped_imgs = images
            warped_tgts = targets
        else:
            warped_imgs = [
                warp_image_rolling_shutter(img, args.velocity_x, rt) for img in images
            ]
            warped_tgts = warp_targets_rolling_shutter(
                targets, args.velocity_x, rt, args.image_size
            )

        preds = run_yolo(model, warped_imgs, conf=args.conf, iou=args.iou, device=args.device)
        tagged = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]
        res = compute_map_ci(tagged, warped_tgts,
                             n_bootstrap=args.bootstrap_iters, seed=args.bootstrap_seed)
        s = res["map50"] / max(baseline_map50, 1e-6)
        print(f"  mAP={res['map50']:.4f} ±{res['map50_ci']:.4f}  S={s:.4f}")
        sweep.append({
            "readout_time":       rt,
            "max_displacement_px": disp_px,
            "map50":              res["map50"],
            "map50_ci":           res["map50_ci"],
        })

    thr = None
    target_map = 0.9 * baseline_map50
    for i in range(1, len(sweep)):
        if sweep[i]["map50"] <= target_map:
            prev, curr = sweep[i - 1], sweep[i]
            frac = (target_map - prev["map50"]) / (curr["map50"] - prev["map50"] + 1e-9)
            thr = prev["max_displacement_px"] + frac * (
                curr["max_displacement_px"] - prev["max_displacement_px"]
            )
            break

    out = {
        "velocity_x":           args.velocity_x,
        "baseline_map50":       baseline_map50,
        "gt_warped":            True,
        "sweep":                sweep,
        "threshold_map_10pct_displacement_px": thr,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    if thr:
        print(f"10% mAP drop at displacement ≈ {thr:.1f} px")
    else:
        print("mAP never drops 10% in tested range — rolling shutter is benign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
