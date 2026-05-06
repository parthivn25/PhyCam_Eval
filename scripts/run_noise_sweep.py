"""
scripts/run_noise_sweep.py — end-to-end sensor noise sensitivity sweep.

Runs YOLO over degraded COCO images across a range of ISO gain values
and generates a sensitivity landscape (mAP + MTF50 vs ISO).

Physical model:
    N(I) = Poisson(I/g)·g + Normal(0, σ_r²)
    where g = (ISO / base_ISO) × base_gain
          σ_r = read noise std (fixed at base_read_noise)

Usage:
    python3 scripts/run_noise_sweep.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --max-images 500 \\
        --output-dir outputs/noise_sweep

Outputs:
    outputs/noise_sweep/
        results.json          — raw mAP + MTF50 for each ISO
        figure_noise_sensitivity.png
"""

import argparse
import json
import time
from pathlib import Path

from phycam_eval.degradations import SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo, run_fasterrcnn
from phycam_eval.eval.mtf import make_slanted_edge_chart, measure_mtf, mtf50
from phycam_eval.eval.sensitivity import SensitivitySweep


# ISO values to sweep (doubling each step = +3 dB SNR per step)
DEFAULT_ISO_VALUES = [100, 200, 400, 800, 1600, 3200, 6400]

BASE_ISO = 100
BASE_READ_NOISE = 0.002
BASE_GAIN = 5e-5


def _load_detector(args):
    """Return (model, run_fn, tag) for the chosen detector."""
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
    p = argparse.ArgumentParser(description="Sensor noise sensitivity sweep")
    p.add_argument("--coco-root",      default="data/coco",   help="COCO data root")
    p.add_argument("--model",          default="yolov8n.pt",  help="YOLO weights")
    p.add_argument("--detector",       default="yolo", choices=["yolo", "fasterrcnn"],
                   help="Detector backend (default: yolo)")
    p.add_argument("--max-images",     type=int, default=500)
    p.add_argument("--image-offset",   type=int, default=0,
                   help="Skip first N sorted COCO image IDs (use 500 for second slice)")
    p.add_argument("--image-size",     type=int, default=640)
    p.add_argument("--output-dir",     default="outputs/noise_sweep")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--base-iso",       type=int, default=BASE_ISO)
    p.add_argument("--base-read-noise", type=float, default=BASE_READ_NOISE)
    p.add_argument("--base-gain",      type=float, default=BASE_GAIN)
    p.add_argument("--iso-values",
                   default=",".join(str(v) for v in DEFAULT_ISO_VALUES),
                   help="Comma-separated ISO values to sweep")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iso_values = [int(v) for v in args.iso_values.split(",")]

    print(f"Loading COCO val images from {args.coco_root} ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=args.image_size,
        image_offset=args.image_offset,
    )

    from phycam_eval.eval.metrics import compute_map, compute_map_ci
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    print(f"Loading detector: {args.detector}")
    try:
        model, run_fn, det_tag = _load_detector(args)
    except ImportError as e:
        print(f"Detector load failed ({e}) — skipping mAP; reporting MTF/SNR only.")
        model, run_fn, det_tag = None, None, args.detector

    # Synthetic chart for MTF measurement.
    mtf_chart, mtf_roi = make_slanted_edge_chart()

    # Baseline uses clean images so sensitivity is measured relative to undegraded data.
    print(f"\n=== Baseline (ISO={args.base_iso}, clean) ===")
    baseline_map50 = 0.0
    if run_fn is not None:
        clean_preds = run_fn(images)
        map_res = compute_map(
            [{**p, "image_id": iid} for p, iid in zip(clean_preds, image_ids)],
            targets)
        baseline_map50 = map_res["map50"]
        print(f"  mAP@50 = {baseline_map50:.4f}")

    baseline_mtf50 = 0.0
    try:
        freqs, mtf_vals = measure_mtf(mtf_chart, roi=mtf_roi)
        baseline_mtf50 = mtf50(freqs, mtf_vals)
        print(f"  MTF50  = {baseline_mtf50:.4f} cy/px (synthetic chart)")
    except Exception as e:
        print(f"  MTF50 failed on baseline: {e}")

    sweep = SensitivitySweep(
        param_name="ISO (gain = ISO/base_ISO)",
        param_values=[float(v) for v in iso_values],
        baseline_map=max(baseline_map50, 1e-6),
        baseline_mtf=max(baseline_mtf50, 1e-6),
    )
    all_data = []

    for iso in iso_values:
        op = SensorNoiseOperator.from_iso(
            iso,
            base_iso=args.base_iso,
            base_read_noise=args.base_read_noise,
            base_gain=args.base_gain,
            seed=42,
        )
        print(f"\n=== ISO = {iso} (gain = {op.gain:.2e}) ===")
        snr = op.snr_db(signal=0.5)
        print(f"  SNR @ 0.5 = {snr:.1f} dB")

        t0 = time.perf_counter()
        degraded = [op(img) for img in images]
        t_degrade = time.perf_counter() - t0
        print(f"  Degradation: {t_degrade*1000:.0f} ms  "
              f"({t_degrade/len(images)*1000:.2f} ms/img)")

        map50_val = 0.0
        map50_ci = 0.0
        if run_fn is not None:
            preds = run_fn(degraded)
            tagged = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]
            map_res = compute_map_ci(tagged, targets)
            map50_val = map_res["map50"]
            map50_ci = map_res["map50_ci"]
            print(f"  mAP@50 = {map50_val:.4f} ±{map50_ci:.4f}  "
                  f"(S = {map50_val/max(baseline_map50,1e-6):.3f})")

        # MTF50 on synthetic chart — noise increases ESF variance, measurably
        # lowering the apparent MTF50 as ISO rises.
        mtf50_val = 0.0
        try:
            deg_chart = op(mtf_chart)
            freqs, mtf_vals = measure_mtf(deg_chart, roi=mtf_roi)
            mtf50_val = mtf50(freqs, mtf_vals)
            print(f"  MTF50  = {mtf50_val:.4f} cy/px  "
                  f"(S = {mtf50_val/max(baseline_mtf50,1e-6):.3f})")
        except Exception as e:
            print(f"  MTF50 failed: {e}")

        sweep.add(float(iso), map50=map50_val, mtf50=mtf50_val, map50_ci=map50_ci)
        all_data.append({
            "iso": iso,
            "gain": op.gain,
            "snr_db": snr,
            "map50": map50_val,
            "map50_ci": map50_ci,
            "mtf50": mtf50_val,
        })

    thr_map = sweep.find_threshold_param("map50", 0.10)
    thr_mtf = sweep.find_threshold_param("mtf50", 0.10)
    print(f"\n10% mAP drop at ISO ≈ {thr_map:.0f}" if thr_map else "\nmAP never drops 10%")
    print(f"10% MTF drop at ISO ≈ {thr_mtf:.0f}" if thr_mtf else "MTF never drops 10%")

    suffix = f"_{det_tag}" if det_tag != "yolo" else ""
    results_path = out_dir / f"results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump({
            "detector": det_tag,
            "base_iso": args.base_iso,
            "base_read_noise": args.base_read_noise,
            "base_gain": args.base_gain,
            "baseline_map50": baseline_map50,
            "baseline_mtf50": baseline_mtf50,
            "sweep": all_data,
            "threshold_map_10pct_iso": thr_map,
            "threshold_mtf_10pct_iso": thr_mtf,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    sweep.plot(
        save_path=str(out_dir / f"figure_noise_sensitivity{suffix}.png"),
        title=f"Sensitivity Landscape: Sensor Noise — {det_tag}",
        close=True,
    )
    print(f"Figure saved to {out_dir}/figure_noise_sensitivity{suffix}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
