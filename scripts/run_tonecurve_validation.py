"""
scripts/run_tonecurve_validation.py — Out-of-sample tone-curve proxy validation.

Calibrates a Q_beta spectral proxy against 8 synthetic tone curves, then tests
out-of-sample on a held-out image split.  Validates that beta_eff tracks
tone-curve mAP ordering (not that it matches numerical values exactly).

Protocol:
  - Calib: first --calib-images from the sorted image subset
  - Test:  next --test-images (OOS) from the same sorted ID sequence
  - For each tone curve: fit beta_eff by minimizing |mAP_Q(calib) - mAP_tc(calib)|
  - Report: Pearson r between mAP_Q(beta_eff) and mAP_tc on test split, n within CI

Tone curves implemented:
  gamma_1.8, gamma_2.2, gamma_2.6, sRGB, sigmoid_c3, sigmoid_c5, sigmoid_c8, filmic

Usage:
    PYTHONPATH=. python3 scripts/run_tonecurve_validation.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --output-dir outputs/hdr_tonecurve_validation_oos
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from phycam_eval.degradations import HDRCompressionOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo
from phycam_eval.eval.metrics import compute_map, compute_map_ci


# ── Tone curve definitions ─────────────────────────────────────────────────────

def _gamma(x: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(x.clip(0, 1), 1.0 / gamma)


def _srgb(x: np.ndarray) -> np.ndarray:
    out = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x.clip(1e-8), 1 / 2.4) - 0.055)
    return out.clip(0, 1)


def _sigmoid(x: np.ndarray, contrast: float) -> np.ndarray:
    mid = 0.5
    raw = 1.0 / (1.0 + np.exp(-contrast * (x - mid)))
    lo = 1.0 / (1.0 + np.exp(-contrast * (0.0 - mid)))
    hi = 1.0 / (1.0 + np.exp(-contrast * (1.0 - mid)))
    return ((raw - lo) / (hi - lo + 1e-8)).clip(0, 1)


def _filmic(x: np.ndarray) -> np.ndarray:
    # Simple filmic: highlight rolloff similar to ACES RRT
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = x.clip(0, 1)
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clip(0, 1)


TONE_CURVES = {
    "gamma_1.8":  lambda x: _gamma(x, 1.8),
    "gamma_2.2":  lambda x: _gamma(x, 2.2),
    "gamma_2.6":  lambda x: _gamma(x, 2.6),
    "sRGB":       _srgb,
    "sigmoid_c3": lambda x: _sigmoid(x, 3.0),
    "sigmoid_c5": lambda x: _sigmoid(x, 5.0),
    "sigmoid_c8": lambda x: _sigmoid(x, 8.0),
    "filmic":     _filmic,
}


def _apply_tone_curve(images: list[np.ndarray], fn) -> list[np.ndarray]:
    return [fn(img.copy()) for img in images]


def _cleanup_runtime(device: str) -> None:
    """Release torch/Ultralytics runtime caches between long repeated passes."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _plot_results(out_dir: Path, curve_results: list[dict], r: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc_map50s = [cr["map50_tonecurve"] for cr in curve_results]
    odrc_map50s = [cr["map50_odrc"] for cr in curve_results]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tc_map50s, odrc_map50s, zorder=3)
    for cr in curve_results:
        ax.annotate(cr["curve_name"], (cr["map50_tonecurve"], cr["map50_odrc"]),
                    fontsize=7, ha="left", va="bottom")
    lims = [min(tc_map50s + odrc_map50s) * 0.97, max(tc_map50s + odrc_map50s) * 1.02]
    ax.plot(lims, lims, "k--", lw=0.8, label="y=x")
    ax.set_xlabel("Tone-curve mAP@50 (OOS)")
    ax.set_ylabel("Q_beta proxy mAP@50 (OOS)")
    ax.set_title(f"OOS tone-curve proxy validation (r={r:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_tonecurve_oos.png", dpi=150)
    fig.savefig(out_dir / "figure_tonecurve_oos.pdf")
    plt.close(fig)


# ── Beta fitting ───────────────────────────────────────────────────────────────

BETA_GRID = np.arange(0.50, 1.01, 0.05).tolist()


def _fit_beta(calib_imgs, image_ids, targets, run_fn, target_map50: float, device: str) -> float:
    """Find the beta that produces the closest mAP to target_map50 on calib set."""
    best_beta, best_dist = 1.0, float("inf")
    for beta in BETA_GRID:
        op = HDRCompressionOperator(beta=beta)
        degraded = [op(img) for img in calib_imgs]
        preds = run_fn(degraded)
        tagged = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]
        m = compute_map(tagged, targets)["map50"]
        dist = abs(m - target_map50)
        if dist < best_dist:
            best_dist = dist
            best_beta = beta
        del degraded, preds, tagged
        _cleanup_runtime(device)
    return best_beta


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Tone-curve proxy out-of-sample validation")
    p.add_argument("--coco-root",       default="data/coco")
    p.add_argument("--model",           default="yolov8n.pt")
    p.add_argument("--image-size",      type=int, default=640)
    p.add_argument("--image-offset",    type=int, default=0)
    p.add_argument("--max-images",      type=int, default=500,
                   help="Total images (calib + test). calib=max//5, test=max-calib.")
    p.add_argument("--calib-images",    type=int, default=None,
                   help="Override calib split (default: max_images // 5)")
    p.add_argument("--test-images",     type=int, default=None,
                   help="Override test split (default: max_images - calib_images)")
    p.add_argument("--bootstrap-iters", type=int, default=100)
    p.add_argument("--bootstrap-seed",  type=int, default=42)
    p.add_argument("--device",          default="cpu")
    p.add_argument("--conf",            type=float, default=0.25)
    p.add_argument("--output-dir",      default="outputs/hdr_tonecurve_validation_oos")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"

    calib_images = args.calib_images if args.calib_images is not None else args.max_images // 5
    test_images  = args.test_images  if args.test_images  is not None else args.max_images - calib_images
    args.calib_images = calib_images
    args.test_images  = test_images
    total_images = calib_images + test_images

    print(f"Loading {total_images} images (calib={args.calib_images}, test={args.test_images}) ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root,
        max_images=total_images,
        image_size=args.image_size,
        image_offset=args.image_offset,
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    from ultralytics import YOLO
    yolo = YOLO(args.model)
    run_fn = lambda imgs: run_yolo(yolo, imgs, conf=args.conf, device=args.device)

    calib_imgs    = images[:args.calib_images]
    calib_ids     = image_ids[:args.calib_images]
    calib_targets = targets[:args.calib_images]
    test_imgs     = images[args.calib_images:]
    test_ids      = image_ids[args.calib_images:]
    test_targets  = targets[args.calib_images:]

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        print(f"Resuming from {checkpoint_path}")

    if checkpoint is not None and "clean_baseline_test" in checkpoint:
        clean_baseline = checkpoint["clean_baseline_test"]["map50"]
        clean_ci = checkpoint["clean_baseline_test"]["map50_ci"]
        print("\n=== Clean baseline (test split) ===")
        print(f"  mAP@50 = {clean_baseline:.4f} ±{clean_ci:.4f}  [checkpoint]")
    else:
        # Clean baseline on test split
        print("\n=== Clean baseline (test split) ===")
        clean_preds = run_fn(test_imgs)
        tagged_clean = [{**p, "image_id": iid} for p, iid in zip(clean_preds, test_ids)]
        res_clean = compute_map_ci(
            tagged_clean, test_targets,
            n_bootstrap=args.bootstrap_iters,
            seed=args.bootstrap_seed,
        )
        clean_baseline = res_clean["map50"]
        clean_ci = res_clean["map50_ci"]
        print(f"  mAP@50 = {clean_baseline:.4f} ±{clean_ci:.4f}")
        checkpoint = {
            "config": {
                "coco_root":       args.coco_root,
                "max_images":      total_images,
                "calib_images":    args.calib_images,
                "test_images":     args.test_images,
                "bootstrap_iters": args.bootstrap_iters,
                "bootstrap_seed":  args.bootstrap_seed,
                "oos":             True,
            },
            "clean_baseline_test": {"map50": clean_baseline, "map50_ci": clean_ci},
            "results": [],
            "partials": {},
        }
        _write_json(checkpoint_path, checkpoint)
        _cleanup_runtime(args.device)

    curve_results = list(checkpoint.get("results", [])) if checkpoint else []
    completed_curves = {cr["curve_name"] for cr in curve_results}
    partials = checkpoint.setdefault("partials", {})

    for curve_name, tc_fn in TONE_CURVES.items():
        if curve_name in completed_curves:
            print(f"\n── {curve_name} ────────────────────────────────────────")
            print("  already complete [checkpoint]")
            continue

        print(f"\n── {curve_name} ────────────────────────────────────────")
        partial = partials.setdefault(curve_name, {})

        # Calib: get tone-curve mAP on calib split
        if "beta_eff" in partial and "map50_calib_tc" in partial:
            beta_eff = float(partial["beta_eff"])
            map50_calib_tc = float(partial["map50_calib_tc"])
            print(f"  beta_eff = {beta_eff:.3f}  (calib mAP_tc={map50_calib_tc:.4f}) [checkpoint]")
        else:
            calib_tc = _apply_tone_curve(calib_imgs, tc_fn)
            preds_calib_tc = run_fn(calib_tc)
            tagged_calib = [{**p, "image_id": iid} for p, iid in zip(preds_calib_tc, calib_ids)]
            map50_calib_tc = compute_map(tagged_calib, calib_targets)["map50"]
            del calib_tc, preds_calib_tc, tagged_calib
            _cleanup_runtime(args.device)

            # Fit beta_eff
            beta_eff = _fit_beta(
                calib_imgs, calib_ids, calib_targets, run_fn, map50_calib_tc, args.device
            )
            print(f"  beta_eff = {beta_eff:.3f}  (calib mAP_tc={map50_calib_tc:.4f})")
            partial.update({
                "beta_eff": float(beta_eff),
                "map50_calib_tc": float(map50_calib_tc),
            })
            _write_json(checkpoint_path, checkpoint)
            print(f"  Partial checkpoint saved to {checkpoint_path}")
            _cleanup_runtime(args.device)

        # Test OOS: tone curve
        if "map50_tonecurve" in partial and "map50_tonecurve_ci" in partial:
            map50_tc = float(partial["map50_tonecurve"])
            ci_tc = float(partial["map50_tonecurve_ci"])
            print(f"  test mAP_tc={map50_tc:.4f} ±{ci_tc:.4f} [checkpoint]")
        else:
            test_tc = _apply_tone_curve(test_imgs, tc_fn)
            preds_test_tc = run_fn(test_tc)
            tagged_test_tc = [{**p, "image_id": iid} for p, iid in zip(preds_test_tc, test_ids)]
            res_tc = compute_map_ci(
                tagged_test_tc, test_targets,
                n_bootstrap=args.bootstrap_iters,
                seed=args.bootstrap_seed,
            )
            map50_tc = res_tc["map50"]
            ci_tc = res_tc["map50_ci"]
            partial.update({
                "map50_tonecurve": float(map50_tc),
                "map50_tonecurve_ci": float(ci_tc),
            })
            _write_json(checkpoint_path, checkpoint)
            print(f"  Partial checkpoint saved to {checkpoint_path}")
            del test_tc, preds_test_tc, tagged_test_tc, res_tc
            _cleanup_runtime(args.device)

        # Test OOS: Q_beta proxy
        op = HDRCompressionOperator(beta=beta_eff)
        test_odrc = [op(img) for img in test_imgs]
        preds_test_odrc = run_fn(test_odrc)
        tagged_test_odrc = [{**p, "image_id": iid} for p, iid in zip(preds_test_odrc, test_ids)]
        res_odrc = compute_map_ci(
            tagged_test_odrc, test_targets,
            n_bootstrap=args.bootstrap_iters,
            seed=args.bootstrap_seed,
        )
        map50_odrc = res_odrc["map50"]
        ci_odrc = res_odrc["map50_ci"]
        del test_odrc, preds_test_odrc, tagged_test_odrc

        abs_delta = abs(map50_odrc - map50_tc)
        within_ci = abs_delta <= ci_tc

        print(f"  test mAP_tc={map50_tc:.4f} ±{ci_tc:.4f}  "
              f"mAP_odrc={map50_odrc:.4f}  delta={abs_delta:.4f}  within_CI={within_ci}")

        curve_results.append({
            "curve_name":          curve_name,
            "beta_eff":            float(beta_eff),
            "calib_n":             args.calib_images,
            "test_n":              args.test_images,
            "map50_tonecurve":     float(map50_tc),
            "map50_tonecurve_ci":  float(ci_tc),
            "map50_odrc":          float(map50_odrc),
            "map50_odrc_ci":       float(ci_odrc),
            "abs_delta":           float(abs_delta),
            "within_ci":           bool(within_ci),
        })
        checkpoint["results"] = curve_results
        partials.pop(curve_name, None)
        _write_json(checkpoint_path, checkpoint)
        print(f"  Checkpoint saved to {checkpoint_path}")
        del res_odrc
        _cleanup_runtime(args.device)

    # Pearson r between tone-curve mAP and ODRC proxy mAP (OOS)
    tc_map50s = [cr["map50_tonecurve"] for cr in curve_results]
    odrc_map50s = [cr["map50_odrc"] for cr in curve_results]
    r = float(np.corrcoef(tc_map50s, odrc_map50s)[0, 1])
    n_within = sum(cr["within_ci"] for cr in curve_results)
    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Pearson r (OOS): {r:.4f}")
    print(f"  Within CI: {n_within}/{len(curve_results)}")

    _plot_results(out_dir, curve_results, r)
    print(f"  Figure saved.")

    out = {
        "config": {
            "coco_root":       args.coco_root,
            "max_images":      total_images,
            "calib_images":    args.calib_images,
            "test_images":     args.test_images,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "oos":             True,
        },
        "clean_baseline_test": {"map50": clean_baseline, "map50_ci": clean_ci},
        "pearson_r_oos":       r,
        "n_within_ci":         n_within,
        "results":             curve_results,
    }
    out_path = out_dir / "results.json"
    _write_json(out_path, out)
    print(f"Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
