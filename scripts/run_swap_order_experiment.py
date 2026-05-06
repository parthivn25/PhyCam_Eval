"""
Pipeline ordering sensitivity experiment.

The super-additivity interaction coefficient gamma may depend on the order of
operations in the ISP pipeline. This script runs a chained sweep with reversed
operator ordering: Q_beta( N_sigma( I ) ) instead of N_sigma( Q_beta( I ) ),
to measure how the interaction structure changes under different ISP pipelines.

Output compares gamma values across orderings on the same image subset and
with the same marginal estimates, so the ordering comparison is matched.

Usage:
    PYTHONPATH=. python3 scripts/run_swap_order_experiment.py \
        --coco-root data/coco --max-images 500 --device mps \
        --bootstrap-iters 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phycam_eval.degradations import HDRCompressionOperator, SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo
from phycam_eval.eval.metrics import compute_map_ci


BETAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
ISOS = [100, 400, 1600, 6400]

# Paper diagnostic grid: full grid through beta=0.7 plus extra severe cells.
PARTIAL_GRID = {
    0.6: [100, 6400],
    0.5: [6400],
}

# ISO calibration constants (must match run_chained_sweep.py)
BASE_ISO = 100
BASE_READ_NOISE = 0.002
BASE_GAIN = 5e-5


def _noise_op(iso):
    return SensorNoiseOperator.from_iso(
        iso, base_iso=BASE_ISO, base_read_noise=BASE_READ_NOISE,
        base_gain=BASE_GAIN, seed=42,
    )


def _progress(current, total, label=""):
    pct = 100 * current / total
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% {label:<30}", end="", flush=True)
    if current == total:
        print()


def _tag(preds, image_ids):
    """Attach image_id to each prediction dict (required by compute_map_ci)."""
    return [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]


def _experiment_cells(full_grid: bool = False):
    """Return ordered (beta, ISO) cells for the diagnostic run."""
    cells = []
    for beta in BETAS:
        iso_list = ISOS if full_grid else PARTIAL_GRID.get(beta, ISOS)
        for iso in iso_list:
            cells.append((beta, iso))
    return cells


def _run_chained_cells(
    *,
    order: str,
    cells,
    images,
    image_ids,
    targets,
    model,
    device,
    bootstrap_iters,
    clean_map,
    s_beta,
    s_iso,
):
    if order == "original":
        label = "ORIGINAL-ORDER chained grid: noise( HDR( I ) )"
    elif order == "swapped":
        label = "REVERSED-ORDER chained grid: HDR( noise( I ) )"
    else:
        raise ValueError(f"Unknown order: {order}")

    print(f"\nRunning {label}...")
    results = []
    for k, (beta, iso) in enumerate(cells):
        _progress(k, len(cells), f"beta={beta} ISO={iso}")
        noise_op = _noise_op(iso)
        hdr_op = HDRCompressionOperator(beta=beta)
        if order == "original":
            deg_images = [noise_op(hdr_op(img)) for img in images]
        else:
            deg_images = [hdr_op(noise_op(img)) for img in images]
        preds = run_yolo(model, deg_images, device=device)
        r = compute_map_ci(_tag(preds, image_ids), targets, n_bootstrap=bootstrap_iters)
        m, ci = r["map50"], r["map50_ci"]
        s_meas = m / clean_map
        s_pred = s_beta[beta] * s_iso[iso]
        delta_s = s_meas - s_pred
        gamma = s_meas / s_pred if s_pred > 0 else float("nan")
        results.append({
            "beta": beta,
            "iso": iso,
            "map50": m,
            "map50_ci": ci,
            "s_meas": round(s_meas, 4),
            "s_pred": round(s_pred, 4),
            "delta_s": round(delta_s, 4),
            "gamma": round(gamma, 4),
        })
    _progress(len(cells), len(cells), "done")
    return results


def run_sweep(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    IMAGE_SIZE = 640

    print("Loading COCO images...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=IMAGE_SIZE
    )
    targets = build_coco_targets(coco_data, image_ids, metas, IMAGE_SIZE)

    print("Loading YOLOv8n model...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    print("\nRunning clean baseline...")
    clean_preds = run_yolo(model, images, device=args.device)
    res = compute_map_ci(_tag(clean_preds, image_ids), targets,
                         n_bootstrap=args.bootstrap_iters)
    clean_map, clean_ci = res["map50"], res["map50_ci"]
    print(f"  Clean: mAP@50 = {clean_map:.4f} ± {clean_ci:.4f}")

    # Individual marginal sensitivities.
    print("\nRunning HDR marginals (Q_beta only)...")
    s_beta = {}
    for i, beta in enumerate(BETAS):
        _progress(i, len(BETAS), f"β={beta}")
        if beta == 1.0:
            s_beta[beta] = 1.0
            continue
        op = HDRCompressionOperator(beta=beta)
        deg_images = [op(img) for img in images]
        preds = run_yolo(model, deg_images, device=args.device)
        r = compute_map_ci(_tag(preds, image_ids), targets, n_bootstrap=50)
        s_beta[beta] = r["map50"] / clean_map
    _progress(len(BETAS), len(BETAS), "done")

    print("\nRunning noise marginals (N_sigma only)...")
    s_iso = {}
    for i, iso in enumerate(ISOS):
        _progress(i, len(ISOS), f"ISO={iso}")
        op = _noise_op(iso)
        deg_images = [op(img) for img in images]
        preds = run_yolo(model, deg_images, device=args.device)
        r = compute_map_ci(_tag(preds, image_ids), targets, n_bootstrap=50)
        s_iso[iso] = r["map50"] / clean_map
    _progress(len(ISOS), len(ISOS), "done")

    cells = _experiment_cells(full_grid=args.full_grid)
    print(f"\nOrdering diagnostic cells: {len(cells)}")

    original_results = []
    if args.compare_original_order:
        original_results = _run_chained_cells(
            order="original",
            cells=cells,
            images=images,
            image_ids=image_ids,
            targets=targets,
            model=model,
            device=args.device,
            bootstrap_iters=args.bootstrap_iters,
            clean_map=clean_map,
            s_beta=s_beta,
            s_iso=s_iso,
        )

    swapped_results = _run_chained_cells(
        order="swapped",
        cells=cells,
        images=images,
        image_ids=image_ids,
        targets=targets,
        model=model,
        device=args.device,
        bootstrap_iters=args.bootstrap_iters,
        clean_map=clean_map,
        s_beta=s_beta,
        s_iso=s_iso,
    )

    comparison = []
    if original_results:
        orig_map = {(r["beta"], r["iso"]): r for r in original_results}
        for r in swapped_results:
            key = (r["beta"], r["iso"])
            if key in orig_map:
                orig = orig_map[key]
                comparison.append({
                    "beta": r["beta"],
                    "iso": r["iso"],
                    "delta_s_original": orig["delta_s"],
                    "delta_s_swapped": r["delta_s"],
                    "gamma_original": orig["gamma"],
                    "gamma_swapped": r["gamma"],
                    "delta_gamma": round(r["gamma"] - orig["gamma"], 4),
                })

    output = {
        "config": {
            "ordering": "Q_beta( N_sigma( I ) )  [HDR after noise]",
            "original_ordering": "N_sigma( Q_beta( I ) )  [noise after HDR]",
            "coco_root": args.coco_root,
            "max_images": args.max_images,
            "bootstrap_iters": args.bootstrap_iters,
            "full_grid": args.full_grid,
            "compare_original_order": args.compare_original_order,
        },
        "clean_baseline": {"map50": clean_map, "map50_ci": clean_ci},
        "marginals": {
            "s_beta": {str(k): v for k, v in s_beta.items()},
            "s_iso": {str(k): v for k, v in s_iso.items()},
        },
        "cells": [{"beta": b, "iso": iso} for b, iso in cells],
        "original_order_results": original_results,
        "swapped_order_results": swapped_results,
        "ordering_comparison": comparison,
    }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("ORDERING COMPARISON SUMMARY")
    print(f"{'='*60}")
    if comparison:
        print(
            f"{'beta':>5} {'ISO':>6} {'dS_orig':>10} {'dS_swap':>10} "
            f"{'g_orig':>8} {'g_swap':>8} {'dg':>8}"
        )
        print("-" * 60)
        for c in comparison:
            print(
                f"{c['beta']:>5} {c['iso']:>6} "
                f"{c['delta_s_original']:>10.4f} {c['delta_s_swapped']:>10.4f} "
                f"{c['gamma_original']:>8.4f} {c['gamma_swapped']:>8.4f} "
                f"{c['delta_gamma']:>8.4f}"
            )
    else:
        print("Original-order comparison skipped; only reversed-order results were saved.")

    key_cells = [(0.8, 6400), (0.7, 6400), (0.9, 6400)]
    print("\nKey cells (as reported in paper):")
    if comparison:
        for c in comparison:
            if (c["beta"], c["iso"]) in key_cells:
                direction = "swap~same" if abs(c["delta_gamma"]) < 0.05 else "ORDERING MATTERS"
                print(
                    f"  beta={c['beta']}, ISO={c['iso']}: "
                    f"gamma_orig={c['gamma_original']:.3f} -> gamma_swap={c['gamma_swapped']:.3f} "
                    f"(delta_gamma={c['delta_gamma']:+.3f}) [{direction}]"
                )

    print(f"\nResults saved to: {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Pipeline ordering sensitivity experiment")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="outputs/swap_order_experiment")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Evaluate the complete 6x4 beta/ISO grid instead of the paper diagnostic grid.",
    )
    parser.add_argument(
        "--no-compare-original-order",
        dest="compare_original_order",
        action="store_false",
        help="Skip same-sample original-order cells and save reversed-order results only.",
    )
    parser.set_defaults(compare_original_order=True)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
