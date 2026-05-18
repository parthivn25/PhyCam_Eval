"""Threshold-free/low-confidence COCO AP sanity checks for PhyCam-Eval.

This script repeats a compact set of YOLOv8n sensitivity checks at both the
paper's within-protocol confidence threshold (0.25) and a near-threshold-free
COCO AP setting (default conf=0.001).  It uses the same deterministic COCO
val2017 slice, uniform 640x640 resize, and degradation operators as the main
sweeps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

from phycam_eval.degradations import DefocusOperator, HDRCompressionOperator, SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo
from phycam_eval.eval.metrics import compute_map, compute_map_ci


DEFAULT_BETAS = [1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.80]
DEFAULT_ALPHAS = [0.0, 1.0, 2.0, 3.0]
DEFAULT_ISOS = [100, 3200, 6400]


def _metric(preds, targets, bootstrap_iters: int, bootstrap_seed: int) -> dict:
    if bootstrap_iters > 0:
        return compute_map_ci(preds, targets, n_bootstrap=bootstrap_iters, seed=bootstrap_seed)
    out = compute_map(preds, targets)
    return {**out, "map50_ci": 0.0, "map50_95_ci": 0.0}


def _find_beta_threshold(rows: list[dict], metric_key: str) -> float | None:
    """Monotone linear interpolation around S=0.9 on rows sorted high-to-low beta."""
    sorted_rows = sorted(rows, key=lambda r: r["param"], reverse=True)
    prev = sorted_rows[0]
    for row in sorted_rows[1:]:
        if row[metric_key] <= 0.9 <= prev[metric_key]:
            x0, y0 = prev["param"], prev[metric_key]
            x1, y1 = row["param"], row[metric_key]
            if abs(y1 - y0) < 1e-12:
                return None
            return x0 + (0.9 - y0) * (x1 - x0) / (y1 - y0)
        prev = row
    return None


def _write_outputs(out_dir: Path, payload: dict, flat_rows: list[dict]) -> None:
    """Write JSON/CSV after each completed row so long runs are recoverable."""
    with (out_dir / "results.partial.json").open("w") as f:
        json.dump(payload, f, indent=2)

    with (out_dir / "results.partial.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conf",
                "operator",
                "param_name",
                "param",
                "map50",
                "map50_95",
                "map50_ci",
                "map50_95_ci",
                "S50",
                "S50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Run low-confidence COCO AP sanity checks")
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--device", default="mps")
    p.add_argument("--conf-values", default="0.25,0.001")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--betas", default=",".join(str(v) for v in DEFAULT_BETAS))
    p.add_argument("--alphas", default=",".join(str(v) for v in DEFAULT_ALPHAS))
    p.add_argument("--isos", default=",".join(str(v) for v in DEFAULT_ISOS))
    p.add_argument("--bootstrap-iters", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--noise-seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/final/threshold_free_sanity")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_values = [float(v) for v in args.conf_values.split(",")]
    betas = [float(v) for v in args.betas.split(",")]
    alphas = [float(v) for v in args.alphas.split(",")]
    isos = [int(v) for v in args.isos.split(",")]

    print(f"Loading {args.max_images} COCO val2017 images ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=args.image_size
    )
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    from ultralytics import YOLO

    model = YOLO(args.model)
    all_results: dict[str, dict] = {}
    flat_rows: list[dict] = []
    payload = {
        "config": {
            "max_images": args.max_images,
            "image_size": args.image_size,
            "device": args.device,
            "conf_values": conf_values,
            "iou": args.iou,
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "noise_seed": args.noise_seed,
            "betas": betas,
            "alphas": alphas,
            "isos": isos,
        },
        "results": all_results,
    }

    for conf in conf_values:
        print(f"\n=== Confidence threshold {conf:g} ===")

        def infer(imgs, desc: str):
            from tqdm import tqdm

            preds = []
            t0 = time.perf_counter()
            for img, iid in tqdm(
                zip(imgs, image_ids),
                total=len(image_ids),
                desc=desc,
                unit="img",
                dynamic_ncols=True,
            ):
                pred = run_yolo(
                    model,
                    [img],
                    conf=conf,
                    iou=args.iou,
                    device=args.device,
                )[0]
                preds.append({**pred, "image_id": iid})
            elapsed = time.perf_counter() - t0
            print(
                f"  {desc}: {elapsed/60:.1f} min "
                f"({elapsed/max(len(image_ids), 1):.2f} s/img)"
            )
            return preds

        clean_preds = infer(images, f"conf={conf:g} clean")
        baseline = _metric(clean_preds, targets, args.bootstrap_iters, args.bootstrap_seed)
        base50 = baseline["map50"]
        base5095 = baseline["map50_95"]
        print(f"clean: mAP@50={base50:.4f}, mAP@[.50:.95]={base5095:.4f}")

        conf_key = f"{conf:g}"
        conf_results = {
            "conf": conf,
            "iou": args.iou,
            "baseline": baseline,
            "hdr": [],
            "defocus": [],
            "noise": [],
        }
        all_results[conf_key] = conf_results
        _write_outputs(out_dir, payload, flat_rows)

        for beta in betas:
            print(f"HDR beta={beta:.2f}")
            op = HDRCompressionOperator(beta=beta)
            degraded = [op(img) for img in images]
            res = _metric(
                infer(degraded, f"conf={conf:g} Q_beta beta={beta:.2f}"),
                targets,
                args.bootstrap_iters,
                args.bootstrap_seed,
            )
            del degraded
            gc.collect()
            row = {
                "operator": "Q_beta",
                "param_name": "beta",
                "param": beta,
                "map50": res["map50"],
                "map50_95": res["map50_95"],
                "map50_ci": res.get("map50_ci", 0.0),
                "map50_95_ci": res.get("map50_95_ci", 0.0),
                "S50": res["map50"] / max(base50, 1e-12),
                "S50_95": res["map50_95"] / max(base5095, 1e-12),
            }
            conf_results["hdr"].append(row)
            flat_rows.append({"conf": conf, **row})
            print(f"  S50={row['S50']:.3f}, S50_95={row['S50_95']:.3f}")
            _write_outputs(out_dir, payload, flat_rows)

        for alpha in alphas:
            print(f"A_phi alpha={alpha:.1f}")
            op = DefocusOperator(alpha=alpha)
            degraded = [op(img) for img in images]
            res = _metric(
                infer(degraded, f"conf={conf:g} A_phi alpha={alpha:.1f}"),
                targets,
                args.bootstrap_iters,
                args.bootstrap_seed,
            )
            del degraded
            gc.collect()
            row = {
                "operator": "A_phi",
                "param_name": "alpha",
                "param": alpha,
                "map50": res["map50"],
                "map50_95": res["map50_95"],
                "map50_ci": res.get("map50_ci", 0.0),
                "map50_95_ci": res.get("map50_95_ci", 0.0),
                "S50": res["map50"] / max(base50, 1e-12),
                "S50_95": res["map50_95"] / max(base5095, 1e-12),
            }
            conf_results["defocus"].append(row)
            flat_rows.append({"conf": conf, **row})
            print(f"  S50={row['S50']:.3f}, S50_95={row['S50_95']:.3f}")
            _write_outputs(out_dir, payload, flat_rows)

        for iso in isos:
            print(f"Noise ISO={iso}")
            op = SensorNoiseOperator.from_iso(iso, seed=args.noise_seed)
            degraded = [op(img) for img in images]
            res = _metric(
                infer(degraded, f"conf={conf:g} N_sigma ISO={iso}"),
                targets,
                args.bootstrap_iters,
                args.bootstrap_seed,
            )
            del degraded
            gc.collect()
            row = {
                "operator": "N_sigma",
                "param_name": "ISO",
                "param": iso,
                "map50": res["map50"],
                "map50_95": res["map50_95"],
                "map50_ci": res.get("map50_ci", 0.0),
                "map50_95_ci": res.get("map50_95_ci", 0.0),
                "S50": res["map50"] / max(base50, 1e-12),
                "S50_95": res["map50_95"] / max(base5095, 1e-12),
            }
            conf_results["noise"].append(row)
            flat_rows.append({"conf": conf, **row})
            print(f"  S50={row['S50']:.3f}, S50_95={row['S50_95']:.3f}")
            _write_outputs(out_dir, payload, flat_rows)

        conf_results["beta_threshold_10pct_S50"] = _find_beta_threshold(
            conf_results["hdr"], "S50"
        )
        conf_results["beta_threshold_10pct_S50_95"] = _find_beta_threshold(
            conf_results["hdr"], "S50_95"
        )
    with (out_dir / "results.json").open("w") as f:
        json.dump(payload, f, indent=2)

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conf",
                "operator",
                "param_name",
                "param",
                "map50",
                "map50_95",
                "map50_ci",
                "map50_95_ci",
                "S50",
                "S50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"\nSaved {out_dir / 'results.json'}")
    print(f"Saved {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
