"""
scripts/run_incoherent_defocus_sweep.py — Incoherent-OTF defocus sweep.

Tests a physically realistic incoherent defocus model: the OTF is the
Fourier transform of the squared-magnitude of the coherent PSF, which
attenuates amplitude (unlike the phase-only A_phi operator).

Physical model:
    h_coh(x)   = F^{-1}{ exp(j alpha rho^2) }     coherent PSF
    h_inc(x)   = |h_coh(x)|^2 / ||h_coh||_1        incoherent PSF (normalized)
    H_inc(f)   = F{ h_inc }                         incoherent OTF
    I_deg[c]   = Re{ F^{-1}{ F{I[c]} . H_inc } }   degraded image

At alpha=0: H_inc = 1 (identity).  At alpha=3: <|H_inc|> approx 0.667.

Usage:
    PYTHONPATH=. python3 scripts/run_incoherent_defocus_sweep.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --output-dir outputs/incoherent_defocus_sweep

Outputs:
    outputs/incoherent_defocus_sweep/
        results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo
from phycam_eval.eval.metrics import compute_map, compute_map_ci


DEFAULT_ALPHAS = [0.0, 1.0, 2.0, 3.0]


def _build_incoherent_otf(H: int, W: int, alpha: float) -> np.ndarray:
    """Return complex incoherent OTF for the given image size and alpha."""
    fy = np.fft.fftfreq(H)[:, np.newaxis] * np.ones((1, W))
    fx = np.fft.fftfreq(W)[np.newaxis, :] * np.ones((H, 1))
    rho = np.sqrt(fy**2 + fx**2)
    rho_norm = rho / max(rho.max(), 1e-10)

    # Coherent OTF = exp(j alpha rho^2)  (same normalization as DefocusOperator)
    H_coh = np.exp(1j * alpha * rho_norm**2)

    # Coherent PSF
    h_coh = np.fft.ifft2(H_coh)

    # Incoherent PSF = |h_coh|^2 (normalized)
    h_inc = np.abs(h_coh) ** 2
    h_inc /= h_inc.sum() + 1e-10

    # Incoherent OTF
    H_inc = np.fft.fft2(h_inc)
    return H_inc


def apply_incoherent_defocus(image: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Apply incoherent defocus to (C, H, W) float32 image.

    Returns (degraded, otf_mean_abs).
    """
    if alpha == 0.0:
        return image.copy(), 1.0

    C, H, W = image.shape
    H_inc = _build_incoherent_otf(H, W, alpha)
    otf_mean_abs = float(np.mean(np.abs(H_inc)))

    out = np.empty_like(image)
    for c in range(C):
        F = np.fft.fft2(image[c])
        out[c] = np.real(np.fft.ifft2(F * H_inc)).clip(0.0, 1.0)
    return out, otf_mean_abs


def main():
    p = argparse.ArgumentParser(description="Incoherent-OTF defocus sweep")
    p.add_argument("--coco-root",    default="data/coco")
    p.add_argument("--model",        default="yolov8n.pt")
    p.add_argument("--max-images",   type=int, default=500)
    p.add_argument("--image-size",   type=int, default=640)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--iou",          type=float, default=0.45)
    p.add_argument("--bootstrap-iters", type=int, default=200)
    p.add_argument("--bootstrap-seed",  type=int, default=42)
    p.add_argument("--alphas",
                   default=",".join(str(a) for a in DEFAULT_ALPHAS))
    p.add_argument("--output-dir",   default="outputs/incoherent_defocus_sweep")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

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
    run_fn = lambda imgs: run_yolo(model, imgs, conf=args.conf, iou=args.iou, device=args.device)

    print("\n=== Baseline (clean) ===")
    clean_preds = run_fn(images)
    baseline_map50 = compute_map(
        [{**p, "image_id": iid} for p, iid in zip(clean_preds, image_ids)], targets
    )["map50"]
    print(f"  mAP@50 = {baseline_map50:.4f}")

    sweep = []
    for alpha in alphas:
        print(f"\n=== alpha = {alpha:.1f} ===")
        pairs = [apply_incoherent_defocus(img, alpha) for img in images]
        degraded_all = [d for d, _ in pairs]
        otf_mean_abs = pairs[0][1]  # same for all images at given alpha

        preds = run_fn(degraded_all)
        tagged = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]
        res = compute_map_ci(tagged, targets, n_bootstrap=args.bootstrap_iters,
                             seed=args.bootstrap_seed)
        s = res["map50"] / max(baseline_map50, 1e-6)
        print(f"  mAP={res['map50']:.4f} ±{res['map50_ci']:.4f}  S={s:.4f}  "
              f"<|H_inc|>={otf_mean_abs:.4f}")
        sweep.append({
            "alpha":       alpha,
            "map50":       res["map50"],
            "map50_ci":    res["map50_ci"],
            "sensitivity": s,
            "otf_mean_abs": otf_mean_abs,
        })

    out = {
        "operator":        "incoherent_defocus (OTF = F{|h|^2}, circ pupil + quadratic phase)",
        "model":           args.model,
        "image_size":      args.image_size,
        "max_images":      args.max_images,
        "conf_thresh":     args.conf,
        "iou_thresh":      args.iou,
        "n_bootstrap":     args.bootstrap_iters,
        "bootstrap_seed":  args.bootstrap_seed,
        "baseline_map50":  baseline_map50,
        "sweep":           sweep,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
