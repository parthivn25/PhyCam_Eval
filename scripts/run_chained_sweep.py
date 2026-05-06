"""
scripts/run_chained_sweep.py — chained HDR+noise sensitivity sweep.

Evaluates the full N_σ(Q_β(I)) pipeline across a 2-D grid of (β, ISO) values
to measure how HDR compression and sensor noise compound.

Physical pipeline:
    I_d = N_σ(Q_β(I_ideal))
    Q_β applied first (dynamic range compression), then N_σ (sensor noise).

Usage:
    python3 scripts/run_chained_sweep.py \\
        --coco-root data/coco \\
        --model yolov8n.pt \\
        --max-images 100 \\
        --output-dir outputs/chained_sweep

Outputs:
    outputs/chained_sweep/
        results.json                  — mAP for every (β, ISO) combination
        figure_chained_heatmap.png    — mAP heatmap
        figure_chained_slices.png     — mAP vs ISO at fixed β values
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phycam_eval.degradations import HDRCompressionOperator, SensorNoiseOperator
from phycam_eval.eval.coco import build_coco_targets, load_coco_images, run_yolo, run_fasterrcnn
from phycam_eval.eval.metrics import compute_map_ci
from phycam_eval.eval.plotting import PAPER_COLORS, paper_style, save_figure, style_axes


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


DEFAULT_BETAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
DEFAULT_ISOS = [100, 400, 1600, 6400]

BASE_ISO = 100
BASE_READ_NOISE = 0.002
BASE_GAIN = 5e-5


def run_grid(images, image_ids, targets, run_fn, betas, iso_values):
    """Run every (β, ISO) combination. Returns baseline_map50, grid of (map50, map50_ci)."""
    # Baseline (β=1, ISO=100 clean)
    print("\n=== Baseline (clean) ===")
    clean_preds = run_fn(images)
    tagged_clean = [{**p, "image_id": iid} for p, iid in zip(clean_preds, image_ids)]
    map_res = compute_map_ci(tagged_clean, targets)
    baseline_map50 = map_res["map50"]
    print(f"  mAP@50 = {baseline_map50:.4f} ±{map_res['map50_ci']:.4f}")

    grid = {}          # (beta, iso) -> {"map50": float, "map50_ci": float}
    for beta in betas:
        hdr_op = HDRCompressionOperator(beta=beta)
        t0 = time.perf_counter()
        hdr_degraded = [hdr_op(img) for img in images]
        print(f"\n--- β = {beta:.2f}  ({time.perf_counter()-t0:.1f}s) ---")

        for iso in iso_values:
            noise_op = SensorNoiseOperator.from_iso(
                iso,
                base_iso=BASE_ISO,
                base_read_noise=BASE_READ_NOISE,
                base_gain=BASE_GAIN,
                seed=42,
            )
            chained = [noise_op(img) for img in hdr_degraded]
            preds = run_fn(chained)
            tagged = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]
            map_res = compute_map_ci(tagged, targets)
            m, ci = map_res["map50"], map_res["map50_ci"]
            grid[(beta, iso)] = {"map50": m, "map50_ci": ci}
            print(f"  ISO={iso:5d}  mAP@50={m:.4f} ±{ci:.4f}  S={m/max(baseline_map50,1e-6):.3f}")

    return baseline_map50, grid


def plot_heatmap(betas, iso_values, grid, baseline_map50, save_path):
    """Normalised mAP heatmap: rows = β (top to bottom), cols = ISO (left to right)."""
    mat = np.array(
        [[grid[(b, iso)]["map50"] / max(baseline_map50, 1e-6) for iso in iso_values] for b in betas]
    )
    with paper_style():
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="YlOrRd",  # Sequential: yellow (good) → red (bad). Better than diverging RdYlGn.
            vmin=0.0,
            vmax=1.05,
            origin="upper",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalised mAP@50", fontsize=9.5)

        ax.set_xticks(range(len(iso_values)))
        ax.set_xticklabels([str(v) for v in iso_values])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("ISO (sensor noise)")
        ax.set_ylabel("β (HDR compression)")
        ax.set_title("Chained pipeline: N_σ(Q_β(I))\nNormalised mAP@50 vs (β, ISO)", loc="left")

        for i, b in enumerate(betas):
            for j, _iso in enumerate(iso_values):
                ax.text(
                    j, i, f"{mat[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="black" if 0.3 < mat[i, j] < 0.85 else "white",
                )

        fig.tight_layout()
        save_figure(fig, save_path)
        plt.close(fig)


def plot_gamma_heatmap(betas, iso_values, grid, baseline_map50, save_path):
    """Super-additivity coefficient γ = S_meas / S_pred heatmap.
    γ < 1.0 indicates super-additive compounding (interaction worse than independence assumption).
    S_pred = S_beta * S_iso (independence model).
    """
    mat = np.full((len(betas), len(iso_values)), np.nan)
    # Marginal sensitivities: S_beta[i] = grid[beta_i, ISO_0] / baseline, S_iso[j] = grid[beta_0, ISO_j] / baseline
    # (using the first point in each dimension as the reference)
    s_beta_marginal = [grid[(b, iso_values[0])]["map50"] / baseline_map50 if (b, iso_values[0]) in grid else np.nan for b in betas]
    s_iso_marginal = [grid[(betas[0], iso)]["map50"] / baseline_map50 if (betas[0], iso) in grid else np.nan for iso in iso_values]

    for i, b in enumerate(betas):
        for j, iso in enumerate(iso_values):
            if (b, iso) in grid:
                s_meas = grid[(b, iso)]["map50"] / baseline_map50
                s_beta = s_beta_marginal[i]
                s_iso = s_iso_marginal[j]
                if not (np.isnan(s_beta) or np.isnan(s_iso)):
                    s_pred = s_beta * s_iso
                    mat[i, j] = s_meas / max(s_pred, 1e-6)

    with paper_style():
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",  # Diverging: red (super-additive, γ<1) → blue (sub-additive, γ>1)
            vmin=0.8,
            vmax=1.2,
            origin="upper",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Super-additivity γ = S_meas / S_pred", fontsize=9.5)

        ax.set_xticks(range(len(iso_values)))
        ax.set_xticklabels([str(v) for v in iso_values])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:.1f}" for b in betas])
        ax.set_xlabel("ISO (sensor noise)")
        ax.set_ylabel("β (HDR compression)")
        ax.set_title("Chained pipeline: super-additivity coefficient γ\nγ<1 = super-additive (compounded worse than independence), γ=1 = independent", loc="left")

        for i, b in enumerate(betas):
            for j, _iso in enumerate(iso_values):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="black" if 0.95 < val < 1.05 else "white",
                    )

        fig.tight_layout()
        save_figure(fig, save_path)
        plt.close(fig)


def plot_slices(betas, iso_values, grid, baseline_map50, save_path):
    """Line plot: mAP vs ISO for each fixed β."""
    import matplotlib.cm as cm
    n = len(betas)
    colors = [cm.viridis(i / max(n - 1, 1)) for i in range(n)]

    with paper_style():
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        style_axes(ax, grid_axis="both")

        ax.axhline(1.0, color=PAPER_COLORS["baseline"], linestyle=(0, (3, 3)),
                   linewidth=1.1, label="Clean baseline")
        ax.axhline(0.9, color=PAPER_COLORS["threshold"], linestyle=(0, (6, 2)),
                   linewidth=1.0, label="10% drop threshold")

        for beta, color in zip(betas, colors):
            s_vals = [grid[(beta, iso)]["map50"] / max(baseline_map50, 1e-6) for iso in iso_values]
            ax.plot(
                iso_values, s_vals,
                color=color,
                marker="o", markersize=6, markerfacecolor="white", markeredgewidth=1.5,
                linewidth=2.0,
                label=f"β = {beta:.1f}",
            )

        ax.set_xscale("log")
        ax.set_xticks(iso_values)
        ax.set_xticklabels([str(v) for v in iso_values])
        ax.set_xlabel("ISO (sensor noise)")
        ax.set_ylabel("Normalised mAP@50")
        ax.set_title("Chained pipeline: mAP vs ISO at fixed β", loc="left")
        ax.legend(loc="lower left", ncol=2, columnspacing=1.2, handlelength=2.2)
        ax.text(
            0.99, 0.97,
            f"Clean mAP@50 = {baseline_map50:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.2, color=PAPER_COLORS["muted"],
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": PAPER_COLORS["grid"]},
        )
        fig.tight_layout()
        save_figure(fig, save_path)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Chained HDR+noise sensitivity sweep")
    p.add_argument("--coco-root",   default="data/coco")
    p.add_argument("--model",       default="yolov8n.pt")
    p.add_argument("--detector",    default="yolo", choices=["yolo", "fasterrcnn"],
                   help="Detector backend (default: yolo)")
    p.add_argument("--max-images",  type=int, default=100)
    p.add_argument("--image-size",  type=int, default=640)
    p.add_argument("--output-dir",  default="outputs/chained_sweep")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--betas",
                   default=",".join(str(v) for v in DEFAULT_BETAS))
    p.add_argument("--iso-values",
                   default=",".join(str(v) for v in DEFAULT_ISOS))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    betas = [float(v) for v in args.betas.split(",")]
    iso_values = [int(v) for v in args.iso_values.split(",")]

    print(f"Loading COCO val images from {args.coco_root} ...")
    image_ids, images, metas, coco_data = load_coco_images(
        args.coco_root, max_images=args.max_images, image_size=args.image_size
    )
    from phycam_eval.eval.metrics import compute_map
    targets = build_coco_targets(coco_data, image_ids, metas, args.image_size)

    print(f"Loading detector: {args.detector}")
    _, run_fn, det_tag = _load_detector(args)

    baseline_map50, grid = run_grid(
        images, image_ids, targets, run_fn, betas, iso_values
    )

    # Save JSON (detector-tagged filename for non-default runs)
    suffix = f"_{det_tag}" if det_tag != "yolo" else ""
    results = {
        "detector": det_tag,
        "baseline_map50": baseline_map50,
        "betas": betas,
        "iso_values": iso_values,
        "grid": [
            {
                "beta": b, "iso": iso,
                "map50": grid[(b, iso)]["map50"],
                "map50_ci": grid[(b, iso)]["map50_ci"],
                "sensitivity": grid[(b, iso)]["map50"] / max(baseline_map50, 1e-6),
            }
            for b in betas for iso in iso_values
        ],
    }
    results_path = out_dir / f"results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_heatmap(betas, iso_values, grid, baseline_map50,
                 str(out_dir / f"figure_chained_heatmap{suffix}.png"))
    print(f"Heatmap saved to {out_dir}/figure_chained_heatmap{suffix}.png")

    plot_gamma_heatmap(betas, iso_values, grid, baseline_map50,
                       str(out_dir / f"figure_chained_gamma_heatmap{suffix}.png"))
    print(f"Gamma heatmap saved to {out_dir}/figure_chained_gamma_heatmap{suffix}.png")

    plot_slices(betas, iso_values, grid, baseline_map50,
                str(out_dir / f"figure_chained_slices{suffix}.png"))
    print(f"Slices plot saved to {out_dir}/figure_chained_slices{suffix}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
