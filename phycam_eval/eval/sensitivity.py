"""
Sensitivity landscape: S(θ) = M(f(I_d(θ))) / M(f(I_ideal))

For each physical parameter θ ∈ {α (defocus), β (HDR compression), gain (ISO), ...}
and each perception metric M ∈ {mAP, IoU, MTF50}, plots how performance degrades
as the parameter deviates from ideal.

This produces the core figures of the paper:
    Figure 3 — mAP and MTF50 on a shared axis vs. physical parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt

from .plotting import (
    PAPER_COLORS,
    add_panel_label,
    paper_style,
    save_figure,
    style_axes,
    style_twin_axis,
)


@dataclass
class SensitivityPoint:
    """One (θ, metric) data point on the sensitivity curve."""
    param_value: float
    map50: float | None = None
    map50_95: float | None = None
    map50_ci: float = 0.0              # 95% CI half-width on absolute mAP@50
    mtf50: float | None = None
    sensitivity_map: float | None = None   # S = map50 / baseline_map50
    sensitivity_mtf: float | None = None   # S = mtf50 / baseline_mtf50


class SensitivitySweep:
    """
    Manages a parameter sweep and accumulates SensitivityPoints.

    Parameters
    ----------
    param_name  : str  — human-readable name, e.g. "defocus_alpha" or "ISO"
    param_values: sequence of floats swept over
    baseline_map: float — mAP on clean (undegraded) images (for normalisation)
    baseline_mtf: float — MTF50 on clean images (for normalisation)
    """

    def __init__(
        self,
        param_name: str,
        param_values: Iterable[float],
        baseline_map: float = 1.0,
        baseline_mtf: float = 0.5,
    ) -> None:
        self.param_name = param_name
        self.param_values = list(param_values)
        self.baseline_map = baseline_map
        self.baseline_mtf = baseline_mtf
        self.points: list[SensitivityPoint] = []

    def add(
        self,
        param_value: float,
        map50: float | None = None,
        mtf50: float | None = None,
        map50_ci: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Record one sweep point."""
        s_map = map50 / max(self.baseline_map, 1e-8) if map50 is not None else None
        s_mtf = mtf50 / max(self.baseline_mtf, 1e-8) if mtf50 is not None else None
        self.points.append(SensitivityPoint(
            param_value=param_value,
            map50=map50,
            map50_ci=map50_ci,
            mtf50=mtf50,
            sensitivity_map=s_map,
            sensitivity_mtf=s_mtf,
        ))

    @property
    def param_array(self) -> np.ndarray:
        return np.array([p.param_value for p in self.points])

    @property
    def map50_array(self) -> np.ndarray:
        return np.array([p.map50 if p.map50 is not None else np.nan for p in self.points])

    @property
    def mtf50_array(self) -> np.ndarray:
        return np.array([p.mtf50 if p.mtf50 is not None else np.nan for p in self.points])

    @property
    def map50_ci_array(self) -> np.ndarray:
        return np.array([p.map50_ci for p in self.points])

    @property
    def sensitivity_map_array(self) -> np.ndarray:
        return np.array([p.sensitivity_map if p.sensitivity_map is not None else np.nan for p in self.points])

    @property
    def sensitivity_mtf_array(self) -> np.ndarray:
        return np.array([p.sensitivity_mtf if p.sensitivity_mtf is not None else np.nan for p in self.points])

    def find_threshold_param(self, metric: str = "map50", drop_fraction: float = 0.1) -> float | None:
        """
        Find the parameter value at which the metric drops by `drop_fraction`
        relative to baseline.  E.g. drop_fraction=0.1 → 10% mAP drop.

        Returns None if the threshold is never crossed.
        """
        target = 1.0 - drop_fraction
        if metric == "map50":
            s = self.sensitivity_map_array
        elif metric == "mtf50":
            s = self.sensitivity_mtf_array
        else:
            raise ValueError(f"Unknown metric: {metric}")

        theta = self.param_array
        below = np.where(s <= target)[0]
        if len(below) == 0:
            return None
        idx = below[0]
        if idx == 0:
            return float(theta[0])
        # Linear interpolation
        t0, t1 = theta[idx - 1], theta[idx]
        s0, s1 = s[idx - 1], s[idx]
        frac = (target - s0) / (s1 - s0)
        return float(t0 + frac * (t1 - t0))

    def plot(
        self,
        save_path: str | None = None,
        figsize: tuple[float, float] = (8.4, 5.8),
        title: str | None = None,
        close: bool = False,
        show_mtf: bool = True,
    ) -> plt.Figure:
        """
        Render a two-panel sensitivity figure:
        panel A — normalised sensitivity curves
        panel B — absolute metric values

        Parameters
        ----------
        show_mtf : bool
            When False, omits all MTF50-related elements (sensitivity line,
            threshold vline, bottom-right axis, and MTF annotation text).
            Use when MTF50 is not reported for the sweep (e.g. non-monotone
            under per-channel Q_β normalisation).
        """
        theta = self.param_array
        s_map = self.sensitivity_map_array
        s_mtf = self.sensitivity_mtf_array
        thr_map = self.find_threshold_param("map50", 0.10)
        thr_mtf = self.find_threshold_param("mtf50", 0.10) if show_mtf else None

        with paper_style():
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 1, height_ratios=(2.2, 1.25), hspace=0.08)
            ax_top = fig.add_subplot(gs[0])
            ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
            ax_bottom_r = ax_bottom.twinx()

            add_panel_label(ax_top, "A")
            add_panel_label(ax_bottom, "B")
            style_axes(ax_top, grid_axis="both")
            style_axes(ax_bottom, grid_axis="y")
            style_twin_axis(ax_bottom_r, PAPER_COLORS["mtf"])

            if title:
                ax_top.set_title(title, loc="left", pad=12)

            ax_top.axhline(
                1.0,
                color=PAPER_COLORS["baseline"],
                linestyle=(0, (3, 3)),
                linewidth=1.1,
                label="Clean baseline",
            )
            ax_top.axhline(
                0.9,
                color=PAPER_COLORS["threshold"],
                linestyle=(0, (6, 2)),
                linewidth=1.0,
                label="10% drop threshold",
            )

            if not np.all(np.isnan(s_map)):
                ci_norm = self.map50_ci_array / max(self.baseline_map, 1e-8)
                if np.any(ci_norm > 0):
                    ax_top.fill_between(
                        theta,
                        s_map - ci_norm,
                        s_map + ci_norm,
                        alpha=0.15,
                        color=PAPER_COLORS["map"],
                        linewidth=0,
                    )
                ax_top.plot(
                    theta,
                    s_map,
                    color=PAPER_COLORS["map"],
                    marker="o",
                    markersize=5.8,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    linewidth=2.2,
                    label="mAP@50 sensitivity",
                )
            if show_mtf and not np.all(np.isnan(s_mtf)):
                ax_top.plot(
                    theta,
                    s_mtf,
                    color=PAPER_COLORS["mtf"],
                    marker="s",
                    markersize=5.2,
                    markerfacecolor="white",
                    markeredgewidth=1.4,
                    linewidth=2.0,
                    label="MTF50 sensitivity",
                )

            if thr_map is not None:
                ax_top.axvline(
                    thr_map,
                    color=PAPER_COLORS["map"],
                    linestyle=(0, (2, 2)),
                    linewidth=1.0,
                    alpha=0.75,
                )
            if show_mtf and thr_mtf is not None:
                ax_top.axvline(
                    thr_mtf,
                    color=PAPER_COLORS["mtf"],
                    linestyle=(0, (2, 2)),
                    linewidth=1.0,
                    alpha=0.75,
                )

            if show_mtf:
                top_max = np.nanmax(np.r_[s_map, s_mtf]) if len(theta) else 1.0
            else:
                top_max = np.nanmax(s_map) if len(theta) and not np.all(np.isnan(s_map)) else 1.0
            ax_top.set_ylim(0.0, max(1.06, min(1.18, top_max + 0.08)))
            ax_top.set_ylabel("Normalized sensitivity")
            ax_top.legend(loc="lower left", ncol=2, columnspacing=1.2, handlelength=2.4)
            ax_top.tick_params(labelbottom=False)

            def _fmt(v: float) -> str:
                """Format threshold: integer if close to one, else 4 sig figs."""
                if v == v and abs(v - round(v)) < 0.5 and abs(v) >= 10:
                    return str(int(round(v)))
                return f"{v:.4g}"

            thresholds = []
            if thr_map is not None:
                thresholds.append(f"mAP 10% drop: {_fmt(thr_map)}")
            if show_mtf and thr_mtf is not None:
                thresholds.append(f"MTF 10% drop: {_fmt(thr_mtf)}")
            if not thresholds:
                thresholds.append("No 10% drop within tested range")
            ax_top.text(
                0.99,
                0.05,
                "\n".join(thresholds),
                transform=ax_top.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.2,
                color=PAPER_COLORS["muted"],
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"]},
            )

            map50_arr = self.map50_array
            mtf50_arr = self.mtf50_array
            if not np.all(np.isnan(map50_arr)):
                ci_abs = self.map50_ci_array
                if np.any(ci_abs > 0):
                    ax_bottom.fill_between(
                        theta,
                        map50_arr - ci_abs,
                        map50_arr + ci_abs,
                        alpha=0.15,
                        color=PAPER_COLORS["map"],
                        linewidth=0,
                    )
                ax_bottom.plot(
                    theta,
                    map50_arr,
                    color=PAPER_COLORS["map"],
                    marker="o",
                    markersize=5.4,
                    markerfacecolor="white",
                    markeredgewidth=1.4,
                    linewidth=1.9,
                )
                ax_bottom.set_ylabel("mAP@50", color=PAPER_COLORS["map"])
                ax_bottom.set_ylim(0.0, 1.0)  # Standardized 0-1 scale for readability
                ax_bottom.tick_params(axis="y", colors=PAPER_COLORS["map"])
                ax_bottom.spines["left"].set_color(PAPER_COLORS["map"])

            if show_mtf and not np.all(np.isnan(mtf50_arr)):
                ax_bottom_r.plot(
                    theta,
                    mtf50_arr,
                    color=PAPER_COLORS["mtf"],
                    marker="s",
                    markersize=4.8,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    linewidth=1.7,
                )
                ax_bottom_r.set_ylabel("MTF50 (cy/px)", color=PAPER_COLORS["mtf"])

            ax_bottom.set_xlabel(self.param_name)
            if len(theta) <= 12:
                ax_bottom.set_xticks(theta)

            ax_bottom.text(
                0.01,
                0.96,
                f"Clean mAP@50 = {self.baseline_map:.4f}\nClean MTF50 = {self.baseline_mtf:.4f}",
                transform=ax_bottom.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color=PAPER_COLORS["muted"],
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PAPER_COLORS["grid"]},
            )

            if save_path:
                save_figure(fig, save_path)
            if close:
                plt.close(fig)

            return fig
