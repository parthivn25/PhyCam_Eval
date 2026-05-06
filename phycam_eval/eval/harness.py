"""
Evaluation harness — runs a detection model over degraded images.

Supports:
  - ultralytics YOLO (primary supported path)  → YOLOHarness
  - torchvision Faster R-CNN (ResNet-50 FPN)   → FasterRCNNHarness
  - Any callable model(images_np) → list[dict]

Usage
-----
    from phycam_eval.eval.harness import YOLOHarness, FasterRCNNHarness
    from phycam_eval.degradations import DefocusOperator

    # YOLO (primary)
    harness = YOLOHarness(
        model_path="yolov8n.pt",
        degradation=DefocusOperator(alpha=1.5),
        device="cpu",
    )
    results = harness.run_coco(coco_root="data/coco", max_images=500)
    print(f"mAP@50 = {results.map50:.4f}")

    # Faster R-CNN (multi-architecture validation)
    harness2 = FasterRCNNHarness(
        degradation=DefocusOperator(alpha=1.5),
        device="cpu",
    )
    results2 = harness2.run_coco(coco_root="data/coco", max_images=500)
    print(f"mAP@50 = {results2.map50:.4f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class HarnessResults:
    """Results from one full harness run."""
    degradation_name: str
    degradation_params: dict = field(default_factory=dict)
    map50: float = 0.0
    map50_95: float = 0.0
    per_class_ap: dict = field(default_factory=dict)
    n_images: int = 0
    elapsed_seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.n_images / max(self.elapsed_seconds, 1e-6)

    def __str__(self) -> str:
        return (f"HarnessResults({self.degradation_name}): "
                f"mAP@50={self.map50:.4f}  mAP@50:95={self.map50_95:.4f}  "
                f"n={self.n_images}  fps={self.fps:.1f}")


class YOLOHarness:
    """
    YOLO-based evaluation harness using ultralytics.

    Applies a physics-inspired degradation to every image before inference,
    then collects COCO-format predictions for mAP computation.

    Parameters
    ----------
    model_path  : str | Path
        Path to a YOLO weights file (e.g. "yolov8n.pt").
        If the file doesn't exist, ultralytics will attempt to download it.
    degradation : callable | None
        Any operator with signature: op(image_np: ndarray) -> ndarray
        where both arrays are (C,H,W) float32 in [0,1].
        None → evaluate on clean images.
    image_size  : int
        Inference resolution (default 640).
    conf_thresh : float
        YOLO confidence threshold (default 0.25).
    iou_thresh  : float
        YOLO NMS IoU threshold (default 0.45).
    device      : str
        "cpu", "cuda", "mps", or torch device string.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        degradation: Callable | None = None,
        image_size: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOHarness. "
                "Install with: pip install ultralytics"
            )
        self.model = YOLO(str(model_path))
        self.degradation = degradation
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

    def _degrade(self, image_np: np.ndarray) -> np.ndarray:
        """Apply degradation to a (C,H,W) float32 array. Returns same shape."""
        if self.degradation is None:
            return image_np
        return self.degradation(image_np)

    def _np_to_uint8_hwc(self, arr: np.ndarray) -> np.ndarray:
        """(C,H,W) float32 [0,1] → (H,W,C) uint8 [0,255] for YOLO."""
        return (arr.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

    def run_coco(
        self,
        coco_root: str | Path,
        split: str = "val2017",
        max_images: int | None = None,
        batch_size: int = 1,
    ) -> HarnessResults:
        """
        Run YOLO inference over COCO val subset with degradation applied.

        Parameters
        ----------
        coco_root  : path to COCO directory (must contain images/ and annotations/)
        split      : "val2017" (default) or "train2017"
        max_images : truncate to this many images
        batch_size : images per YOLO call (default 1 for determinism)

        Returns
        -------
        HarnessResults with mAP@50, mAP@50:95, per_class_ap
        """
        from .coco import build_coco_targets, load_coco_images, run_yolo
        from .metrics import compute_map

        image_ids, images, metas, coco_data = load_coco_images(
            coco_root=coco_root,
            split=split,
            max_images=max_images,
            image_size=self.image_size,
        )
        targets = build_coco_targets(coco_data, image_ids, metas, self.image_size)

        degraded = [self._degrade(img) for img in images]
        t_start = time.perf_counter()

        preds = run_yolo(
            self.model,
            degraded,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
        )
        all_preds = [{**p, "image_id": iid} for p, iid in zip(preds, image_ids)]

        elapsed = time.perf_counter() - t_start

        deg_name = repr(self.degradation) if self.degradation else "Clean"
        results_obj = HarnessResults(
            degradation_name=deg_name,
            n_images=len(image_ids),
            elapsed_seconds=elapsed,
        )

        if all_preds:
            map_res = compute_map(all_preds, targets)
            results_obj.map50    = map_res["map50"]
            results_obj.map50_95 = map_res["map50_95"]
            results_obj.per_class_ap = map_res.get("per_class_ap", {})

        return results_obj


class FasterRCNNHarness:
    """
    Torchvision Faster R-CNN evaluation harness (ResNet-50 FPN backbone).

    Drop-in alternative to YOLOHarness for multi-architecture robustness
    validation.  Uses COCO-pretrained weights; category IDs are already
    in COCO sparse format (1–90), so no remapping is needed.

    Parameters
    ----------
    weights     : str
        Torchvision weights name.  "COCO_V1" (default) selects the
        standard COCO-pretrained ResNet-50 FPN checkpoint.
    degradation : callable | None
        Same spec as YOLOHarness: op(image_np: ndarray) -> ndarray,
        (C,H,W) float32 [0,1] → (C,H,W) float32 [0,1].
    image_size  : int
        Resize target before inference (default 640).
    conf_thresh : float
        Score threshold; predictions below this are discarded (default 0.25).
    device      : str
        "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        weights: str = "COCO_V1",
        degradation=None,
        image_size: int = 640,
        conf_thresh: float = 0.25,
        device: str = "cpu",
    ) -> None:
        try:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
        except ImportError:
            raise ImportError(
                "torchvision is required for FasterRCNNHarness. "
                "Install with: pip install torchvision"
            )
        import torch as _torch
        self._torch = _torch
        w = FasterRCNN_ResNet50_FPN_Weights[weights]
        self.model = fasterrcnn_resnet50_fpn(weights=w)
        self.model.eval()
        self.model.to(device)
        self.degradation = degradation
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.device = device

    def _degrade(self, image_np: np.ndarray) -> np.ndarray:
        if self.degradation is None:
            return image_np
        return self.degradation(image_np)

    def run_coco(
        self,
        coco_root,
        split: str = "val2017",
        max_images: int | None = None,
        batch_size: int = 1,
    ) -> HarnessResults:
        """
        Run Faster R-CNN inference over COCO val subset with degradation applied.

        Parameters and return value match YOLOHarness.run_coco exactly so the
        two harnesses are interchangeable in sweep scripts.
        """
        from .coco import build_coco_targets, load_coco_images
        from .metrics import compute_map

        image_ids, images, metas, coco_data = load_coco_images(
            coco_root=coco_root,
            split=split,
            max_images=max_images,
            image_size=self.image_size,
        )
        targets = build_coco_targets(coco_data, image_ids, metas, self.image_size)

        degraded = [self._degrade(img) for img in images]
        t_start = time.perf_counter()

        all_preds = []
        with self._torch.no_grad():
            for img_chw, iid in zip(degraded, image_ids):
                tensor = self._torch.from_numpy(img_chw).float().to(self.device)
                result = self.model([tensor])[0]
                boxes  = result["boxes"].cpu().numpy()
                labels = result["labels"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                mask   = scores >= self.conf_thresh
                all_preds.append({
                    "image_id": iid,
                    "boxes":    boxes[mask],
                    "labels":   labels[mask],
                    "scores":   scores[mask],
                })

        elapsed = time.perf_counter() - t_start
        deg_name = repr(self.degradation) if self.degradation else "Clean"
        results_obj = HarnessResults(
            degradation_name=deg_name,
            n_images=len(image_ids),
            elapsed_seconds=elapsed,
        )
        if all_preds:
            map_res = compute_map(all_preds, targets)
            results_obj.map50    = map_res["map50"]
            results_obj.map50_95 = map_res["map50_95"]
            results_obj.per_class_ap = map_res.get("per_class_ap", {})

        return results_obj

    def __repr__(self) -> str:
        return (
            f"FasterRCNNHarness(weights=COCO_V1, "
            f"conf={self.conf_thresh}, device={self.device})"
        )
