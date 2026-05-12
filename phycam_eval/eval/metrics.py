"""
Detection metrics for PhyCam-Eval.

Wraps pycocotools for mAP computation and adds per-class and sensitivity-curve
utilities needed for the sensitivity landscape S(θ).
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np


# Ultralytics/YOLO models expose COCO classes as contiguous indices 0..79.
# COCO annotations use the original paper IDs 1..90 with gaps, so we need an
# explicit remap before computing pycocotools metrics.
COCO80_TO_91 = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
], dtype=np.int64)


def yolo_to_coco_category_ids(class_ids) -> np.ndarray:
    """Map contiguous YOLO COCO80 class indices to sparse COCO category IDs."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if class_ids.size == 0:
        return class_ids.copy()
    if class_ids.min() < 0 or class_ids.max() >= len(COCO80_TO_91):
        raise ValueError("YOLO class IDs must be in [0, 79] for COCO models")
    return COCO80_TO_91[class_ids]


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute COCO-style mAP.

    Parameters
    ----------
    predictions : list of dicts with keys:
        "image_id" : int
        "boxes"    : (N, 4) tensor, xyxy format
        "labels"   : (N,)   tensor, integer category IDs
        "scores"   : (N,)   tensor, confidence scores
    targets : list of dicts with keys:
        "image_id" : int
        "boxes"    : (M, 4) tensor, xyxy format
        "labels"   : (M,)   tensor

    Returns
    -------
    dict with keys:
        "map50"      : float — mAP @ IoU=0.50
        "map50_95"   : float — mAP @ IoU=0.50:0.05:0.95
        "per_class_ap" : dict[int, float]
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError(
            "pycocotools is required for mAP computation. "
            "Install with: pip install pycocotools"
        )

    target_image_ids = [int(t["image_id"]) for t in targets]
    pred_image_ids = [int(p["image_id"]) for p in predictions]
    if len(target_image_ids) != len(set(target_image_ids)):
        raise ValueError("targets must contain unique image_id values")
    if len(pred_image_ids) != len(set(pred_image_ids)):
        raise ValueError("predictions must contain unique image_id values")

    # Build COCO-format GT annotations
    gt_annotations = []
    gt_images = []
    # COCOeval stores matched annotation IDs in arrays where 0 is the sentinel
    # for "unmatched", so real annotation IDs must be positive.
    ann_id = 1

    for target in targets:
        image_id = int(target["image_id"])
        gt_images.append({"id": image_id})
        boxes = target["boxes"]
        labels = target["labels"]
        areas = target.get("area", None)
        iscrowd = target.get("iscrowd", None)

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else box
            if areas is not None and len(areas) > idx:
                area_val = areas[idx].item() if hasattr(areas[idx], "item") else areas[idx]
            else:
                area_val = (x2 - x1) * (y2 - y1)
            if iscrowd is not None and len(iscrowd) > idx:
                crowd_val = iscrowd[idx].item() if hasattr(iscrowd[idx], "item") else iscrowd[idx]
            else:
                crowd_val = 0
            gt_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                "area": float(area_val),
                "iscrowd": int(crowd_val),
            })
            ann_id += 1

    # Collect unique category IDs
    all_cat_ids = sorted(set(a["category_id"] for a in gt_annotations))
    gt_dataset = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": [{"id": c, "name": str(c)} for c in all_cat_ids],
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    # Build COCO-format predictions
    dt_list = []
    for pred in predictions:
        image_id = int(pred["image_id"])
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else box
            dt_list.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    if not dt_list:
        return {"map50": 0.0, "map50_95": 0.0, "per_class_ap": {}}

    coco_dt = coco_gt.loadRes(dt_list)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if iou_thresholds is not None:
        evaluator.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # stats layout: [mAP@50:95, mAP@50, mAP@75, mAP@50:95_small, _medium, _large,
    #                 AR@1, AR@10, AR@100, AR@100_small, AR@100_medium, AR@100_large]
    map50_95 = float(evaluator.stats[0])
    map50 = float(evaluator.stats[1])
    map75 = float(evaluator.stats[2])
    map50_95_small = float(evaluator.stats[3])
    map50_95_medium = float(evaluator.stats[4])
    map50_95_large = float(evaluator.stats[5])

    # Per-class AP @ IoU=0.50 (precision at iouThr=0.50, areaRng='all', maxDets=100)
    per_class_ap: dict[int, float] = {}
    for cat_idx, cat_id in enumerate(all_cat_ids):
        # precision shape: (T, R, K, A, M)
        # T=10 iou thresholds, R=101 recall pts, K=cats, A=4 area, M=3 maxdets
        precision = evaluator.eval["precision"]
        # Use the actual 0.50 threshold index when a custom IoU grid is supplied.
        iou_idx = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
        prec = precision[iou_idx, :, cat_idx, 0, 2]  # (101,)
        prec = prec[prec > -1]
        ap = float(prec.mean()) if len(prec) > 0 else 0.0
        per_class_ap[cat_id] = ap

    return {
        "map50": map50,
        "map50_95": map50_95,
        "map75": map75,
        "map50_95_small": map50_95_small,
        "map50_95_medium": map50_95_medium,
        "map50_95_large": map50_95_large,
        "per_class_ap": per_class_ap,
    }


def compute_map_ci(
    predictions: list[dict],
    targets: list[dict],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute COCO-style mAP with a bootstrap 95% confidence interval.

    Resamples image IDs with replacement `n_bootstrap` times and re-evaluates
    mAP@50 on each resample.  Returns the main mAP plus:
        "map50_ci" : float — half-width of the 95% interval (1.96 * std_bootstrap)

    The bootstrap is run with suppressed pycocotools stdout to avoid 20x log spam.
    """
    import contextlib, io

    main = compute_map(predictions, targets)

    image_ids = [t["image_id"] for t in targets]
    pred_by_id = {p["image_id"]: p for p in predictions}
    tgt_by_id  = {t["image_id"]: t for t in targets}
    if set(pred_by_id) != set(tgt_by_id):
        raise ValueError(
            "compute_map_ci requires one prediction record for every target image_id"
        )
    if len(image_ids) < 2:
        return {**main, "map50_ci": 0.0}

    rng = np.random.default_rng(seed)
    N = len(image_ids)

    from tqdm import tqdm
    bootstrap_map50: list[float] = []
    bootstrap_map50_95: list[float] = []
    bootstrap_errors: list[str] = []
    for _ in tqdm(range(n_bootstrap), desc="  bootstrap", unit="resample",
                  dynamic_ncols=True, leave=False):
        chosen = rng.choice(image_ids, size=N, replace=True).tolist()
        # Reassign sequential IDs so pycocotools treats each draw as a distinct
        # image — without this, duplicate image_ids collapse the imgs dict while
        # tripling GT boxes in imgToAnns, corrupting precision/recall matching.
        sub_preds, sub_targets = [], []
        for new_id, orig_id in enumerate(chosen):
            p = pred_by_id[orig_id].copy()
            t = tgt_by_id[orig_id].copy()
            p["image_id"] = new_id
            t["image_id"] = new_id
            sub_preds.append(p)
            sub_targets.append(t)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = compute_map(sub_preds, sub_targets)
            bootstrap_map50.append(r["map50"])
            bootstrap_map50_95.append(r["map50_95"])
        except Exception as exc:
            bootstrap_errors.append(repr(exc))

    if bootstrap_errors:
        msg = (
            f"{len(bootstrap_errors)} of {n_bootstrap} bootstrap mAP resamples failed; "
            f"first failure: {bootstrap_errors[0]}"
        )
        if not bootstrap_map50:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _ci(samples: list[float]) -> float:
        return 1.96 * float(np.std(samples, ddof=1)) if len(samples) >= 2 else 0.0

    return {
        **main,
        "map50_ci": _ci(bootstrap_map50),
        "map50_95_ci": _ci(bootstrap_map50_95),
        "bootstrap_failures": len(bootstrap_errors),
    }


def sensitivity_ratio(
    degraded_metric: float,
    baseline_metric: float,
    eps: float = 1e-8,
) -> float:
    """
    Compute the sensitivity ratio S(θ) = M(degraded) / M(baseline).

    S = 1.0 → no degradation effect.
    S = 0.5 → metric dropped by 50%.
    """
    return degraded_metric / max(baseline_metric, eps)
