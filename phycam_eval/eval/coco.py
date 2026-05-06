"""
Shared COCO evaluation helpers used by the sweep scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .metrics import yolo_to_coco_category_ids


def load_coco_images(
    coco_root: str | Path,
    split: str = "val2017",
    max_images: int | None = 100,
    image_size: int = 640,
    image_offset: int = 0,
) -> tuple[list[int], list[np.ndarray], list[dict], dict]:
    """Load and resize COCO images as CHW float32 arrays in [0, 1]."""
    coco_root = Path(coco_root)
    ann_file = coco_root / "annotations" / f"instances_{split}.json"
    img_dir = coco_root / "images" / split

    with open(ann_file) as f:
        coco_data = json.load(f)

    image_ids = sorted(img["id"] for img in coco_data["images"])
    if image_offset:
        image_ids = image_ids[image_offset:]
    if max_images is not None:
        image_ids = image_ids[:max_images]
    id_to_meta = {img["id"]: img for img in coco_data["images"]}

    images, metas = [], []
    for image_id in image_ids:
        meta = id_to_meta[image_id]
        path = img_dir / meta["file_name"]
        pil = PILImage.open(path).convert("RGB").resize((image_size, image_size))
        arr = np.array(pil, dtype=np.float32) / 255.0
        images.append(arr.transpose(2, 0, 1))
        metas.append(meta)

    return image_ids, images, metas, coco_data


def build_coco_targets(
    coco_data: dict,
    image_ids: list[int],
    metas: list[dict],
    image_size: int,
) -> list[dict]:
    """Convert COCO annotations to resized xyxy targets for mAP evaluation."""
    id_to_anns: dict[int, list] = {image_id: [] for image_id in image_ids}
    for ann in coco_data["annotations"]:
        if ann["image_id"] in id_to_anns:
            id_to_anns[ann["image_id"]].append(ann)

    targets = []
    for image_id, meta in zip(image_ids, metas):
        anns = id_to_anns[image_id]
        scale_x = image_size / meta["width"]
        scale_y = image_size / meta["height"]
        gt_boxes = [
            [
                ann["bbox"][0] * scale_x,
                ann["bbox"][1] * scale_y,
                (ann["bbox"][0] + ann["bbox"][2]) * scale_x,
                (ann["bbox"][1] + ann["bbox"][3]) * scale_y,
            ]
            for ann in anns
        ]
        targets.append(
            {
                "image_id": image_id,
                "boxes": (
                    np.array(gt_boxes, dtype=np.float32)
                    if gt_boxes
                    else np.zeros((0, 4), dtype=np.float32)
                ),
                "labels": np.array([ann["category_id"] for ann in anns], dtype=int),
            }
        )
    return targets


def pick_edge_roi(image_np: np.ndarray, size: int = 128) -> tuple[int, int, int, int]:
    """Pick a slanted-edge ROI if possible, otherwise fall back to a center crop."""
    from .mtf import find_slanted_edge_roi

    height, width = image_np.shape[1:]
    gray = 0.299 * image_np[0] + 0.587 * image_np[1] + 0.114 * image_np[2]
    roi = find_slanted_edge_roi(gray)
    if roi is not None:
        return roi

    row0 = max(0, height // 2 - size // 2)
    col0 = max(0, width // 2 - size // 2)
    return (row0, min(height, row0 + size), col0, min(width, col0 + size))


def run_fasterrcnn(
    model,
    images_np: list[np.ndarray],
    conf: float = 0.25,
    device: str = "cpu",
) -> list[dict]:
    """Run torchvision Faster R-CNN on CHW float32 images. Returns COCO-format predictions.

    Torchvision detection models output COCO sparse category IDs (1–90) directly,
    so no remapping is needed (unlike YOLO's contiguous 0–79 indices).
    """
    import torch
    preds = []
    with torch.no_grad():
        for img_chw in images_np:
            tensor = torch.from_numpy(img_chw).float().to(device)
            result = model([tensor])[0]
            boxes  = result["boxes"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            mask   = scores >= conf
            preds.append({
                "boxes":  boxes[mask],
                "labels": labels[mask],
                "scores": scores[mask],
            })
    return preds


def run_detr(
    model,
    processor,
    images_np: list[np.ndarray],
    conf: float = 0.25,
    device: str = "cpu",
) -> list[dict]:
    """Run HuggingFace DETR on CHW float32 images. Returns COCO-format predictions.

    HuggingFace post_process_object_detection returns COCO 91-class IDs (1–90)
    directly, so no category remapping is needed unlike YOLO.
    """
    import torch
    from PIL import Image as PILImage

    preds = []
    with torch.no_grad():
        for img_chw in images_np:
            img_hwc = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_hwc)
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            target_sizes = torch.tensor(
                [[img_hwc.shape[0], img_hwc.shape[1]]], device=device
            )
            results = processor.post_process_object_detection(
                outputs, threshold=conf, target_sizes=target_sizes
            )[0]
            preds.append({
                "boxes":  results["boxes"].cpu().numpy(),
                "labels": results["labels"].cpu().numpy(),
                "scores": results["scores"].cpu().numpy(),
            })
    return preds


def warp_image_rolling_shutter(
    image: np.ndarray,
    velocity_x: float,
    readout_time: float,
) -> np.ndarray:
    """Apply horizontal rolling-shutter warp to a (C, H, W) float32 image.

    Row y is shifted left by velocity_x * readout_time * (y / H) pixels.
    Uses bilinear interpolation with edge-clamping.
    """
    from scipy.ndimage import map_coordinates

    C, H, W = image.shape
    rows = np.arange(H)
    cols = np.arange(W)
    col_grid, row_grid = np.meshgrid(cols, rows, indexing="xy")
    dx = velocity_x * readout_time * (row_grid / H)
    src_cols = (col_grid - dx).clip(0, W - 1)

    out = np.empty_like(image)
    for c in range(C):
        out[c] = map_coordinates(
            image[c], [row_grid, src_cols], order=1, mode="nearest"
        )
    return out


def warp_targets_rolling_shutter(
    targets: list[dict],
    velocity_x: float,
    readout_time: float,
    image_size: int,
) -> list[dict]:
    """Shift GT bounding boxes to match a rolling-shutter warped image.

    Each box is shifted by the displacement at its vertical midpoint so that
    mAP is measured on the warped image with correctly aligned ground truth.
    """
    warped = []
    for t in targets:
        boxes = t["boxes"].copy()
        if len(boxes):
            mid_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
            dx = velocity_x * readout_time * (mid_y / image_size)
            boxes[:, 0] = (boxes[:, 0] + dx).clip(0, image_size)
            boxes[:, 2] = (boxes[:, 2] + dx).clip(0, image_size)
        warped.append({**t, "boxes": boxes})
    return warped


def run_yolo(
    model,
    images_np: list[np.ndarray],
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
) -> list[dict]:
    """Run YOLO on CHW float32 images and return resized COCO-format predictions."""
    preds = []
    for img_chw in images_np:
        img_hwc = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        results = model(img_hwc, conf=conf, iou=iou, device=device, verbose=False)
        boxes, labels, scores = [], [], []
        for res in results:
            if res.boxes is not None and len(res.boxes):
                cls = res.boxes.cls.cpu().numpy().astype(int)
                boxes.append(res.boxes.xyxy.cpu().numpy())
                labels.append(yolo_to_coco_category_ids(cls))
                scores.append(res.boxes.conf.cpu().numpy())
        preds.append(
            {
                "boxes": np.concatenate(boxes, 0) if boxes else np.zeros((0, 4)),
                "labels": np.concatenate(labels, 0) if labels else np.zeros((0,), int),
                "scores": np.concatenate(scores, 0) if scores else np.zeros((0,)),
            }
        )
    return preds
