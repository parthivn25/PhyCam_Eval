"""Unit tests for phycam_eval.eval.coco helper functions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from phycam_eval.eval.coco import load_coco_images, run_yolo


def _make_tiny_coco(root):
    img_dir = root / "images" / "val2017"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for idx in [1, 2, 3]:
        img_path = img_dir / f"{idx:012d}.jpg"
        Image.new("RGB", (8, 8), color=(idx * 30, idx * 20, idx * 10)).save(img_path)

    with open(ann_dir / "instances_val2017.json", "w") as f:
        json.dump(
            {
                "images": [
                    {"id": 1, "file_name": "000000000001.jpg", "width": 8, "height": 8},
                    {"id": 2, "file_name": "000000000002.jpg", "width": 8, "height": 8},
                    {"id": 3, "file_name": "000000000003.jpg", "width": 8, "height": 8},
                ],
                "annotations": [],
                "categories": [],
            },
            f,
        )


def test_load_coco_images_supports_none_max_images(tmp_path):
    root = tmp_path / "coco"
    _make_tiny_coco(root)

    image_ids, images, metas, _coco_data = load_coco_images(
        coco_root=root,
        split="val2017",
        max_images=None,
        image_size=8,
    )

    assert image_ids == [1, 2, 3]
    assert len(images) == 3
    assert images[0].shape == (3, 8, 8)
    assert metas[1]["id"] == 2


def test_load_coco_images_applies_image_offset_before_max_images(tmp_path):
    root = tmp_path / "coco"
    _make_tiny_coco(root)

    image_ids, images, metas, _coco_data = load_coco_images(
        coco_root=root,
        split="val2017",
        max_images=1,
        image_size=8,
        image_offset=1,
    )

    assert image_ids == [2]
    assert len(images) == 1
    assert metas[0]["id"] == 2


def test_run_yolo_forwards_conf_and_iou():
    calls = []

    class DummyYOLO:
        def __call__(self, _img_hwc, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    image = np.zeros((3, 8, 8), dtype=np.float32)
    preds = run_yolo(DummyYOLO(), [image], conf=0.12, iou=0.77, device="cpu")

    assert len(calls) == 1
    assert calls[0]["conf"] == 0.12
    assert calls[0]["iou"] == 0.77
    assert preds[0]["boxes"].shape == (0, 4)
    assert preds[0]["labels"].shape == (0,)
    assert preds[0]["scores"].shape == (0,)
