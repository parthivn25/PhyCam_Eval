"""
Unit tests for dataset/evaluation helpers in phycam_eval.benchmarks
"""

from __future__ import annotations

import json

import torch
from PIL import Image

from phycam_eval.benchmarks.coco_subset import COCOSubset


def test_coco_subset_applies_degradation_before_normalization(tmp_path):
    root = tmp_path / "coco"
    img_dir = root / "images" / "val2017"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    image_path = img_dir / "000000000001.jpg"
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(image_path)

    with open(ann_dir / "instances_val2017.json", "w") as f:
        json.dump(
            {
                "images": [
                    {"id": 1, "file_name": image_path.name, "width": 8, "height": 8}
                ],
                "annotations": [],
            },
            f,
        )

    seen = {}

    def degradation(x):
        seen["min"] = float(x.min())
        seen["max"] = float(x.max())
        return x

    dataset = COCOSubset(
        root=root,
        max_images=1,
        degradation=degradation,
        image_size=8,
        normalize=True,
    )

    sample = dataset[0]["image"]

    assert 0.0 <= seen["min"] <= 1.0
    assert 0.0 <= seen["max"] <= 1.0
    assert isinstance(sample, torch.Tensor)
    assert sample.min().item() < 0.0


def test_coco_subset_scales_targets_with_resized_image(tmp_path):
    root = tmp_path / "coco"
    img_dir = root / "images" / "val2017"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    image_path = img_dir / "000000000001.jpg"
    Image.new("RGB", (10, 20), color=(0, 0, 0)).save(image_path)

    with open(ann_dir / "instances_val2017.json", "w") as f:
        json.dump(
            {
                "images": [
                    {"id": 1, "file_name": image_path.name, "width": 10, "height": 20}
                ],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "area": 12.0,
                        "iscrowd": 1,
                    }
                ],
            },
            f,
        )

    dataset = COCOSubset(root=root, max_images=1, image_size=100, normalize=False)
    target = dataset[0]["annotations"]

    torch.testing.assert_close(
        target["boxes"],
        torch.tensor([[10.0, 10.0, 40.0, 30.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(target["area"], torch.tensor([600.0]))
    assert target["iscrowd"].tolist() == [1]
