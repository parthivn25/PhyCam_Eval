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

