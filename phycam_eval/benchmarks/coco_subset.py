"""
COCO validation subset dataset adapter.

Downloads / loads a subset of COCO val2017 and wraps it as a PyTorch Dataset
that optionally applies a degradation operator on-the-fly.

Usage
-----
    dataset = COCOSubset(
        root="data/coco",
        split="val2017",
        max_images=5000,
        degradation=DefocusOperator(alpha=1.5),
    )
    loader = DataLoader(dataset, batch_size=8)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalisation (used by YOLO / detection models)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class COCOSubset(Dataset):
    """
    COCO val2017 subset dataset.

    Parameters
    ----------
    root : str | Path
        Directory containing 'images/val2017/' and 'annotations/' subdirs.
        If the data does not exist, call COCOSubset.download(root).
    split : str
        "val2017" (default) or "train2017"
    max_images : int | None
        If set, truncate to the first max_images images (sorted by image_id).
    degradation : callable | None
        Applied to each image tensor (C, H, W) before returning.
    image_size : int
        Resize images to (image_size, image_size). Default 640 (YOLO standard).
    normalize : bool
        Apply ImageNet mean/std normalisation. Default True.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val2017",
        max_images: int | None = None,
        degradation: Callable | None = None,
        image_size: int = 640,
        normalize: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.degradation = degradation

        # Load COCO annotations
        ann_file = self.root / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Run COCOSubset.download('{root}') or set root to your COCO directory."
            )

        with open(ann_file) as f:
            coco_data = json.load(f)

        # Build image_id → filename and image_id → annotations maps
        self._images = {img["id"]: img for img in coco_data["images"]}
        self._annotations: dict[int, list] = {img_id: [] for img_id in self._images}
        for ann in coco_data["annotations"]:
            self._annotations[ann["image_id"]].append(ann)

        self._image_ids = sorted(self._images.keys())
        if max_images is not None:
            self._image_ids = self._image_ids[:max_images]

        self._resize = transforms.Resize((image_size, image_size))
        self._to_tensor = transforms.ToTensor()  # → [0, 1]
        self._normalize = (
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
            if normalize else None
        )

        self._img_dir = self.root / "images" / split

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self._image_ids[idx]
        meta = self._images[image_id]

        img_path = self._img_dir / meta["file_name"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self._to_tensor(self._resize(image))  # (3, H, W) float32 in [0, 1]

        if self.degradation is not None:
            image_tensor = self.degradation(image_tensor)
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.as_tensor(np.asarray(image_tensor), dtype=torch.float32)
            else:
                image_tensor = image_tensor.to(dtype=torch.float32)

        if self._normalize is not None:
            image_tensor = self._normalize(image_tensor)

        # Build target dict compatible with pycocotools / torchvision
        anns = self._annotations[image_id]
        boxes, labels, areas, iscrowds = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowds.append(ann.get("iscrowd", 0))

        target = {
            "image_id": torch.tensor(image_id),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowds, dtype=torch.long),
        }
        return {"image": image_tensor, "annotations": target}

    @staticmethod
    def download(root: str | Path, split: str = "val2017") -> None:
        """
        Download COCO val2017 images and annotations to `root`.

        Requires ~1 GB for val2017 (5K images) + ~241 MB for annotations.
        """
        import subprocess
        root = Path(root)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)

        base_url = "http://images.cocodataset.org"
        files = {
            "images": f"{base_url}/zips/{split}.zip",
            "annotations": f"{base_url}/annotations/annotations_trainval2017.zip",
        }

        for key, url in files.items():
            dest = root / f"{key}.zip"
            if not dest.exists():
                print(f"Downloading {url} → {dest}")
                subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
                subprocess.run(["unzip", "-q", str(dest), "-d", str(root)], check=True)

        print(f"COCO {split} ready at {root}")
