"""
Unit tests for phycam_eval.eval.metrics
"""

import numpy as np
import pytest

from phycam_eval.eval.metrics import (
    COCO80_TO_91,
    compute_map,
    compute_map_ci,
    yolo_to_coco_category_ids,
)


def test_coco_class_mapping_matches_known_reference():
    mapped = yolo_to_coco_category_ids([0, 11, 23, 79])
    assert mapped.tolist() == [1, 13, 25, 90]


def test_coco_class_mapping_rejects_out_of_range_ids():
    with pytest.raises(ValueError):
        yolo_to_coco_category_ids([-1, 80])


def test_coco_class_mapping_round_trips_empty_input():
    mapped = yolo_to_coco_category_ids(np.array([], dtype=int))
    assert mapped.shape == (0,)
    assert mapped.dtype == COCO80_TO_91.dtype


def test_compute_map_honors_custom_iou_thresholds():
    target = {
        "image_id": 1,
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }
    pred = {
        "image_id": 1,
        "boxes": np.array([[0.0, 0.0, 6.0, 10.0]], dtype=np.float32),
        "scores": np.array([0.9], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
    }

    passes_at_50 = compute_map([pred], [target], iou_thresholds=[0.50])
    fails_at_75 = compute_map([pred], [target], iou_thresholds=[0.75])

    assert passes_at_50["map50_95"] > 0.99
    assert fails_at_75["map50_95"] == pytest.approx(0.0)


def test_compute_map_rejects_duplicate_image_records():
    pred = {
        "image_id": 1,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }
    target = {
        "image_id": 1,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }

    with pytest.raises(ValueError, match="unique image_id"):
        compute_map([pred, pred.copy()], [target])


def test_compute_map_ci_requires_paired_prediction_records():
    predictions = [
        {
            "image_id": 1,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }
    ]
    targets = [
        {
            "image_id": 1,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        },
        {
            "image_id": 2,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        },
    ]

    with pytest.raises(ValueError, match="every target image_id"):
        compute_map_ci(predictions, targets, n_bootstrap=2)
