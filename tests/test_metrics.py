"""
Unit tests for phycam_eval.eval.metrics
"""

import numpy as np
import pytest

from phycam_eval.eval.metrics import COCO80_TO_91, yolo_to_coco_category_ids


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

