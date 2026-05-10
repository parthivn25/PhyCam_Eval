"""Protocol regression tests for paper sweep scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _load_report_script(name: str):
    path = Path(__file__).resolve().parents[1] / "report" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_gamma_bootstrap_resamples_with_unique_image_ids(monkeypatch):
    mod = _load_script("run_gamma_bootstrap.py")

    seen_batches = []

    def fake_compute_map(preds, targets):
        pred_ids = [p["image_id"] for p in preds]
        target_ids = [t["image_id"] for t in targets]
        assert pred_ids == target_ids
        assert pred_ids == list(range(len(pred_ids)))
        seen_batches.append(pred_ids)
        return {"map50": 1.0}

    monkeypatch.setattr(mod, "compute_map", fake_compute_map)

    preds = [
        {"image_id": 10, "boxes": np.zeros((0, 4)), "labels": np.zeros((0,)), "scores": np.zeros((0,))},
        {"image_id": 20, "boxes": np.zeros((0, 4)), "labels": np.zeros((0,)), "scores": np.zeros((0,))},
    ]
    targets = [
        {"image_id": 10, "boxes": np.zeros((0, 4)), "labels": np.zeros((0,))},
        {"image_id": 20, "boxes": np.zeros((0, 4)), "labels": np.zeros((0,))},
    ]

    out = mod.bootstrap_map(preds, targets, [10, 20], 3, np.random.default_rng(0))

    assert out.tolist() == [1.0, 1.0, 1.0]
    assert len(seen_batches) == 3


def test_rerun_defocus_validation_accepts_normalized_sensitivity(tmp_path):
    mod = _load_report_script("rerun_all.py")
    out = tmp_path
    (out / "results.json").write_text(
        json.dumps({
            "baseline_map50": 0.33,
            "sweep": [{"alpha": 3.0, "map50": 0.33, "sensitivity": 1.0}],
        })
    )

    assert mod._validate_defocus(out) == []
