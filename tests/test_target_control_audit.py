"""Test pairing, split reconstruction and scheduler accounting, without fits."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "target_audit", SCRIPTS / "audit_target_controls.py"
        )
        assert spec is not None and spec.loader is not None
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def test_split_keys_cover_all_targets_once_and_preserve_group_boundaries():
    sentences = [["a", str(i), "b"] for i in range(50)]
    vocabulary = ["a", "b"] + [str(i) for i in range(50)]
    keys = module().split_keys(sentences, vocabulary)
    assert len(keys["test"]) == 20 and len(keys["val_0"]) == 16
    for left, right in [("train_0", "val_0"), ("development", "test")]:
        assert not set(keys[left][:, 0]) & set(keys[right][:, 0])
    all_keys = np.concatenate([keys[k] for k in ("train_0", "val_0", "test")])
    assert len(set(map(tuple, all_keys))) == 100
    for group, pos, target in all_keys:
        assert pos in (1, 2) and vocabulary[target] == sentences[group][pos]


def test_job_overlap_counts_end_before_simultaneous_start():
    jobs = [
        dict(started_at=a, finished_at=b) for a, b in [("01", "02"), ("01", "03"), ("02", "04")]
    ]
    assert module().peak_overlap(jobs) == 2


def test_effect_keeps_seed_pairing_and_sentence_macro_separate():
    values = np.array([[[[1.0], [9.0]], [[2.0], [6.0]]], [[[0.0], [3.0]], [[1.0], [3.0]]]])
    counts = np.array([1, 3])
    samples = np.array([[[2.0], [1.0]], [[3.0], [1.0]]])
    result = module().effect(values, counts, samples, samples, 0, 1)
    np.testing.assert_allclose(result["per_seed"], [[1.75], [1]])
    np.testing.assert_allclose(result["mean"], [1.375])
    np.testing.assert_allclose(result["sentence_macro"], [1.25])


def test_corrupted_export_is_rejected_before_any_statistics(tmp_path):
    audit = module()
    payload = tmp_path / "payload.json"
    payload.write_text("original")
    (tmp_path / "export.json").write_text(
        json.dumps(
            dict(included_sha256={"payload.json": audit.sha(payload.read_bytes())}, omitted=[])
        )
    )
    payload.write_text("modified")
    with pytest.raises(AssertionError, match="payload.json"):
        audit.audit(tmp_path, draws=10)
