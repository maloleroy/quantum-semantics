"""Verify saved result/array agreement and preserve final artifact provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import write_json


def audit():
    root = Path(__file__).parent
    results = root / "results"
    checked = 0
    for stage in ["main", "phrases", "full_classical", "calibration", "born_control"]:
        record = json.loads((results / stage / "test_results.json").read_text())["metrics"]
        lock = json.loads((results / stage / "selection_lock.json").read_text())["jobs"]
        assert set(lock) <= set(record), stage
        for key, metrics in record.items():
            arrays = np.load(results / stage / f"{key}_test.npz")
            for metric in ["nll", "top1", "top5"]:
                assert np.isclose(arrays[metric].mean(), metrics[metric], atol=1e-12), (
                    stage,
                    key,
                    metric,
                )
            assert len(arrays["targets"]) == metrics["count"]
            checked += 1
        for key in lock:
            trained = json.loads((results / stage / f"{key}.json").read_text())
            best = min(v["validation_nll"] for v in trained["history"])
            assert np.isclose(best, trained["validation"]["nll"], atol=1e-8), (stage, key)
            weights = torch.load(results / stage / f"{key}.pt", weights_only=True)
            assert all(torch.isfinite(v).all() for v in weights.values()), (stage, key)
    splits = {}
    for corpus in ["tatoeba_data", "phrases_data", "tatoeba_full_data"]:
        data = np.load(results / corpus / "dataset.npz")
        groups = [set(data[f"{s}_groups"]) for s in ["train", "validation", "test"]]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), corpus
        manifest = json.loads((results / corpus / "dataset.json").read_text())
        source = Path(manifest["source"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
        assert data["transitions"].sum() == manifest["partitions"]["train"]["full_targets"]
        splits[corpus] = {
            "disjoint_groups": True,
            "source_hash_matches": True,
            "training_transition_count": int(data["transitions"].sum()),
        }
    files = list(root.glob("*.py")) + list(root.glob("*.md"))
    files += list(results.glob("*.png")) + [Path("output/pdf/hamiltonian_encoder_research.pdf")]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    write_json(results / "source_manifest.json", manifest)
    result = {
        "test_metric_records_checked": checked,
        "checkpoint_selection_verified": True,
        "all_saved_weights_finite": True,
        "datasets": splits,
    }
    write_json(results / "audit.json", result)
    return result


if __name__ == "__main__":
    print(audit())
