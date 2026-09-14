import json
import runpy
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("encoding", ["qcse", "classical", "legacy-semantic"])
def test_training_resume_matches_uninterrupted_run(encoding, tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "semantic_experiment.py"

    def run(*options):
        monkeypatch.setattr(sys, "argv", [str(script), *options])
        runpy.run_path(str(script), run_name="__main__")

    common = [
        "--model",
        "semantic" if encoding == "legacy-semantic" else "attention",
        "--encoding",
        "qcse" if encoding == "legacy-semantic" else encoding,
        "--qubits",
        "2",
        "--embedding-dim",
        "4",
        "--window",
        "3",
        "--samples-per-epoch",
        "17",
        "--batch-size",
        "8",
    ]
    run(*common, "--epochs", "1", "--output", str(tmp_path / "resumed"))
    resumed = next((tmp_path / "resumed").iterdir())
    assert not (resumed / "test.json").exists()
    if encoding == "legacy-semantic":
        checkpoint = resumed / "checkpoint.pt"
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        for key in ("model", "encoding", "context_alpha"):
            saved["settings"].pop(key)
        torch.save(saved, checkpoint)
    # Fresh-run defaults and batch overrides must not change the saved fit.
    run("--resume", str(resumed), "--epochs", "2", "--batch-size", "32", "--evaluate-test")
    run(*common, "--epochs", "2", "--output", str(tmp_path / "full"), "--evaluate-test")
    full = next((tmp_path / "full").iterdir())
    a, b = [
        torch.load(path / "checkpoint.pt", weights_only=True, map_location="cpu")
        for path in (resumed, full)
    ]
    assert a["model_config"] == b["model_config"]
    assert a["rng_state"] == b["rng_state"]
    for name, weights in a["model"].items():
        torch.testing.assert_close(weights, b["model"][name], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][name], rtol=0, atol=0)
    assert all(int(state["step"]) == 6 for state in a["optimizer"]["state"].values())
    for split in ("train_ids", "val_ids", "test_ids"):
        assert a[split] == b[split]
    for left, right in zip(a["history"], b["history"], strict=True):
        for split in ("train", "validation"):
            assert left[split]["cross_entropy"] == pytest.approx(right[split]["cross_entropy"])
    test = json.loads((resumed / "test.json").read_text())
    assert test["examples"] == len(a["test_ids"])
    assert test["epoch"] == 2
    assert len(json.loads((resumed / "predictions.json").read_text())) == 8
