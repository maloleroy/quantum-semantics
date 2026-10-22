import importlib.util
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
        "--eval-examples",
        "2",
        "--final-eval-examples",
        "2",
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
    assert all(row["validation"]["examples"] == min(2, len(a["val_ids"])) for row in a["history"])
    test = json.loads((resumed / "test.json").read_text())
    assert test["examples"] == min(2, len(a["test_ids"]))
    assert test["epoch"] == 2
    assert len(json.loads((resumed / "predictions.json").read_text())) == min(
        8, 2, len(a["test_ids"])
    )


def test_early_stopping_resume_keeps_chronological_checkpoint_and_selected_state(
    tmp_path, monkeypatch
):
    script = Path(__file__).resolve().parents[1] / "scripts/semantic_experiment.py"
    spec = importlib.util.spec_from_file_location("focused_semantic", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def constant_scores(model, contexts, targets, ids, batch_size):
        return dict(
            cross_entropy=1.0,
            top1=0.5,
            top5=0.8,
            cosine_top1=0.5,
            cosine_top5=0.8,
            examples=len(ids),
        )

    monkeypatch.setattr(module, "evaluate", constant_scores)

    def run(*options):
        monkeypatch.setattr(sys, "argv", [str(script), *options])
        module.main()

    common = [
        "--qubits",
        "2",
        "--embedding-dim",
        "4",
        "--window",
        "3",
        "--samples-per-epoch",
        "17",
        "--eval-examples",
        "2",
        "--final-eval-examples",
        "2",
        "--batch-size",
        "8",
        "--early-stopping-patience",
        "2",
        "--min-epochs",
        "2",
    ]
    run(*common, "--epochs", "1", "--output", str(tmp_path / "resumed"))
    resumed = next((tmp_path / "resumed").iterdir())
    # Resume uses the saved stopping protocol even without repeating CLI flags.
    run("--resume", str(resumed), "--epochs", "5", "--evaluate-test")
    run(*common, "--epochs", "5", "--output", str(tmp_path / "full"), "--evaluate-test")
    full = next((tmp_path / "full").iterdir())
    a, b = [torch.load(p / "checkpoint.pt", weights_only=True) for p in (resumed, full)]
    assert a["history"][-1]["epoch"] == b["history"][-1]["epoch"] == 2
    assert a["selection"]["epoch"] == 0 and a["selection"]["stale_epochs"] == 2
    assert a["rng_state"] == b["rng_state"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    for index, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][name], rtol=0, atol=0)
    selected = torch.load(resumed / "selected.pt", weights_only=True)
    assert selected["epoch"] == json.loads((resumed / "test.json").read_text())["epoch"] == 0
    assert any(not torch.equal(a["model"][k], selected["model"][k]) for k in a["model"])
    for name in selected["model"]:
        torch.testing.assert_close(selected["model"][name], a["selection"]["model"][name])
