import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from test_robust_benchmark import small_study as small_study


def load_probe():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_nlp_trainability.py"
    spec = importlib.util.spec_from_file_location("replicated_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_head_replicates_fix_weights_within_size_and_do_not_change_circuit_draws():
    module, _ = load_probe()
    contexts = torch.tensor([[17, 17, 1, 2], [3, 4, 5, 6]])
    targets = torch.tensor([3, 7])
    args = (17, 4, contexts, targets, 2, 2, 0.05, [42, 43, 44], "cpu")
    a, x = module.probe(*args, head_seed=619)
    b, y = module.probe(*args, head_seed=620)
    repeated, z = module.probe(*args, head_seed=619)
    assert a == repeated and a["common_state_sha256"] != b["common_state_sha256"]
    np.testing.assert_array_equal(x["gradients"], z["gradients"])
    np.testing.assert_array_equal(x["seeds"], y["seeds"])
    assert not np.array_equal(x["gradients"], y["gradients"])


def test_copied_probe_exports_every_batch_head_condition_from_training_only(small_study):
    root, manifest = small_study
    _, path = load_probe()
    copied = root / "copied_probe.py"
    shutil.copyfile(path, copied)
    output = root / "replicas"
    subprocess.run(
        [
            sys.executable,
            str(copied),
            str(root),
            "--device",
            "cpu",
            "--output",
            str(output),
            "--qubits",
            "2",
            "--layers",
            "2",
            "--init-scales",
            "0.05",
            "--seeds",
            "3",
            "--batch-seeds",
            "11",
            "12",
            "--head-seeds",
            "619",
            "620",
        ],
        check=True,
        capture_output=True,
    )
    run = next(output.glob("nlp-gradient-probe-*"))
    summary = json.loads((run / "summary.json").read_text())
    assert summary["manifest"] == manifest["fingerprint"]
    assert summary["code_sha256"] == manifest["code_sha256"]
    assert summary["analyzer_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert (run / "executed_probe.py").read_bytes() == copied.read_bytes()
    assert {(r["batch_seed"], r["head_seed"]) for r in summary["rows"]} == {
        (batch, head) for batch in (11, 12) for head in (619, 620)
    }
    for head in (619, 620):
        assert (
            len({r["common_state_sha256"] for r in summary["rows"] if r["head_seed"] == head}) == 1
        )
    with np.load(root / "causal.npz") as prepared:
        for file in run.glob("inputs-*.npz"):
            with np.load(file) as data:
                assert np.isin(data["ids"], prepared["train_0"]).all()
                assert not np.isin(data["ids"], prepared["test"]).any()
                np.testing.assert_array_equal(data["contexts"], prepared["contexts"][data["ids"]])
    assert len(list(run.glob("batch*-head*.npz"))) == 4
