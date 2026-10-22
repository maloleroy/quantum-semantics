import json

import numpy as np
import pytest

from qcse.cli import parser, run
from qcse.outputs import new_run_directory, save_npz
from qcse.training import load_run


def test_new_runs_and_continuation_keep_prior_results(tmp_path):
    data = tmp_path / "sentences.csv"
    data.write_text("a b c\nb c a\nc a b\na a c\n")
    root = tmp_path / "outputs"
    command = ["train", "--data", str(data), "--output", str(root), "--epochs", "1"]
    run(parser().parse_args(command))
    first = next(root.iterdir())
    original = (first / "run.npz").read_bytes()
    run(parser().parse_args(command))
    assert len(list(root.iterdir())) == 2
    assert (first / "run.npz").read_bytes() == original
    assert json.loads((first / "run_info.json").read_text())["status"] == "complete"
    run(
        parser().parse_args(
            ["continue", str(first / "run.npz"), "--epochs", "1", "--output", str(root)]
        )
    )
    continued = next(root.glob("continue-*"))
    assert (first / "run.npz").read_bytes() == original
    saved = load_run(continued / "run.npz")
    assert saved["state"]["epoch"] == 2
    assert saved["metadata"]["provenance"]["sources"][0]["sha256"]
    run(parser().parse_args(["continue", str(continued / "run.npz"), "--epochs", "1"]))
    assert len(list(root.iterdir())) == 3
    assert load_run(continued / "run.npz")["state"]["epoch"] == 3


def test_failed_checkpoint_write_preserves_previous_archive(tmp_path, monkeypatch):
    path = tmp_path / "run.npz"
    save_npz(path, weights=np.ones(2))
    before = path.read_bytes()

    def interrupted(*args, **kwargs):
        args[0].write(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="disk full"):
        save_npz(path, weights=np.zeros(2))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_interruption_is_recorded(tmp_path):
    with pytest.raises(KeyboardInterrupt), new_run_directory(tmp_path, "train", {}) as output:
        raise KeyboardInterrupt
    assert json.loads((output / "run_info.json").read_text())["status"] == "interrupted"
