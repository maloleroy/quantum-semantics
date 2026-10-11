"""Fixed-budget scoring, retained controls and the executable research plan."""

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_robust_benchmark import small_study as small_study

from qcse import benchmark


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fixed", [False, True])
def test_flat_monitor_stops_default_but_fixed_scores_final_state(small_study, monkeypatch, fixed):
    original_root, original = small_study
    root = original_root / "budget"
    settings = dict(
        original["settings"],
        objectives=["causal"],
        arms=["mlp"],
        seeds=[42],
        epochs=4,
        patience=1,
        min_epochs=1,
        fixed_epochs=fixed,
    )
    manifest = benchmark.prepare(root, settings)
    original_metrics = benchmark.metrics

    def flat_monitor(values):
        if len(values) == settings["eval_examples"]:
            return dict(nll=1.0, top1=0.1, top5=0.2, examples=len(values))
        return original_metrics(values)

    monkeypatch.setattr(benchmark, "metrics", flat_monitor)
    for i, task in enumerate(manifest["tasks"]):
        output = root / "cv" / f"task-{i:04d}"
        result = benchmark.fit(root, task, output)
        assert result["stopped_epoch"] == (4 if fixed else 2)
        assert result["selected_epoch"] == (4 if fixed else 1)
        benchmark.validate_result(root, manifest, task, output, True)
        if fixed:
            selected = torch.load(output / "selected.pt", weights_only=False)["model"]
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)["model"]
            for key in selected:
                torch.testing.assert_close(selected[key], checkpoint[key], rtol=0, atol=0)
    chosen = benchmark.selection(root)
    assert chosen["refits"][0]["epochs"] == (4 if fixed else 1)
    task = chosen["refits"][0]
    result = benchmark.fit(root, task, root / "refit/task-0000", refit_epochs=task["epochs"])
    assert result["selected_epoch"] == task["epochs"]
    if fixed:
        output = root / "cv/task-0000"
        result = json.loads((output / "result.json").read_text())
        result.update(selected_epoch=2, stopped_epoch=2, optimizer_updates=6, training_draws=18)
        (output / "result.json").write_text(json.dumps(result))
        with pytest.raises(ValueError, match="budget/update"):
            benchmark.validate_result(root, manifest, manifest["tasks"][0], output, True)


def test_refit_controls_survive_family_selection_without_duplicates(small_study):
    original_root, original = small_study
    root = original_root / "controls"
    arms = ["hybrid-crz-chain", "hybrid-identity"]
    settings = dict(
        original["settings"],
        objectives=["causal"],
        arms=arms,
        refit_controls=arms,
        seeds=[42],
        fixed_epochs=True,
    )
    manifest = benchmark.prepare(root, settings)
    for i, task in enumerate(manifest["tasks"]):
        benchmark.fit(root, task, root / "cv" / f"task-{i:04d}")
    selected = benchmark.selection(root)
    assert len(selected["refits"]) == 2
    assert {task["arm"]["name"] for task in selected["refits"]} == set(arms)
    assert all(task["epochs"] == 2 for task in selected["refits"])


def test_screen_blocks_use_long_budget_and_known_controls():
    repository = Path(__file__).resolve().parents[1]
    cli = script("robust_benchmark.py").parser()
    blocks = {
        "reference": 18,
        "hybrid-circuits": 10,
        "trainability": 15,
        "qcse-circuits": 11,
        "context": 8,
        "window-2": 5,
        "window-8": 5,
        "cleaned": 5,
        "phrases": 5,
        "cbow": 7,
    }
    known = {arm["name"] for arm in benchmark.configurations(True)}
    for block, count in blocks.items():
        command = subprocess.check_output(
            ["bash", "scripts/prepare_benchmark_screen.sh", block, "--dry-run"],
            cwd=repository,
            text=True,
        )
        args = cli.parse_args(shlex.split(command)[2:])
        assert len(args.arms) == len(set(args.arms)) == count
        assert set(args.arms) <= known
        assert args.epochs == 100 and args.fixed_epochs and args.samples_per_epoch == 20000
        assert args.folds == 3 and args.seeds == [42] and args.final_eval_sentences == 0
        assert args.objectives == (["cbow"] if block == "cbow" else ["causal"])
        if block in ("phrases", "cleaned"):
            assert args.datasets == [block]
    assert sum(blocks.values()) * 3 == 267


def test_fixed_compute_estimate_uses_requested_budget_and_full_final_counts(tmp_path, monkeypatch):
    module = script("estimate_benchmark_compute.py")
    task = dict(objective="cbow", arm=dict(name="model", family="model"), fold=0)
    settings = dict(
        epochs=100,
        fixed_epochs=True,
        samples_per_epoch=20000,
        batch_size=64,
        microbatch_size=8,
        threads=4,
        objectives=["cbow"],
        folds=2,
        eval_examples=2048,
        seeds=[42],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                fingerprint="study", settings=settings, tasks=[task], code_sha256={"source": "hash"}
            )
        )
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            dict(
                manifest="study",
                objective="cbow",
                device="cuda",
                batch=64,
                microbatch=8,
                threads=4,
                effective_source_sha256={"source": "hash"},
                measurements=[
                    dict(
                        arm=task["arm"],
                        trainable_parameters=1,
                        training_batch_seconds=1,
                        evaluation_batch_seconds=2,
                    )
                ],
            )
        )
    )
    np.savez(
        tmp_path / "cbow.npz",
        val_eval_0=np.arange(300),
        val_eval_1=np.arange(320),
        test_eval=np.arange(100),
    )
    output = tmp_path / "compute.json"
    monkeypatch.setattr(
        sys, "argv", ["estimate", str(tmp_path), str(profile), "--output", str(output)]
    )
    module.main()
    result = json.loads(output.read_text())
    assert result["reference_objective"] == "cbow"
    assert [row["trained_epochs"] for row in result["scenarios"]] == [100]
    scenario = result["scenarios"][0]
    expected_seconds = 313 * 100 + 101 * 2048 * 2 / 64 + 300 * 2 / 64 + 20
    assert scenario["cv_reference_hours"] == expected_seconds / 3600
    assert scenario["configurations"][0]["max_fit_reference_hours"] == expected_seconds / 3600
    original_profile = json.loads(profile.read_text())
    for key, value in (
        ("microbatch", 16),
        ("threads", 8),
        ("effective_source_sha256", {"source": "changed"}),
        ("objective", "causal"),
        ("measurements", []),
    ):
        profile.write_text(json.dumps(dict(original_profile, **{key: value})))
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2


def test_dimension_pilot_records_separate_cuda_mechanism_probe(tmp_path, monkeypatch):
    module = script("run_robust_benchmark_job.py")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(tasks=[dict(objective="causal", arm=dict(qubits=2))]))
    )
    commands = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(sys, "argv", ["job", "pilot", str(tmp_path), "--task-id", "0"])
    module.main()
    assert len(commands) == 7
    probe = commands[-2]
    assert probe[1] == "scripts/probe_barren_plateaus.py"
    assert probe[probe.index("--device") + 1] == "cuda"
    data = json.loads(next((tmp_path / "job-timings").glob("*.json")).read_text())
    assert data["phases"][-2]["name"] == "trainability_probe"
