import itertools
import os
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from qcse.cli import parser

SWEEP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/training_sweep.py"))
SEMANTIC_SWEEP = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/semantic_cluster_sweep.py")
)
ATTENTION_CV = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/attention_cross_validation.py")
)
ATTENTION_REPORT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/report_attention_cv.py")
)


def test_validation_top1_is_primary_selection_metric():
    history = [
        {
            "epoch": 1,
            "validation": {"cosine_top1": 0.10, "cosine_top5": 0.30, "cross_entropy": 0.10},
        },
        {
            "epoch": 2,
            "validation": {"cosine_top1": 0.20, "cosine_top5": 0.20, "cross_entropy": 0.20},
        },
        {
            "epoch": 3,
            "validation": {"cosine_top1": 0.20, "cosine_top5": 0.25, "cross_entropy": 0.30},
        },
    ]
    assert SEMANTIC_SWEEP["best_validation_row"](history)["epoch"] == 3
    assert ATTENTION_CV["best_validation_row"](history)["epoch"] == 3

    setups = [
        {
            "setup": {"name": "low-ce"},
            "validation": {
                "cosine_top1_mean": 0.10,
                "cosine_top5_mean": 0.50,
                "cross_entropy_mean": 0.10,
            },
        },
        {
            "setup": {"name": "high-top1"},
            "validation": {
                "cosine_top1_mean": 0.20,
                "cosine_top5_mean": 0.20,
                "cross_entropy_mean": 0.20,
            },
        },
    ]
    assert ATTENTION_REPORT["best_validation_row"](setups)["setup"]["name"] == "high-top1"


def test_sweep_covers_45_causal_configurations_and_two_fold_cv(tmp_path):
    jobs = SWEEP["experiments"]()
    assert len(jobs) == 45
    assert [job["experiment_id"] for job in jobs] == list(range(45))
    assert Counter(job["objective"] for job in jobs) == {"causal": 45}
    assert {tuple(job["datasets"]) for job in jobs} == {("phrases", "cleaned")}
    assert {job["cleaning"] for job in jobs} == {"dedupe"}
    assert {job["sampling"] for job in jobs} == {"uniform", "balanced"}
    commands = set()
    for job in jobs:
        command = SWEEP["command"](job, "cuda", tmp_path)
        args = parser().parse_args(command[3:])
        assert args.epochs == 50
        assert args.samples_per_epoch == 5000
        assert args.eval_examples == 2048
        assert args.final_eval_examples == 10000
        assert args.folds == 2
        assert args.val_fraction == 0.2
        assert args.device == "cuda"
        assert args.max_examples is None
        commands.add(tuple(command))
    assert len(commands) == 45
    sampled = [job for job in jobs if job["max_sentences"] is not None]
    assert len(sampled) == 6
    assert {(job["max_sentences"], job["sampling"]) for job in sampled} == set(
        itertools.product((5000, 20000, 50000), ("uniform", "balanced"))
    )
    grid = [
        job
        for job in jobs
        if (job["layers"], job["batch_size"]) == (8, 64) and job["max_sentences"] is None
    ]
    assert len(grid) == 27
    assert {(job["alpha"], job["learning_rate"], job["window"]) for job in grid} == set(
        itertools.product((0.1, 1.0, 3.0), (0.0001, 0.0003, 0.001), (2, 4, 8))
    )
    comparisons = [job for job in jobs if job not in grid and job not in sampled]
    assert len(comparisons) == 12
    for job in comparisons:
        assert (job["alpha"], job["learning_rate"]) == (1.0, 0.0003)
        assert job["window"] in (2, 8)


def test_smoke_covers_extremes_with_small_budgets(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["training_sweep.py", "--smoke"])
    SWEEP["main"]()
    commands = [parser().parse_args(args[3:]) for args in calls]
    assert len(commands) == 7
    assert {args.alpha for args in commands} == {0.1, 1.0, 3.0}
    assert {args.learning_rate for args in commands} == {0.0001, 0.0003, 0.001}
    assert {args.window for args in commands} == {2, 4, 8}
    assert all(
        args.max_sentences == 16 and args.epochs == 2 and args.samples_per_epoch == 32
        for args in commands
    )


def test_nine_groups_execute_every_experiment_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    for group in range(9):
        monkeypatch.setattr(
            sys, "argv", ["training_sweep.py", "--group-id", str(group), "--output", str(tmp_path)]
        )
        SWEEP["main"]()
        assert len(calls) == (group + 1) * 5
    paths = [Path(args[args.index("--output") + 1]).name for args in calls]
    assert paths == [f"experiment-{i:03d}" for i in range(45)]


def test_group_stops_on_failed_experiment(monkeypatch):
    calls = []

    def fail(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["training_sweep.py", "--group-id", "0"])
    with pytest.raises(subprocess.CalledProcessError):
        SWEEP["main"]()
    assert len(calls) == 1


def test_submission_submits_single_independent_array(tmp_path):
    # Exercise the real shell wrapper without contacting a Slurm scheduler.
    binary = tmp_path / "sbatch"
    log = tmp_path / "calls"
    binary.write_text(
        "#!/bin/bash\n"
        'echo "$*" >> "$SBATCH_TEST_LOG"\n'
        'count=$(wc -l < "$SBATCH_TEST_LOG")\n'
        'echo "$((1000 + count));cluster"\n'
    )
    binary.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", str(root / "scripts/submit_sweep.sh"), "--partition=prod10"],
        check=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "SBATCH_TEST_LOG": str(log)},
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert "--dependency" not in lines[0]
    assert "--array=0-44%10" in lines[0]
    assert "slurm-dgx-a100-10gb-qcse.sbatch" in lines[0]


def test_dgx_launcher_submits_three_independent_arrays(tmp_path):
    binary = tmp_path / "sbatch"
    log = tmp_path / "calls"
    binary.write_text(
        "#!/bin/bash\n"
        'echo "$*" >> "$SBATCH_TEST_LOG"\n'
        'count=$(wc -l < "$SBATCH_TEST_LOG")\n'
        'echo "$((2000 + count));cluster"\n'
    )
    binary.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", str(root / "scripts/submit_dgx_a100_10gb.sh")],
        check=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "SBATCH_TEST_LOG": str(log)},
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert all("--partition=dgx-a100" in line for line in lines)
    assert all("--gres=gpu:nvidia_a100_1g.10gb:1" in line for line in lines)
    assert all("--dependency" not in line for line in lines)
    assert "--array=0-44%10" in lines[0]
    assert "--array=0-9%10" in lines[1]
    assert "--array=0-1%2" in lines[2]
