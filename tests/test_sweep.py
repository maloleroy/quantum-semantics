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


def test_sweep_covers_75_causal_configurations_and_five_fold_cv(tmp_path):
    jobs = SWEEP["experiments"]()
    assert len(jobs) == 75
    assert [job["experiment_id"] for job in jobs] == list(range(75))
    assert Counter(job["objective"] for job in jobs) == {"causal": 75}
    assert {tuple(job["datasets"]) for job in jobs} == {("phrases", "cleaned")}
    assert {job["cleaning"] for job in jobs} == {"dedupe"}
    assert {job["sampling"] for job in jobs} == {"uniform", "balanced"}
    commands = set()
    for job in jobs:
        command = SWEEP["command"](job, "cuda", tmp_path)
        args = parser().parse_args(command[3:])
        assert args.epochs == 150
        assert args.samples_per_epoch == 5000
        assert args.eval_examples == 2048
        assert args.folds == 5
        assert args.device == "cuda"
        assert args.max_examples is None
        commands.add(tuple(command))
    assert len(commands) == 75
    sampled = [job for job in jobs if job["max_sentences"] is not None]
    assert len(sampled) == 6
    assert {(job["max_sentences"], job["sampling"]) for job in sampled} == set(
        itertools.product((5000, 20000, 50000), ("uniform", "balanced"))
    )
    for objective in ("causal",):
        group = [job for job in jobs if job["objective"] == objective]
        grid = [
            job
            for job in group
            if (job["layers"], job["batch_size"]) == (8, 64) and job["max_sentences"] is None
        ]
        assert len(grid) == 45
        assert {(job["alpha"], job["learning_rate"], job["window"]) for job in grid} == set(
            itertools.product((0.1, 1.0, 3.0), (0.0001, 0.0003, 0.001), (2, 4, 6, 8, 12))
        )
        comparisons = [job for job in group if job not in grid and job not in sampled]
        assert len(comparisons) == 24
        for job in comparisons:
            assert (job["alpha"], job["learning_rate"]) == (1.0, 0.0003)
            assert any(
                reference["window"] == job["window"]
                and reference["alpha"] == job["alpha"]
                and reference["learning_rate"] == job["learning_rate"]
                for reference in grid
            )


def test_smoke_uses_new_grid_and_small_budgets(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["training_sweep.py", "--smoke"])
    SWEEP["main"]()
    commands = [parser().parse_args(args[3:]) for args in calls]
    assert len(commands) == 11
    assert {args.alpha for args in commands} == {0.1, 1.0, 3.0}
    assert {args.learning_rate for args in commands} == {0.0001, 0.0003, 0.001}
    assert {args.window for args in commands} == {2, 4, 6, 8, 12}
    assert {(args.layers, args.batch_size) for args in commands} == {
        (8, 64),
        (2, 16),
        (2, 64),
        (8, 16),
        (8, 256),
        (64, 64),
        (64, 256),
    }
    assert all(
        args.max_sentences == 16 and args.epochs == 2 and args.samples_per_epoch == 32
        for args in commands
    )


def test_fifteen_groups_execute_every_experiment_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    for group in range(15):
        monkeypatch.setattr(
            sys, "argv", ["training_sweep.py", "--group-id", str(group), "--output", str(tmp_path)]
        )
        SWEEP["main"]()
        assert len(calls) == (group + 1) * 5
    paths = [Path(args[args.index("--output") + 1]).name for args in calls]
    assert paths == [f"experiment-{i:03d}" for i in range(75)]


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


def test_submission_chains_corresponding_array_tasks(tmp_path):
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
    assert len(lines) == 2
    assert "--dependency" not in lines[0]
    assert "--dependency=aftercorr:1001" in lines[1]
    assert "--array=0-9%10" in lines[0]
    assert "--array=0-4%10" in lines[1]
    for wave, line in enumerate(lines):
        assert line.endswith(f"slurm-prod10.sbatch {wave}")
