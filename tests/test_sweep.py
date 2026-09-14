import os
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from qcse.cli import parser

SWEEP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/training_sweep.py"))


def test_sweep_covers_150_valid_distinct_ten_epoch_jobs(tmp_path):
    jobs = SWEEP["experiments"]()
    assert len(jobs) == 150
    assert [job["experiment_id"] for job in jobs] == list(range(150))
    assert Counter(job["objective"] for job in jobs) == {"causal": 75, "cbow": 75}
    assert {len(job["datasets"]) for job in jobs} == {1, 2, 3}
    assert len({tuple(job["datasets"]) for job in jobs}) == 7
    assert {job["cleaning"] for job in jobs} == {"basic", "dedupe", "strict"}
    assert {job["sampling"] for job in jobs} == {"uniform", "balanced"}
    commands = set()
    for job in jobs:
        command = SWEEP["command"](job, "cuda", tmp_path)
        args = parser().parse_args(command[3:])
        assert args.epochs == 10
        assert args.device == "cuda"
        assert args.max_examples == 512
        commands.add(tuple(command))
    assert len(commands) == 150


def test_thirty_groups_execute_every_experiment_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    for group in range(30):
        monkeypatch.setattr(
            sys, "argv", ["training_sweep.py", "--group-id", str(group), "--output", str(tmp_path)]
        )
        SWEEP["main"]()
        assert len(calls) == (group + 1) * 5
    paths = [Path(args[args.index("--output") + 1]).name for args in calls]
    assert paths == [f"experiment-{i:03d}" for i in range(150)]


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
    assert len(lines) == 3
    assert "--dependency" not in lines[0]
    assert "--dependency=aftercorr:1001" in lines[1]
    assert "--dependency=aftercorr:1002" in lines[2]
    for wave, line in enumerate(lines):
        assert line.endswith(f"slurm-prod10.sbatch {wave}")
