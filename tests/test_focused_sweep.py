import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

FOCUSED = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/focused_sweep.py"))


def test_focused_manifest_has_two_repetitions_per_configuration():
    assert {family: len(configs) for family, configs in FOCUSED["CONFIGS"].items()} == {
        "causal": 8,
        "encoding": 3,
        "attention": 4,
    }
    for family, configs in FOCUSED["CONFIGS"].items():
        assert [FOCUSED["task_configuration"](family, 2 * i)[1] for i in range(len(configs))] == [
            0
        ] * len(configs)
        odd_repetitions = [
            FOCUSED["task_configuration"](family, 2 * i + 1)[1] for i in range(len(configs))
        ]
        assert odd_repetitions == [1] * len(configs)


def test_focused_commands_select_deterministic_configuration_and_seed(tmp_path):
    args = type(
        "Args",
        (),
        {
            "epochs": 50,
            "samples_per_epoch": 5000,
            "eval_examples": 2048,
            "final_eval_examples": 10000,
            "early_stopping_patience": 8,
            "min_epochs": 10,
            "threads": 1,
            "device": "cuda",
            "output": tmp_path,
        },
    )()
    command = FOCUSED["command"]("causal", 15, args)
    assert "--name" in command and command[command.index("--name") + 1] == "batch-256"
    assert "--repetition" in command and command[command.index("--repetition") + 1] == "1"
    assert "--seed" in command and command[command.index("--seed") + 1] == "43"
    assert "--early-stopping-patience" in command

    attention = FOCUSED["command"]("attention", 7, args)
    assert "--encoding" in attention
    assert "--no-quantum-attention" in attention


def test_submitter_waits_between_arrays_and_clears_resource_overrides(tmp_path):
    result, jobs = submit(tmp_path, {"SBATCH_ARRAY_INX": "0-99%99", "SBATCH_GRES": "gpu:8"})
    assert result.returncode == 0, result.stderr
    assert len(jobs) == 3
    for row, last in zip(jobs, (15, 5, 7), strict=True):
        assert f"--array=0-{last}%4" in row["args"]
        assert "--wait" in row["args"] and "--gres=gpu:nvidia_a100_1g.10gb:1" in row["args"]
        assert row["overrides"] == []


def test_submitter_blocks_active_or_unknown_queue_states(tmp_path):
    result, jobs = submit(tmp_path, {"FAKE_QUEUE": "15|RUNNING"})
    assert result.returncode != 0 and jobs == []
    result, jobs = submit(tmp_path, {"FAKE_QUEUE": "15|UNKNOWN"})
    assert result.returncode != 0 and jobs == []


def test_submitter_rejects_large_concurrency_and_stops_on_failure(tmp_path):
    result, jobs = submit(tmp_path, {"FOCUSED_CONCURRENCY": "5"})
    assert result.returncode != 0 and jobs == []
    result, jobs = submit(tmp_path, {"FAKE_FAIL": "1"})
    assert result.returncode != 0 and len(jobs) == 1


def submit(tmp_path, overrides):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(root / "scripts/submit_focused_sweep.sh", tmp_path / "scripts")
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    for name, body in {
        "flock": "exit 0",
        "squeue": 'printf "%s\\n" "${FAKE_QUEUE:-1|COMPLETED}"',
        "sbatch": """exec python3 - "$@" <<'PY'
import json,os,sys
with open(os.environ['FAKE_LOG'],'a') as f:
    overrides=[k for k in os.environ if k.startswith('SBATCH_')]
    f.write(json.dumps(dict(args=sys.argv[1:],overrides=overrides))+'\\n')
sys.exit(int(os.environ.get('FAKE_FAIL','0')))
PY""",
    }.items():
        path = binary / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    log = tmp_path / "jobs.jsonl"
    log.unlink(missing_ok=True)
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(("FOCUSED_", "SBATCH_", "FAKE_"))
    }
    environment.update(PATH=str(binary) + os.pathsep + os.environ["PATH"], FAKE_LOG=str(log))
    environment.update(overrides)
    result = subprocess.run(
        ["bash", str(tmp_path / "scripts/submit_focused_sweep.sh")],
        text=True,
        capture_output=True,
        env=environment,
    )
    return result, [
        json.loads(line) for line in log.read_text().splitlines()
    ] if log.exists() else []
