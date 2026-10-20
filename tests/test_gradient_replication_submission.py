"""Exercise Slurm guards and sequential diagnostic jobs without allocating GPUs."""

import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    stub = tmp_path / "python_stub.py"
    stub.write_text("""
import json, os, sys
from pathlib import Path
if sys.argv[1] == '-':
    sys.stdin.read()
    sys.exit(int(os.environ.get('PREFLIGHT_ERROR', '0')))
with Path('calls.jsonl').open('a') as stream:
    stream.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1].endswith('check_benchmark_backend.py'):
    print('{}')
else:
    objective = Path(sys.argv[2]).name
    if os.environ.get('FAIL_PROBE') == objective:
        sys.exit(1)
    output = Path(sys.argv[sys.argv.index('--output')+1])
    output.mkdir(parents=True)
    (output/'summary.json').write_text('{}')
""")
    python.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(stub))
        + ' "$@"\n'
    )
    python.chmod(0o755)
    for name, body in {
        "flock": 'exit "${LOCK_ERROR:-0}"',
        "squeue": 'printf "%s" "${QUEUE:-}"; exit "${QUEUE_ERROR:-0}"',
    }.items():
        path = bin_path / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    (tmp_path / "sbatch.py").write_text("""
import json, os, subprocess, sys
from pathlib import Path
Path('submission.json').write_text(json.dumps(dict(
    args=sys.argv[1:], inherited=[k for k in os.environ if k.startswith('SBATCH_')])))
job=Path('submitted.sbatch')
job.write_text(sys.stdin.read())
env=dict(os.environ, SLURM_SUBMIT_DIR=str(Path.cwd()), SLURM_JOB_ID='123', SLURM_CPUS_PER_TASK='4')
sys.exit(subprocess.run(['bash', str(job.resolve())], env=env).returncode)
""")
    (bin_path / "sbatch").write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(tmp_path / "sbatch.py"))
        + ' "$@"\n'
    )
    (bin_path / "sbatch").chmod(0o755)
    outputs = tmp_path / "outputs"
    followup = outputs / "robust-followup"
    followup.mkdir(parents=True)
    (outputs / "probe-nlp-trainability-d43cc96.py").write_text("copied diagnostic")
    for objective in ("causal", "cbow"):
        (followup / f"confirm-{objective}-study.txt").write_text(f"outputs/{objective}\n")
    script = Path(__file__).resolve().parents[1] / "scripts/submit_gradient_replications.sh"

    def run(**environment):
        return subprocess.run(
            ["bash", str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env=dict(
                os.environ,
                PATH=str(bin_path) + os.pathsep + os.environ["PATH"],
                SBATCH_ARRAY_INX="0-99",
                SBATCH_GRES="gpu:8",
                **environment,
            ),
        )

    return tmp_path, run


def test_terminal_jobs_allow_one_mig_and_both_objectives_run_sequentially(launcher):
    root, run = launcher
    result = run(QUEUE="9115_0|COMPLETED\n9116|CANCELLED\n")
    assert result.returncode == 0, result.stderr
    submission = json.loads((root / "submission.json").read_text())
    assert "--gres=gpu:nvidia_a100_1g.10gb:1" in submission["args"]
    assert "--ntasks=1" in submission["args"] and "--nodes=1" in submission["args"]
    assert not submission["inherited"]
    assert not any(a.startswith("--array") for a in submission["args"])
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert calls[0][0] == "scripts/check_benchmark_backend.py"
    assert [Path(c[1]).name for c in calls[1:]] == ["causal", "cbow"]
    for call in calls[1:]:
        assert call[call.index("--seeds") + 1] == "32"
        assert call[call.index("--batch-seeds") + 1 : call.index("--head-seeds")] == [
            "619",
            "620",
            "621",
        ]
        assert call[call.index("--head-seeds") + 1 :] == ["619", "719", "819"]
    with zipfile.ZipFile(root / "outputs/robust-gradient-replicated-123.zip") as archive:
        assert archive.testzip() is None
        assert any(n.endswith("causal/summary.json") for n in archive.namelist())
        assert any(n.endswith("cbow/summary.json") for n in archive.namelist())
    timing = root / "outputs/robust-gradient-replicated/job-123/timing.tsv"
    assert [line.split()[0] for line in timing.read_text().splitlines()] == [
        "stage",
        "backend-preflight",
        "causal",
        "cbow",
    ]


@pytest.mark.parametrize(
    "environment",
    [
        {"QUEUE": "1|RUNNING\n"},
        {"QUEUE": "1|PENDING\n"},
        {"QUEUE": "1|UNKNOWN\n"},
        {"QUEUE_ERROR": "1"},
        {"PREFLIGHT_ERROR": "1"},
        {"LOCK_ERROR": "1"},
    ],
)
def test_unsafe_or_unverified_submission_is_blocked(launcher, environment):
    root, run = launcher
    assert run(**environment).returncode != 0
    assert not (root / "submission.json").exists()


def test_first_probe_failure_stops_second_objective_and_packaging(launcher):
    root, run = launcher
    assert run(FAIL_PROBE="causal").returncode != 0
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert not (root / "outputs/robust-gradient-replicated-123.zip").exists()
