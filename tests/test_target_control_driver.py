"""Driver ordering, queue safety and restart checks without Slurm or training."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def test_profile_then_bounded_arrays_stop_on_failure_and_resume(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    driver = tools / "run_target_controls.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/run_target_controls.sh", driver)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    stub = tmp_path / "stub.py"
    stub.write_text("""
import json, os, sys
from pathlib import Path
args=sys.argv[1:]
with Path('calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
name=Path(args[0]).name
if name=='target_control_study.py':
    stage=args[1]
    if stage=='prepare':
        root=Path('outputs/cohort').resolve()
        for recipe in ('L4','L2pad','L2R2'): (root/recipe).mkdir(parents=True)
        Path('outputs/robust-target-controls-study.txt').write_text(str(root)+'\\n')
    elif stage=='complete': sys.exit(0 if (Path(args[2])/'done').exists() else 1)
    elif stage=='estimate': (Path(args[2])/'compute.json').write_text('{}')
    elif stage=='report': (Path(args[2])/'reported').touch()
    elif stage!='verify': raise AssertionError(args)
elif name=='profile_robust_benchmark.py':
    Path(args[args.index('--output')+1]).write_text('{}')
elif name=='check_benchmark_backend.py': print('{}')
elif name=='submit_robust_benchmark.py':
    assert args[args.index('--stage')+1]=='fit'
    assert args[args.index('--concurrency')+1]=='4' and '--wait' in args
    assert '--sbatch-option=--time=24:00:00' in args
    root=Path(args[1])
    if os.environ.get('FAIL_FIT')==root.name: sys.exit(1)
    (root/'done').touch()
else: raise AssertionError(args)
""")
    python.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(stub))
        + ' "$@"\n'
    )
    python.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    for name, body in {
        "flock": "exit 0",
        "squeue": 'printf "%s" "${QUEUE:-}"',
        "sbatch": """[[ "$*" == *"--gres=gpu:nvidia_a100_1g.10gb:1"* ]] || exit 2
[[ "$*" == *"--wait"* ]] || exit 2
[[ "$*" != *"--array"* && -z "${SBATCH_GRES:-}" ]] || exit 3
cat > profile.sbatch
export SLURM_SUBMIT_DIR="$PWD" SLURM_CPUS_PER_TASK=4
bash profile.sbatch""",
    }.items():
        file = bins / name
        file.write_text("#!/bin/bash\n" + body + "\n")
        file.chmod(0o755)
    env = dict(os.environ, PATH=str(bins) + os.pathsep + os.environ["PATH"], SBATCH_GRES="gpu:8")

    def run(**extra):
        return subprocess.run(
            ["bash", str(driver)],
            cwd=tmp_path,
            env=dict(env, **extra),
            capture_output=True,
            text=True,
        )

    blocked = run(QUEUE="1|RUNNING\n")
    assert blocked.returncode != 0 and not (tmp_path / "profile.sbatch").exists()
    failed = run(QUEUE="1|COMPLETED\n", FAIL_FIT="L2pad")
    assert failed.returncode != 0
    root = tmp_path / "outputs/cohort"
    assert (root / "L4/done").exists() and not (root / "reported").exists()

    def calls():
        return [json.loads(s) for s in (tmp_path / "calls.jsonl").read_text().splitlines()]

    before = len(calls())
    resumed = run()
    assert resumed.returncode == 0, resumed.stderr
    added = calls()[before:]
    assert not any(Path(c[0]).name == "profile_robust_benchmark.py" for c in added)
    submits = [c for c in added if Path(c[0]).name == "submit_robust_benchmark.py"]
    assert [Path(c[1]).name for c in submits] == ["L2pad", "L2R2"]
    assert (root / "reported").exists()
    before = len(calls())
    assert run().returncode == 0
    assert not any(Path(c[0]).name == "submit_robust_benchmark.py" for c in calls()[before:])
