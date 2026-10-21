"""Single-MIG pilot, waited bounded array, failure propagation and restart."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def test_pilot_then_eight_fits_restart_without_resubmitting_pilot(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    driver = tools / "run_causal_controls.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/run_causal_controls.sh", driver)
    stub = tmp_path / "stub.py"
    stub.write_text("""
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path('calls.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='-c':
    exec(args[1]);sys.exit(0)
name=Path(args[0]).name
if name=='target_control_study.py':
    stage=args[1]
    if stage=='prepare-causal':
        root=Path('outputs/completion').resolve();(root/'L4').mkdir(parents=True)
        (root/'protocol.json').write_text(json.dumps(dict(parent='outputs/parent')))
        Path('outputs/robust-causal-controls-study.txt').write_text(str(root)+'\\n')
    elif stage=='complete':sys.exit(0 if (Path(args[2])/'done').exists() else 1)
    elif stage=='estimate':(Path(args[2])/'compute.json').write_text('{}')
    elif stage=='report':(Path(args[2])/'reported').touch()
    elif stage not in ('verify','verify-pilot'):raise AssertionError(args)
elif name in ('profile_robust_benchmark.py','profile_benchmark_microbatches.py'):
    target=Path(args[args.index('--output')+1])
    if name=='profile_benchmark_microbatches.py':target=target/'microbatch-report.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}')
elif name=='check_benchmark_backend.py':print('{}')
elif name=='submit_robust_benchmark.py':
    assert args[args.index('--stage')+1]=='fit'
    assert args[args.index('--concurrency')+1]=='4' and '--wait' in args
    assert '--sbatch-option=--time=24:00:00' in args
    if os.environ.get('FAIL_FIT'):sys.exit(1)
    (Path(args[1])/'done').touch()
else:raise AssertionError(args)
""")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
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
[[ "$*" == *"--wait"* && "$*" != *"--array"* ]] || exit 2
[[ -z "${SBATCH_GRES:-}" && -z "${SBATCH_ARRAY_INX:-}" ]] || exit 3
cat > profile.sbatch
export SLURM_SUBMIT_DIR="$PWD" SLURM_CPUS_PER_TASK=4
bash profile.sbatch""",
    }.items():
        file = bins / name
        file.write_text("#!/bin/bash\n" + body + "\n")
        file.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        SBATCH_GRES="gpu:8",
        SBATCH_ARRAY_INX="0-9%8",
    )

    def run(**extra):
        return subprocess.run(
            ["bash", str(driver)],
            cwd=tmp_path,
            env=dict(env, **extra),
            capture_output=True,
            text=True,
        )

    def calls():
        return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]

    assert run(QUEUE="1|RUNNING\n").returncode != 0
    assert not (tmp_path / "profile.sbatch").exists()
    failed = run(QUEUE="1|COMPLETED\n", FAIL_FIT="yes")
    assert failed.returncode != 0
    assert (tmp_path / "outputs/completion/compute.json").exists()
    assert not (tmp_path / "outputs/completion/reported").exists()
    before = len(calls())
    resumed = run()
    assert resumed.returncode == 0, resumed.stderr
    new = calls()[before:]
    assert not any(Path(c[0]).name.startswith("profile_") for c in new)
    assert sum(Path(c[0]).name == "submit_robust_benchmark.py" for c in new) == 1
    assert (tmp_path / "outputs/completion/reported").exists()
    before = len(calls())
    assert run().returncode == 0
    assert not any(Path(c[0]).name == "submit_robust_benchmark.py" for c in calls()[before:])
