"""Exercise workflow ordering, failure propagation and restart without Slurm or training."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def test_followup_driver_stops_on_failure_then_resumes_and_skips_completed(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    driver = scripts / "run_benchmark_followup.sh"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/run_benchmark_followup.sh", driver
    )
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    stub = tmp_path / "stub_python.py"
    stub.write_text(
        """
import json
import os
import sys
from pathlib import Path
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[1:]
with Path('calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args)+'\\n')
name = Path(args[0]).name
if name == 'prepare_benchmark_followup.py':
    stage, objective = args[1], args[args.index('--objective')+1]
    root = Path('outputs/fake')/f'{stage}-{objective}'
    root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps({'tasks':[{},{}]}))
    (root/'job-timings').mkdir()
    (root/'job-timings/job.json').write_text(json.dumps({'started_at':'2026-10-16T00:00:00Z'}))
    print(f'Prepared 2 CV tasks at {root}')
elif name == 'submit_robust_benchmark.py':
    assert '--wait' in args and args[args.index('--concurrency')+1] == '4'
    assert '--sbatch-option=--time=24:00:00' in args
    root = Path(args[1]); stage = args[args.index('--stage')+1]
    if stage == 'fit' and os.environ.get('FAIL_FIT'):
        sys.exit(1)
    indexes = range(2) if stage == 'fit' else range(1)
    for i in indexes:
        out = root/('refit' if stage == 'refit' else 'cv')/f'task-{i:04d}'
        out.mkdir(parents=True, exist_ok=True)
        (out/'result.json').write_text('{}')
    if stage == 'pilot':
        (root/'compute.json').write_text(json.dumps({
            'reference_device':'cuda',
            'scenarios':[{'longest_cv_fit_reference_hours':1.0}]
        }))
elif name == 'robust_benchmark.py':
    root = Path(args[2])
    if args[1] == 'select':
        assert all((root/'cv'/f'task-{i:04d}'/'result.json').exists() for i in range(2))
        (root/'selection.json').write_text(json.dumps({'refits':[{}]}))
    if args[1] == 'report':
        assert (root/'refit/task-0000/result.json').exists()
elif name != 'report_benchmark_timing.py':
    raise AssertionError(args)
"""
    )
    python.write_text(
        "#!/bin/bash\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(stub))
        + ' "$@"\n'
    )
    python.chmod(0o755)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in (("flock", "exit 0"), ("sacct", "echo 'JobID|ElapsedRaw'")):
        target = binaries / name
        target.write_text("#!/bin/bash\n" + body + "\n")
        target.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"])
    failed = subprocess.run(
        ["bash", str(driver)],
        cwd=tmp_path,
        env=dict(env, FAIL_FIT="1"),
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    calls_path = tmp_path / "calls.jsonl"

    def calls():
        return [json.loads(line) for line in calls_path.read_text().splitlines()]

    initial = calls()
    assert len([c for c in initial if c[0].endswith("submit_robust_benchmark.py")]) == 2
    assert not any(c[0].endswith("robust_benchmark.py") and c[1] == "select" for c in initial)
    subprocess.run(["bash", str(driver)], cwd=tmp_path, env=env, check=True, capture_output=True)
    all_calls = calls()
    submissions = [c for c in all_calls if c[0].endswith("submit_robust_benchmark.py")]
    # The failed fit is retried, not the already completed first pilot.
    assert sum(c[c.index("--stage") + 1] == "pilot" for c in submissions) == 4
    assert sum(c[c.index("--stage") + 1] == "fit" for c in submissions) == 5
    first_refit = next(
        i
        for i, c in enumerate(all_calls)
        if c[0].endswith("submit_robust_benchmark.py") and c[c.index("--stage") + 1] == "refit"
    )
    assert (
        sum(
            c[0].endswith("robust_benchmark.py") and c[1] == "select"
            for c in all_calls[:first_refit]
        )
        == 4
    )
    assert sum(c[c.index("--stage") + 1] == "refit" for c in submissions) == 2
    count = len(all_calls)
    subprocess.run(["bash", str(driver)], cwd=tmp_path, env=env, check=True, capture_output=True)
    assert not any(c[0].endswith("submit_robust_benchmark.py") for c in calls()[count:])
