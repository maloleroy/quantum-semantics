#!/bin/bash
# One sequential workflow; only the existing bounded submitter allocates GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/robust-followup
exec 9> outputs/robust-followup/driver.lock
flock -n 9 || { echo 'A follow-up driver is already active.' >&2; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

complete() {
  .venv/bin/python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
root, stage = Path(sys.argv[1]), sys.argv[2]
plan = root / ('manifest.json' if stage == 'cv' else 'selection.json')
key = 'tasks' if stage == 'cv' else 'refits'
tasks = json.loads(plan.read_text())[key]
sys.exit(0 if all((root/stage/f'task-{i:04d}'/'result.json').is_file()
                  for i in range(len(tasks))) else 1)
PY
}

submit() {
  .venv/bin/python scripts/submit_robust_benchmark.py "$1" --stage "$2" \
    --concurrency 4 --wait --sbatch-option=--time=24:00:00
}

record_accounting() {
  local study="$1"
  local start
  start=$(.venv/bin/python - "$study" <<'PY'
import json
import sys
from pathlib import Path
print(min(json.loads(p.read_text())['started_at'][:10]
      for p in (Path(sys.argv[1])/'job-timings').glob('*.json')))
PY
)
  sacct --starttime "$start" --name qcse-bert --parsable2 \
    --format=JobID,State,ExitCode,ElapsedRaw,Start,End,AllocTRES > "$study/slurm-accounting.psv"
}

for stage in refine confirm; do
  for objective in causal cbow; do
    pointer="outputs/robust-followup/$stage-$objective-study.txt"
    if [[ ! -f "$pointer" ]]; then
      extra=(--objective "$objective")
      if [[ "$stage" == confirm ]]; then
        extra+=(--from-study "$(cat "outputs/robust-followup/refine-$objective-study.txt")")
      fi
      prepared=$(.venv/bin/python scripts/prepare_benchmark_followup.py "$stage" "${extra[@]}")
      study="${prepared##* CV tasks at }"
      test -f "$study/manifest.json"
      printf '%s\n' "$study" > "$pointer"
    fi
    study=$(cat "$pointer")
    .venv/bin/python scripts/robust_benchmark.py status "$study"
    if ! complete "$study" cv; then
      if [[ ! -f "$study/compute.json" || ! -f "$study/cv/task-0000/result.json" ]]; then
        submit "$study" pilot
      fi
      .venv/bin/python - "$study/compute.json" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1]))
if plan['reference_device'] != 'cuda' or any(
    row['longest_cv_fit_reference_hours'] > 22 for row in plan['scenarios']
):
    raise SystemExit('Review pilot: CUDA required and estimated fit must leave 2h within the 24h limit.')
PY
      submit "$study" fit
    fi
    .venv/bin/python scripts/robust_benchmark.py select "$study"
    record_accounting "$study"
    .venv/bin/python scripts/report_benchmark_timing.py "$study" \
      --concurrency 4 --slurm-accounting "$study/slurm-accounting.psv"
  done
done

# Both objectives' confirmation choices are frozen before any held-out scoring.
for objective in causal cbow; do
  study=$(cat "outputs/robust-followup/confirm-$objective-study.txt")
  if ! complete "$study" refit; then
    submit "$study" refit
  fi
  .venv/bin/python scripts/robust_benchmark.py report "$study" --bootstrap-draws 2000
  record_accounting "$study"
  .venv/bin/python scripts/report_benchmark_timing.py "$study" \
    --concurrency 4 --slurm-accounting "$study/slurm-accounting.psv"
done
echo 'Follow-up complete: 108 refinement CV fits, 150 confirmation CV fits, 30 refits.'
