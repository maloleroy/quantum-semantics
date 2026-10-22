#!/bin/bash
# Copied tools only; diagnostic pilot then eight missing fixed-budget L4 fits.
set -euo pipefail
if [[ $# -ne 0 ]]; then
    echo 'Usage: run_causal_controls.sh (rerun unchanged to resume)' >&2
    exit 1
fi
tools=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs logs
exec 8> outputs/causal-controls-driver.lock
flock -n 8 || { echo 'A causal-control driver is already running.' >&2; exit 1; }
pointer=outputs/robust-causal-controls-study.txt
if [[ ! -s "$pointer" ]]; then
    .venv/bin/python "$tools/target_control_study.py" prepare-causal
fi
cohort=$(cat "$pointer")
.venv/bin/python "$tools/target_control_study.py" verify "$cohort"
for name in "${!SBATCH_@}"; do unset "$name"; done
if [[ ! -s "$cohort/compute.json" ]]; then
    exec 9> logs/benchmark-submission.lock
    flock -n 9 || { echo 'Another benchmark submission is in progress.' >&2; exit 1; }
    queue=$(squeue --me --partition=prod10 --states=all --noheader --format='%i|%T')
    while IFS='|' read -r job state; do
        [[ -z "$job" ]] && continue
        state=${state//[[:space:]]/}
        case "$state" in
            BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|TIMEOUT) ;;
            *) echo "Wait for prod10 job $job ($state) before starting." >&2; exit 1 ;;
        esac
    done <<< "$queue"
    export CONTROL_COHORT="$cohort" CONTROL_TOOLS="$tools"
    sbatch --wait --job-name=qcse-causal-profile --partition=prod10 \
        --gres=gpu:nvidia_a100_1g.10gb:1 --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --time=01:00:00 --export=ALL \
        --output=logs/qcse-causal-profile-%j.out \
        --error=logs/qcse-causal-profile-%j.err <<'SBATCH'
#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the original repository checkout}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cp "$0" "$CONTROL_COHORT/profile-job.sbatch"
.venv/bin/python "$CONTROL_TOOLS/target_control_study.py" verify "$CONTROL_COHORT"
.venv/bin/python scripts/check_benchmark_backend.py --device cuda --trainability \
    > "$CONTROL_COHORT/backend-preflight.json"
parent=$(.venv/bin/python -c 'import json,os; print(json.load(open(os.environ["CONTROL_COHORT"]+"/protocol.json"))["parent"])')
.venv/bin/python "$CONTROL_TOOLS/profile_benchmark_microbatches.py" "$parent" \
    --output "$CONTROL_COHORT/pilot" --device cuda
.venv/bin/python "$CONTROL_TOOLS/target_control_study.py" verify-pilot "$CONTROL_COHORT"
.venv/bin/python scripts/profile_robust_benchmark.py "$CONTROL_COHORT/L4" \
    --device cuda --objective causal --repeats 5 \
    --output "$CONTROL_COHORT/L4/profile-causal.json"
.venv/bin/python "$CONTROL_TOOLS/target_control_study.py" estimate "$CONTROL_COHORT"
SBATCH
    flock -u 9
    exec 9>&-
fi
.venv/bin/python "$tools/target_control_study.py" verify-pilot "$cohort"
.venv/bin/python "$tools/target_control_study.py" estimate "$cohort"
if ! .venv/bin/python "$tools/target_control_study.py" complete "$cohort/L4"; then
    .venv/bin/python scripts/submit_robust_benchmark.py "$cohort/L4" \
        --stage fit --concurrency 4 --wait --sbatch-option=--time=24:00:00
fi
.venv/bin/python "$tools/target_control_study.py" report "$cohort"
echo "Completed eight new L4 development fits; anchors preserved; no test/refits. Cohort: $cohort"
