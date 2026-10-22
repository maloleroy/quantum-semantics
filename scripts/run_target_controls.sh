#!/bin/bash
# Copied tooling; preserve the original trainer and resume the saved cohort.
set -euo pipefail
if [[ $# -gt 1 || ( -n "${1:-}" && "${1:-}" != --new ) ]]; then
    echo 'Usage: run_target_controls.sh [--new]' >&2
    exit 1
fi
tools=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs logs
exec 8> outputs/target-controls-driver.lock
flock -n 8 || { echo 'A target-control driver is already running.' >&2; exit 1; }
pointer=outputs/robust-target-controls-study.txt
if [[ "${1:-}" == --new || ! -s "$pointer" ]]; then
    .venv/bin/python "$tools/target_control_study.py" prepare
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
    sbatch --wait --job-name=qcse-control-profile --partition=prod10 \
        --gres=gpu:nvidia_a100_1g.10gb:1 --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --time=01:00:00 --export=ALL \
        --output=logs/qcse-control-profile-%j.out \
        --error=logs/qcse-control-profile-%j.err <<'SBATCH'
#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the original repository checkout}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
cp "$0" "$CONTROL_COHORT/profile-job.sbatch"
.venv/bin/python "$CONTROL_TOOLS/target_control_study.py" verify "$CONTROL_COHORT"
.venv/bin/python scripts/check_benchmark_backend.py --device cuda --trainability \
    > "$CONTROL_COHORT/backend-preflight.json"
for recipe in L4 L2pad L2R2; do
    objective=cbow
    [[ "$recipe" == L4 ]] && objective=causal
    .venv/bin/python scripts/profile_robust_benchmark.py "$CONTROL_COHORT/$recipe" \
        --device cuda --objective "$objective" --repeats 5 \
        --output "$CONTROL_COHORT/$recipe/profile-$objective.json"
done
.venv/bin/python "$CONTROL_TOOLS/target_control_study.py" estimate "$CONTROL_COHORT"
SBATCH
    flock -u 9
    exec 9>&-
fi
.venv/bin/python "$tools/target_control_study.py" estimate "$cohort"
for recipe in L4 L2pad L2R2; do
    .venv/bin/python "$tools/target_control_study.py" verify "$cohort"
    if ! .venv/bin/python "$tools/target_control_study.py" complete "$cohort/$recipe"; then
        .venv/bin/python scripts/submit_robust_benchmark.py "$cohort/$recipe" \
            --stage fit --concurrency 4 --wait --sbatch-option=--time=24:00:00
    fi
done
.venv/bin/python "$tools/target_control_study.py" report "$cohort"
echo "Completed 32 development fits; no test selection or refits. Cohort: $cohort"
