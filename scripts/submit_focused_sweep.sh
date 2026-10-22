#!/bin/bash
# Historical focused benchmark: sequential arrays, at most four MIG instances.
# Each task performs one 64/16/20 fit and one validation-selected held-out test.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
if [[ $# -ne 0 ]]; then
    echo 'Usage: submit_focused_sweep.sh (configure via FOCUSED_* variables)' >&2
    exit 1
fi

concurrency=${FOCUSED_CONCURRENCY:-4}
case "$concurrency" in
    1|2|3|4) ;;
    *) echo 'FOCUSED_CONCURRENCY must be 1, 2, 3 or 4.' >&2; exit 1 ;;
esac
epochs=${FOCUSED_EPOCHS:-50}
patience=${FOCUSED_PATIENCE:-8}
min_epochs=${FOCUSED_MIN_EPOCHS:-10}
output=${FOCUSED_OUTPUT:-outputs/focused-sweep-50}
if [[ "$output" == *,* || "$output" == *$'\n'* ]]; then
    echo 'FOCUSED_OUTPUT cannot contain commas or newlines.' >&2
    exit 1
fi
exec 9> logs/benchmark-submission.lock
flock -n 9 || { echo 'Another benchmark submission is in progress.' >&2; exit 1; }
for name in "${!SBATCH_@}"; do unset "$name"; done

for family in causal encoding attention; do
    queue=$(squeue --me --partition=prod10 --states=all --noheader --format='%i|%T')
    while IFS='|' read -r job state; do
        [[ -z "$job" ]] && continue
        state=${state//[[:space:]]/}
        case "$state" in
            BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|TIMEOUT) ;;
            *) echo "Wait for prod10 job $job ($state) before starting." >&2; exit 1 ;;
        esac
    done <<< "$queue"
    case "$family" in
        causal) last=15 ;;
        encoding) last=5 ;;
        attention) last=7 ;;
    esac
    echo "Submitting $family array and waiting for completion."
    sbatch --parsable --wait --partition=prod10 --gres=gpu:nvidia_a100_1g.10gb:1 \
        --nodes=1 --ntasks=1 --array="0-${last}%${concurrency}" \
        --export="ALL,FOCUSED_FAMILY=${family},FOCUSED_EPOCHS=${epochs},FOCUSED_PATIENCE=${patience},FOCUSED_MIN_EPOCHS=${min_epochs},FOCUSED_OUTPUT=${output}" \
        slurm-focused-prod10.sbatch
done
echo 'Completed historical focused arrays; no scheduler dependencies were submitted.'
