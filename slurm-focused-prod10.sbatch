#!/bin/bash
# One focused configuration/repetition per independent Slurm array task.
# Submit with: bash scripts/submit_focused_sweep.sh
# The batch resource settings match the site's existing prod10 allocation.
#SBATCH --job-name=focused
#SBATCH --partition=prod10
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --output=logs/focused-%A_%a.out
#SBATCH --error=logs/focused-%A_%a.err

set -euo pipefail
task_id=${SLURM_ARRAY_TASK_ID:?Expected a focused benchmark array task}
family=${FOCUSED_FAMILY:?Expected FOCUSED_FAMILY=causal,encoding or attention}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository checkout}"
if [[ ! -f pyproject.toml && -f quantum-semantics/pyproject.toml ]]; then
    cd quantum-semantics
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run uv sync --locked in the checkout before submitting.' >&2
    exit 1
fi

mkdir -p logs
export FOCUSED_THREADS="${FOCUSED_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OMP_NUM_THREADS="$FOCUSED_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

nvidia-smi
uv run --no-sync python scripts/check_backend.py --device cuda
uv run --no-sync python scripts/focused_sweep.py \
    --family "$family" \
    --task-id "$task_id" \
    --device cuda \
    --epochs "${FOCUSED_EPOCHS:-50}" \
    --samples-per-epoch "${FOCUSED_SAMPLES_PER_EPOCH:-5000}" \
    --eval-examples "${FOCUSED_EVAL_EXAMPLES:-2048}" \
    --final-eval-examples "${FOCUSED_FINAL_EVAL_EXAMPLES:-10000}" \
    --early-stopping-patience "${FOCUSED_PATIENCE:-8}" \
    --min-epochs "${FOCUSED_MIN_EPOCHS:-10}" \
    --threads "$FOCUSED_THREADS" \
    --output "${FOCUSED_OUTPUT:-outputs/focused-sweep-50}"
