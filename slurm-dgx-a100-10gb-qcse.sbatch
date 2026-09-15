#!/bin/bash
# One QCSE configuration per independent DGX A100 10 GB job.
# Submit with: bash scripts/submit_sweep.sh
# The array task is the experiment ID 0..44; there are no scheduler dependencies.
#SBATCH --job-name=qcse-one
#SBATCH --partition=dgx-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --array=0-44%10
#SBATCH --output=logs/qcse-%A_%a.out
#SBATCH --error=logs/qcse-%A_%a.err

set -euo pipefail
experiment_id=${SLURM_ARRAY_TASK_ID:?Expected a QCSE experiment array task}
epochs=${QCSE_EPOCHS:-50}
output=${QCSE_OUTPUT:-outputs/sweep-dgx-a100-10gb}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository checkout}"

if [[ ! -f pyproject.toml && -f quantum-semantics/pyproject.toml ]]; then
    cd quantum-semantics
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run uv sync --locked in the checkout before submitting.' >&2
    exit 1
fi

mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

nvidia-smi
uv run --no-sync python scripts/check_backend.py --device cuda
uv run --no-sync pytest tests/test_simulation.py tests/test_cross_validation.py tests/test_sampling.py -q -k cuda
uv run --no-sync python scripts/training_sweep.py \
    --experiment-id "$experiment_id" \
    --device cuda \
    --epochs "$epochs" \
    --output "$output"
