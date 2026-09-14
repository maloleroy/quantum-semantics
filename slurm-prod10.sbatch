#!/bin/bash
# Submit with bash scripts/submit_sweep.sh.
#SBATCH --job-name=qml-sweep
#SBATCH --partition=prod10
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail
group_id=${SLURM_ARRAY_TASK_ID:?Expected an array job}
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the checkout or its parent}"
if [[ ! -f pyproject.toml && -f quantum-semantics/pyproject.toml ]]; then
    cd quantum-semantics
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run uv sync --locked in quantum-semantics before submitting the array.' >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES, including its MIG UUID.
nvidia-smi
uv run --no-sync python scripts/check_backend.py --device cuda
uv run --no-sync pytest tests/test_simulation.py tests/test_cross_validation.py tests/test_sampling.py -q -k cuda
uv run --no-sync python scripts/training_sweep.py \
    --group-id "$group_id" --device cuda
