#!/bin/bash
# One semantic configuration per independent Slurm array job.
# The configuration itself contains three seeded 64/16/20 repeats.
# Submit with: SEMANTIC_EPOCHS=50 bash scripts/submit_semantic_sweep.sh
#SBATCH --job-name=semantic
# DGX A100 10 GB MIG slices are exposed through the prod10 Slurm partition.
#SBATCH --partition=prod10
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --output=logs/semantic-%A_%a.out
#SBATCH --error=logs/semantic-%A_%a.err
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail

experience_id=${SLURM_ARRAY_TASK_ID:?Expected a semantic experiment array task}
epochs=${SEMANTIC_EPOCHS:-50}
output=${SEMANTIC_OUTPUT:-outputs/semantic-cluster-${epochs}}
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
uv run --no-sync python scripts/semantic_cluster_sweep.py \
    --experience-id "$experience_id" \
    --epochs "$epochs" \
    --device cuda \
    --output "$output"
