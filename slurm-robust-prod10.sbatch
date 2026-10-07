#!/bin/bash
# One configuration/seed/fold (or selected refit) per independent GPU job.
#SBATCH --job-name=qcse-bert
#SBATCH --partition=prod10
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --output=logs/qcse-bert-%A_%a.out
#SBATCH --error=logs/qcse-bert-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the locked repository checkout}"
test -x .venv/bin/python || { echo 'Run uv sync --locked before submitting.' >&2; exit 1; }
stage=${BENCHMARK_STAGE:?Expected fit or refit}
study=${BENCHMARK_STUDY:?Expected prepared study directory}
task=$((${SLURM_ARRAY_TASK_ID:?Expected array task ID} + ${BENCHMARK_OFFSET:-0}))
[[ "$stage" == fit || "$stage" == refit ]] || exit 1

# Preserve Slurm's CUDA_VISIBLE_DEVICES, including MIG UUIDs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
nvidia-smi
.venv/bin/python scripts/check_backend.py --device cuda
.venv/bin/python scripts/check_benchmark_backend.py --device cuda
.venv/bin/python scripts/robust_benchmark.py "$stage" "$study" \
    --task-id "$task" --device cuda --resume
