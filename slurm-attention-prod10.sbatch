#!/bin/bash
# Two independent jobs: fixed QCSE and learned classical input encoding.
# Submit from the checkout after uv sync --locked and mkdir -p logs.
#SBATCH --job-name=attention
#SBATCH --partition=dgx-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --array=0-1%2
#SBATCH --output=logs/attention-%A_%a.out
#SBATCH --error=logs/attention-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository checkout}"
encodings=(qcse classical)
export ATTENTION_ENCODING="${encodings[${SLURM_ARRAY_TASK_ID:?Expected array task}]}"
export ATTENTION_OUTPUT="${ATTENTION_OUTPUT:-outputs/attention-cluster}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

nvidia-smi
uv run --no-sync python scripts/check_backend.py --device cuda
# Exercise this differentiable attention path on CUDA before the full-pool run.
uv run --no-sync python scripts/semantic_experiment.py \
    --model attention --encoding "$ATTENTION_ENCODING" --device cuda \
    --epochs 1 --samples-per-epoch 17 --batch-size 8 \
    --output "$ATTENTION_OUTPUT/preflight/$ATTENTION_ENCODING"
bash scripts/run_attention_comparison.sh \
    --datasets phrases cleaned --device cuda \
    --epochs "${ATTENTION_EPOCHS:-50}" --samples-per-epoch 5000 --evaluate-test
