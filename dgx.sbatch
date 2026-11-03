#!/bin/bash
# Submit from the checkout: mkdir -p output/slurm; sbatch dgx.sbatch [command ...]
#SBATCH --job-name=qcse
#SBATCH --partition=prod80
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=output/slurm/%x-%j.out
#SBATCH --error=output/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the checkout}"
export PATH="$HOME/.local/bin:$PATH"
[[ -f pyproject.toml ]] || { echo 'Submit from the checkout root.' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run uv sync --locked --no-group cpu --extra cuda before submitting.' >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export QCSE_DEVICE=cuda
export QCSE_REQUIRE_CUDA=1
# Keep the preinstalled CUDA environment; plain uv run otherwise selects CPU defaults.
export UV_NO_SYNC=1
# Preserve Slurm's CUDA_VISIBLE_DEVICES, including its MIG UUID.
nvidia-smi
# Fail before any workload if CUDA is missing; never silently fall back to CPU.
uv run python -c 'import torch; from src.device import execution_device; d = execution_device("cuda"); assert torch.ones(1, device=d).is_cuda; print("CUDA:", torch.cuda.get_device_name(d), torch.__version__)'
# Match qcse --threads to the allocation when running training/evaluation.
if [[ $# -eq 0 ]]; then
    set -- uv run --locked pytest -q
fi
printf 'Host: %s; job: %s; CPUs: %s\n' "$(hostname)" "$SLURM_JOB_ID" "$OMP_NUM_THREADS"
printf 'Command: '; printf '%q ' "$@"; printf '\n'
exec "$@"
