#!/bin/bash
# 45 experiments in 9 independent jobs, five experiments each.
# Pass site overrides as sbatch options, e.g. --partition=prod20 --gres=gpu:nvidia_a100_1g.10gb:1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
submitted=$(sbatch --parsable "$@" --array="0-8%10" slurm-prod10.sbatch)
echo "Submitted array ${submitted%%;*} (9 jobs, 5 experiments each)"
