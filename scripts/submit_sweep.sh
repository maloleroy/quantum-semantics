#!/bin/bash
# 75 experiments: ten initial jobs and five dependent jobs, five experiments each.
# Pass site overrides as sbatch options, e.g. --partition=prod20 --gres=gpu:nvidia_a100_1g.10gb:1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
previous=""
for wave in 0 1; do
    count=10
    if [[ "$wave" == 1 ]]; then count=5; fi
    if [[ -n "$previous" ]]; then
        submitted=$(sbatch --parsable --partition=prod10 --gres=gpu:nvidia_a100_1g.10gb:1 "$@" --array="0-$((count - 1))%10" --dependency="aftercorr:$previous" \
            slurm-prod10.sbatch "$wave")
    else
        submitted=$(sbatch --parsable --partition=prod10 --gres=gpu:nvidia_a100_1g.10gb:1 "$@" --array="0-$((count - 1))%10" slurm-prod10.sbatch "$wave")
    fi
    previous="${submitted%%;*}"
    echo "Wave $wave: array $previous ($count jobs, 5 experiments each)"
done
