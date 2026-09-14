#!/bin/bash
# Ten independent chains, each with three jobs of five sequential experiments.
# Pass site overrides as sbatch options, e.g. --partition=prod20 --gres=gpu:nvidia_a100_1g.10gb:1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
previous=""
for wave in 0 1 2; do
    if [[ -n "$previous" ]]; then
        submitted=$(sbatch --parsable "$@" --dependency="aftercorr:$previous" \
            slurm-prod10.sbatch "$wave")
    else
        submitted=$(sbatch --parsable "$@" slurm-prod10.sbatch "$wave")
    fi
    previous="${submitted%%;*}"
    echo "Wave $wave: array $previous (10 jobs, 5 experiments each)"
done
