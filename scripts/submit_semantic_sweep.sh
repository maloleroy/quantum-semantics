#!/bin/bash
# Ten independent jobs, one configuration per job. Each job runs three repeats.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
epochs=${SEMANTIC_EPOCHS:-50}
output=${SEMANTIC_OUTPUT:-outputs/semantic-cluster-${epochs}}
array=${SEMANTIC_ARRAY:-0-9%10}
partition=${DGX_PARTITION:-dgx-a100}
gres=${DGX_GRES:-gpu:nvidia_a100_1g.10gb:1}
submitted=$(sbatch --parsable \
    --partition="$partition" \
    --gres="$gres" \
    --array="$array" \
    --export="ALL,SEMANTIC_EPOCHS=${epochs},SEMANTIC_OUTPUT=${output}" \
    "$@" slurm-semantic-prod10.sbatch)
echo "Submitted independent semantic jobs ${submitted%%;*} (array ${array}, epochs ${epochs}, output ${output})"
