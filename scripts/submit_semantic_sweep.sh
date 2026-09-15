#!/bin/bash
# Ten independent jobs, one configuration per job. Each job runs three repeats.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
epochs=${SEMANTIC_EPOCHS:-50}
output=${SEMANTIC_OUTPUT:-outputs/semantic-cluster-${epochs}}
array=${SEMANTIC_ARRAY:-0-9%10}
cpus=${DGX_CPUS_PER_TASK:-32}
submitted=$(sbatch --parsable \
    --array="$array" \
    --cpus-per-task="$cpus" \
    --export="ALL,SEMANTIC_EPOCHS=${epochs},SEMANTIC_OUTPUT=${output}" \
    "$@" slurm-semantic-prod10.sbatch)
echo "Submitted independent semantic jobs ${submitted%%;*} (array ${array}, ${cpus} CPUs/job, epochs ${epochs}, output ${output})"
