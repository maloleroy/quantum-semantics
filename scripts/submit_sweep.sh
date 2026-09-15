#!/bin/bash
# Submit the 45 QCSE experiments as nine independent array jobs, five experiments per job.
# The confirmed DGX A100 10 GB MIG defaults are in the batch file; pass site
# overrides as sbatch options when needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
concurrency=${DGX_CONCURRENCY:-10}
cpus=${DGX_CPUS_PER_TASK:-32}
epochs=${QCSE_EPOCHS:-50}
output=${QCSE_OUTPUT:-outputs/sweep}
submitted=$(sbatch --parsable \
    --array="0-8%${concurrency}" \
    --cpus-per-task="$cpus" \
    "$@" \
    --export="ALL,QCSE_EPOCHS=${epochs},QCSE_OUTPUT=${output}" \
    slurm-prod10.sbatch)
echo "Submitted independent QCSE array ${submitted%%;*} (9 jobs, 5 experiments/job, ${cpus} CPUs/job, concurrency ${concurrency})"
