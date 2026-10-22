#!/bin/bash
# Submit all project DGX A100 10 GB arrays independently, without dependencies.
# No array waits for another array; each task performs its own CUDA preflight.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

concurrency=${DGX_CONCURRENCY:-10}
qcse_threads=${QCSE_THREADS:-32}
epochs=${DGX_EPOCHS:-50}
qcse_output=${QCSE_OUTPUT:-outputs/sweep-dgx-a100-10gb}
semantic_output=${SEMANTIC_OUTPUT:-outputs/semantic-cluster-${epochs}}
attention_output=${ATTENTION_OUTPUT:-outputs/attention-dgx-a100-10gb}

qcse=$(sbatch --parsable \
    --array="0-8%${concurrency}" "$@" \
    --export="ALL,QCSE_EPOCHS=${epochs},QCSE_OUTPUT=${qcse_output},QCSE_THREADS=${qcse_threads}" \
    slurm-prod10.sbatch)
semantic=$(sbatch --parsable \
    --array="0-9%${concurrency}" "$@" \
    --export="ALL,SEMANTIC_EPOCHS=${epochs},SEMANTIC_OUTPUT=${semantic_output}" \
    slurm-semantic-prod10.sbatch)
attention=$(sbatch --parsable \
    --array="0-1%2" "$@" \
    --export="ALL,ATTENTION_EPOCHS=${epochs},ATTENTION_OUTPUT=${attention_output}" \
    slurm-attention-prod10.sbatch)

echo "Submitted independent DGX arrays:"
echo "  qcse      ${qcse%%;*} (9 jobs, 5 experiments/job, ${qcse_threads} internal threads, concurrency ${concurrency})"
echo "  semantic  ${semantic%%;*} (10 jobs, batch-file CPU allocation, concurrency ${concurrency})"
echo "  attention ${attention%%;*} (2 jobs, batch-file CPU allocation, concurrency 2)"
echo "No scheduler dependencies were submitted."
