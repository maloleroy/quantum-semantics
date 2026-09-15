#!/bin/bash
# Submit the 45 QCSE experiments as independent array tasks, one experiment per job.
# Defaults target the DGX A100 10 GB MIG partition; pass site overrides as sbatch options.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
partition=${DGX_PARTITION:-dgx-a100}
gres=${DGX_GRES:-gpu:nvidia_a100_1g.10gb:1}
concurrency=${DGX_CONCURRENCY:-10}
submitted=$(sbatch --parsable \
    --partition="$partition" \
    --gres="$gres" \
    --array="0-44%${concurrency}" \
    "$@" \
    slurm-dgx-a100-10gb-qcse.sbatch)
echo "Submitted independent QCSE array ${submitted%%;*} (45 jobs, concurrency ${concurrency})"
