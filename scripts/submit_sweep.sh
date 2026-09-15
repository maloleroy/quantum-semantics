#!/bin/bash
# Submit the 45 QCSE experiments as independent array tasks, one experiment per job.
# The confirmed DGX A100 10 GB MIG defaults are in the batch file; pass site
# overrides as sbatch options when needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
concurrency=${DGX_CONCURRENCY:-10}
submitted=$(sbatch --parsable \
    --array="0-44%${concurrency}" \
    "$@" \
    slurm-dgx-a100-10gb-qcse.sbatch)
echo "Submitted independent QCSE array ${submitted%%;*} (45 jobs, concurrency ${concurrency})"
