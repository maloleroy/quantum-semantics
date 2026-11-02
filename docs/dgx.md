# Running on the DGX A100

Use the configured SSH alias `dgx` (account `quantum-ml-2026`). The current code
uses **CPU PyTorch**, including on the DGX: there is no CUDA device option.
Requesting an A100 slice does not accelerate this simulator. `dgx.sbatch` retains
the supplied allocation: `prod10`, four CPU cores and one 10 GB MIG slice, with
a 30-minute default limit. It preserves Slurm's `CUDA_VISIBLE_DEVICES`.

## Setup

From the local checkout, copy the code, locked dependencies and small corpus:

```bash
rsync -aR --exclude=__pycache__ src tests data/phrases.csv README.md \
  pyproject.toml uv.lock .python-version dgx.sbatch dgx:quantum-semantics/
ssh dgx
cd ~/quantum-semantics
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
mkdir -p output/slurm
```

`uv` is installed at `~/.local/bin/uv`; noninteractive SSH does not initially
include it in PATH. The batch file adds it. Install dependencies before submitting.
Use a separate checkout for concurrent experiments on different source revisions;
do not update source or dependencies beneath a running job. Copy larger corpora or
prepared NPZ datasets explicitly, preserving their paths. Never copy a local `.venv`.

## Submit tests and experiments

Run these commands **on the DGX, from the checkout root**. Create `output/slurm`
before submission because Slurm opens logs before the script runs.

```bash
# With no command, the batch file runs uv run --locked pytest -q.
sbatch --wait --job-name=qcse-tests dgx.sbatch

# Small end-to-end example; --wait returns after preparation finishes.
sbatch --wait --job-name=qcse-prepare dgx.sbatch uv run --locked qcse prepare \
  --data data/phrases.csv --output output/dgx-data \
  --vocabulary 16 --window 4 --train-limit 256 --validation-limit 64 --test-limit 64
```

Use the exact dataset path printed in `output/slurm/qcse-prepare-JOBID.out`:

```bash
sbatch --job-name=qcse-train --time=02:00:00 dgx.sbatch uv run --locked qcse train \
  --dataset output/dgx-data/dataset-REPLACE/dataset.npz \
  --output output/dgx-runs --models hamiltonian adaptive mlp bigram \
  --epochs 2 --seeds 11 --rank 4 --batch-size 64 \
  --loss nll --selection success --threads 4
```

For research fits, substitute the declared dataset, model settings, seeds and
budget. Match `--threads` to `--cpus-per-task`; batch thread environment variables
alone do not override the CLI's default of one Torch thread. For several commands
in one allocation, put them in a shell script under `output/` with
`set -euo pipefail`, then submit `sbatch dgx.sbatch bash output/experiment.sh`.
No array launcher is required.

Training writes a unique run directory and locks validation-selected weights
without scoring test. Once training has completed, use the printed run path:

```bash
sbatch --wait --job-name=qcse-evaluate dgx.sbatch uv run --locked qcse evaluate \
  --run output/dgx-runs/run-REPLACE --threads 4
```

Declare the validation selection criterion before training; score held-out test
only after locking weights. `evaluate` refuses to overwrite existing test scores.
Keep the matching source checkout for evaluation/prediction: source and weight
hashes are checked. Checkpoints contain inference weights, not optimizer resume.
Sampling success, argmax accuracy and NLL remain separate metrics.

## Logs and results

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS
tail -f output/slurm/qcse-train-JOBID.out
cat output/slurm/qcse-train-JOBID.err
```

Require `COMPLETED` and exit `0:0`, then inspect the artifacts. Logs use `%x-%j`
(job name and ID). Runs contain CSV histories/scores, NPZ weights/arrays and small
JSON metadata. From the local machine, copy back a specific run and its logs:

```bash
rsync -a dgx:quantum-semantics/output/dgx-runs/run-REPLACE/ output/dgx-result/
rsync -a dgx:quantum-semantics/output/slurm/ output/dgx-slurm/
```

## Verified 2 November 2026

On `dgx-a100`, Python 3.12.3 / PyTorch 2.14.0+cpu:

| Slurm job | Check | Result | Allocation elapsed |
|---|---|---|---|
| 10771 | Full pytest suite | 73 passed | 9 s |
| 10772 | Prepare/train/evaluate/predict/plot | Passed | 14 s |

Both used `prod10`, four CPUs and one MIG slice, completed with exit `0:0`, and
produced empty stderr logs. The smoke experiment used the small recipe above,
two epochs and cosine decay for Hamiltonian, adaptive, MLP and bigram models.
Four locked fits, source/dataset/weight hashes, finite NPZ arrays, CSV histories,
test scores and a PNG plot were verified after copying back. This is a workflow
check, not a model comparison or a full research run. Raw evidence and generated
verification scripts are retained locally under ignored `output/dgx-verification/`;
remote smoke artifacts are under `output/dgx-smoke-10772/`.
