# Running CUDA experiments on the DGX A100

Use SSH alias `dgx` (account `quantum-ml-2026`). The simulator supports CUDA for
all model families, training, evaluation, prediction and generation. Laptop
commands still default to CPU PyTorch. Intensive DGX work must use GPU; CPU
comparisons are limited to explicitly requested, bounded benchmarks.

## Install and submit

From the local checkout, copy code and the small source corpus to the separate
CUDA checkout used for verification:

```bash
rsync -aR --exclude=__pycache__ src tests data/phrases.csv README.md \
  pyproject.toml uv.lock .python-version dgx.sbatch dgx:quantum-semantics-cuda/
ssh dgx
cd ~/quantum-semantics-cuda
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --no-group cpu --extra cuda
mkdir -p output/slurm
sbatch --wait --job-name=qcse-tests dgx.sbatch
```

The `cpu` dependency group is enabled by default for the laptop. Disable it when
selecting the conflicting `cuda` extra, which installs locked PyTorch 2.14.0+cu130
on Linux. This follows [uv's optional accelerator configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
Install dependencies before submitting; never copy the laptop's `.venv`.
For strict Pyright on a DGX without system Node, bootstrap a stable version once:
`PYRIGHT_PYTHON_NODE_VERSION=22.16.0 uv run --no-sync pyright .`.
This avoids nodeenv selecting an unsupported alpha release.

`dgx.sbatch` requests `prod80`, one full A100-SXM4-80GB, four CPUs and 30 minutes.
It sets `QCSE_DEVICE=cuda`, requires CUDA tests, preserves Slurm's
`CUDA_VISIBLE_DEVICES`, and verifies a real GPU tensor allocation before running.
It exports `UV_NO_SYNC=1` so plain `uv run` uses the prepared CUDA environment
instead of reverting to the default CPU dependency group. Outside the batch
script, use `uv run --no-sync ... --device cuda` after CUDA setup. Missing CUDA
raises an error; there is no silent CPU fallback.

The full A100 is the verified performance configuration. The original 10 GB MIG
slice has fewer compute resources; the speedups below do not establish its speed.
Logs must exist before submission because Slurm opens them before running the script.

## Run an experiment

Copy prepared NPZ data explicitly, or prepare a corpus in an allocation:

```bash
sbatch --wait --job-name=qcse-prepare dgx.sbatch uv run qcse prepare \
  --data data/phrases.csv --output output/dgx-data --vocabulary 16 --window 4 \
  --train-limit 256 --validation-limit 64 --test-limit 64
```

For research fits, use the declared dataset, model configuration and budget.
For example, with a prepared 512-token, 128-context WikiText dataset:

```bash
sbatch --job-name=qcse-train --time=02:00:00 dgx.sbatch uv run qcse train \
  --dataset output/DATASET/dataset.npz --output output/dgx-runs \
  --models adaptive adaptive_mlp --rank 128 --recent-window 16 \
  --epochs 30 --seeds 11 --batch-size 1024 \
  --loss nll --selection nll --device cuda --threads 4
```

This is a command template, not a new research protocol: changing batch size
changes optimizer updates and must be declared before a scientific comparison.
Use adequately sized batches for GPU throughput. Tiny tests or single-token
inference can be slower on GPU because launch overhead dominates.

`--threads` controls CPU support threads, not GPU parallelism. Keep models and
forward/backward tensors on CUDA. Custom Python runners must explicitly pass
`device="cuda"` to `train`, `fit`, `build_model` or `load_predictor`; their API
default stays CPU. Training data are resident on the selected device; reduce
data/batch size if the dataset and model exceed VRAM. Legacy CPU-only launchers
must be updated before using them for intensive DGX work.

Training locks validation-selected weights without scoring test. After completion,
use the printed run path:

```bash
sbatch --wait --job-name=qcse-evaluate dgx.sbatch uv run qcse evaluate \
  --run output/dgx-runs/run-REPLACE --device cuda --threads 4
sbatch --wait --job-name=qcse-predict dgx.sbatch uv run qcse predict \
  --run output/dgx-runs/run-REPLACE --fit adaptive-s11 --text 'The history of' \
  --device cuda --threads 4
```

Declare selection before training and lock weights before held-out scoring.
`evaluate` refuses to overwrite test scores. Sampling success, argmax accuracy
and NLL remain distinct. Checkpoints are portable CPU/GPU NPZ inference weights;
optimizer resume is not implemented. Keep the matching source checkout for each
run: evaluation/prediction check source and weight hashes. Do not change code or
dependencies beneath a running job.

## Inspect and retrieve

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS
tail -f output/slurm/qcse-train-JOBID.out
cat output/slurm/qcse-train-JOBID.err
```

Require `COMPLETED` and exit `0:0`, then check fit/run metadata for `device: cuda`
and the accelerator name. Logs use job name and ID. From the laptop:

```bash
rsync -a dgx:quantum-semantics-cuda/output/dgx-runs/run-REPLACE/ output/dgx-result/
rsync -a dgx:quantum-semantics-cuda/output/slurm/ output/dgx-slurm/
```

For several commands in one allocation, use a `set -euo pipefail` shell script
under `output/` and submit `sbatch dgx.sbatch bash output/experiment.sh`.

## Verified execution and performance, 2–3 November 2026

Slurm job **10773**, `prod80`, full A100, 16 allocated CPU threads, Python 3.12.3,
PyTorch 2.14.0+cu130 / CUDA 13.0. Final job **10843** on 3 November passed
**109 tests**, Ruff, strict Pyright and the prepare/train/evaluate/predict/generate/plot
CLI workflow. All four CLI fits and test metadata record CUDA execution.
CPU/GPU probabilities and gradients agree across all 33 model kinds; locked
training, scoring, checkpoint portability and repeatable GPU generation pass.
The laptop separately passes 75 CPU tests and skips the 34 CUDA checks.

Three repeats per device used identical existing WikiText data (first 8,192 train and
first 1,024 validation rows), vocabulary 512, context 128, rank 128, two layers, seed 11,
two epochs, batches 1,024, Adam .001 and NLL selection. Both devices retain
float64/complex128. Timings synchronize CUDA and alternate device order. The
reported CPU baseline is the faster median of four and sixteen threads; GPU uses
four CPU support threads. All selected checkpoints agree within tolerance.

| Model | CPU fit median | CUDA fit median | Speedup | Update speedup |
|---|---:|---:|---:|---:|
| Adaptive quantum | 5.586 s | 1.348 s | **4.14×** | 5.09× |
| Matched adaptive MLP | 3.988 s | 1.210 s | **3.30×** | 2.88× |

Fit wall time includes model construction, transfers, optimization, validation,
selected-training scoring and artifact writing. It excludes interpreter startup,
input dataset loading/preparation, scheduler delay and CUDA context warmup.
Update timings separately use three warmup updates and three blocks of eight
synchronized updates. Profiling records 743/593 GPU kernels, including GPU matrix
products, for one quantum/MLP update; observed peak allocation stays below 326 MB.
The profiler's start/stop stderr messages are informational.

These are simulator throughput measurements, not quantum hardware speedup or new
predictive evidence. Benchmark test partitions were not scored. Compact evidence:
[fit timings](experiments/dgx-cuda-fits.csv), [update timings](experiments/dgx-cuda-steps.csv).
Raw logs, weights, arrays, hashes and generated benchmark tools are retained in
ignored `output/dgx-cuda/` and remote `~/quantum-semantics-cuda/output/`.
To repeat using the retained inputs/tools on the DGX:

```bash
sbatch --wait --cpus-per-task=16 dgx.sbatch uv run python -m output.cuda_tools.benchmark
```

The benchmark creates `output/cuda-benchmark`; move an existing benchmark directory
aside before repeating. This command needs the retained `output/cuda-input/full.npz`
and generated tool; neither is a tracked source dataset or packaged CLI command.
