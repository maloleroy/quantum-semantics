# Reproducing this study

Run from the QCSE repository. Install the locked environment with `uv sync`.
Training is entirely local CPU simulation; no quantum/cloud credentials are used.
The supplied `data/tatoeba.csv` is versioned in Git. Its SHA-256, size summaries
and split parameters are recorded in `results/*_data/dataset.json`.
The original `phrases.csv` is already versioned.

Use a fresh output directory when changing any configuration. Completed jobs are
cached and configuration mismatches are rejected. An interrupted individual job
restarts; this experimental runner does not resume inside an epoch. Logs contain
one line per completed job, not a per-minibatch stream. `.pt` weights and `.npz`
arrays remain local and ignored; compact JSON records, reports and figures are
versioned. These checkpoints are study artifacts, not `QCSEModel.load` archives.

## Main corpus

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python -m experiments.hamiltonian_study.run prepare \
  --data data/tatoeba.csv --dataset outputs/hamiltonian-data \
  --vocabulary 512 --window 8 --train-limit 8192 --val-limit 4096 --test-limit 8192
uv run python -m experiments.hamiltonian_study.run baselines \
  --dataset outputs/hamiltonian-data
uv run python -m experiments.hamiltonian_study.run train \
  --dataset outputs/hamiltonian-data --output outputs/hamiltonian-main \
  --methods walk predictive_residual no_graph shuffled diffusion directed_diffusion \
    embedding qcse_bce qcse_nll \
  --layers 16 --seeds 11 22 33 --epochs 30 --batch-size 128 --workers 6 \
  > outputs/hamiltonian-main.log 2>&1
```

Use the already fixed comparison list in
`experiments/hamiltonian_study/results/main/selection_lock.json` to reproduce the
held-out comparison. For new research, write a new selection lock before looking
at test results. The file lists exact job keys and the selection rule.

```bash
uv run python -m experiments.hamiltonian_study.run evaluate \
  --dataset outputs/hamiltonian-data --output outputs/hamiltonian-main \
  --lock experiments/hamiltonian_study/results/main/selection_lock.json
```

## Depth, full-training classical control, and corpus replication

```bash
uv run python -m experiments.hamiltonian_study.run train \
  --dataset outputs/hamiltonian-data --output outputs/hamiltonian-depth \
  --methods walk --layers 1 4 8 32 --seeds 11 22 33 --epochs 30 --workers 3

uv run python -m experiments.hamiltonian_study.run prepare \
  --dataset outputs/hamiltonian-full-data --train-limit 0
uv run python -m experiments.hamiltonian_study.run baselines \
  --dataset outputs/hamiltonian-full-data
uv run python -m experiments.hamiltonian_study.run train \
  --dataset outputs/hamiltonian-full-data --output outputs/hamiltonian-full-classical \
  --methods embedding --layers 16 --seeds 11 22 33 --epochs 10 \
  --batch-size 1024 --eval-every 2 --workers 3
uv run python -m experiments.hamiltonian_study.run evaluate \
  --dataset outputs/hamiltonian-full-data --output outputs/hamiltonian-full-classical \
  --lock experiments/hamiltonian_study/results/full_classical/selection_lock.json

uv run python -m experiments.hamiltonian_study.run prepare \
  --data phrases.csv --dataset outputs/hamiltonian-phrases-data \
  --vocabulary 0 --train-limit 0 --val-limit 0 --test-limit 0
uv run python -m experiments.hamiltonian_study.run baselines \
  --dataset outputs/hamiltonian-phrases-data
uv run python -m experiments.hamiltonian_study.run train \
  --dataset outputs/hamiltonian-phrases-data --output outputs/hamiltonian-phrases \
  --methods walk predictive_residual embedding qcse_nll \
  --layers 16 --seeds 11 22 33 --epochs 30 --workers 3
uv run python -m experiments.hamiltonian_study.run evaluate \
  --dataset outputs/hamiltonian-phrases-data --output outputs/hamiltonian-phrases \
  --lock experiments/hamiltonian_study/results/phrases/selection_lock.json
```

The initial screen used 2,048 training and validation targets, 15 epochs, seed 11,
and 16 layers. `screen` contains the count-graph family and walk PPMI/nPMI controls;
`refinement_screen` adds ordered context phases, directed dilation, and the
near-identity predictive residual. Each record stores the exact configuration
and source digest. The first screen predates the near-identity refinement; its
implementation is preserved in Git commit `d5b1bb3`.

## Checks and figures

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
uv run python -m experiments.hamiltonian_study.theory
uv run python -m experiments.hamiltonian_study.circuits
uv run python -m experiments.hamiltonian_study.analyze
```

`theory`, `circuits`, and `analyze` rebuild the included report artifacts at their
fixed `experiments/hamiltonian_study/results/` paths. For newly reproduced runs,
change `analyze.ROOT` or place the result folders under the same directory layout.
Numerical versions, CPU worker counts, commit IDs and full per-epoch validation
histories are saved with each stage. Exact sparse-exponential, Qiskit, permutation,
normalization and finite-difference tests are independent of the training outcomes.

## Additional controls and report generation

For the included dataset layout, the scalar calibration controls and their locked
test evaluation are:

```bash
uv run python -m experiments.hamiltonian_study.calibration
uv run python -m experiments.hamiltonian_study.calibration --evaluate
uv run python -m experiments.hamiltonian_study.static_walk
```

The joint Born control uses the main training command with `--methods qcse_born`,
16 layers, 30 epochs and seeds 11/22/33; its selection manifest is
`results/born_control/selection_lock.json`. Keep it in a separate output directory.

The semantic probe downloads the authors' `https://fh295.github.io/SimLex-999.zip`.
Extract `SimLex-999/SimLex-999.txt` into `results/external/SimLex-999.txt` under this
study. The expected SHA-256 and coverage are in `results/simlex_source.json`.
This external data is not committed. Run the probe only after fitting the models;
it does not tune models or refit vocabulary.

```bash
uv run python -m experiments.hamiltonian_study.semantic_probe
uv run python -m experiments.hamiltonian_study.analyze
uv run python -m experiments.hamiltonian_study.write_report
uv run python -m experiments.hamiltonian_study.build_pdf
uv run python -m experiments.hamiltonian_study.audit
```

The report writer is specific to the included study and contains its interpreted
findings. For different experimental outcomes, revise the narrative as well as
regenerating tables. `audit` checks metric arrays, selected checkpoints, data hashes,
split separation, finite weights, and writes the final artifact hash manifest.
ReportLab is a development dependency for PDF rendering; Poppler renders its pages
for visual quality checks.
