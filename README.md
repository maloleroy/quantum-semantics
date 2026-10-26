# Quantum semantics

The current focus is **Hamiltonian encoders for causal word prediction**.
The [Hamiltonian report](experiments/hamiltonian_study/REPORT.md) contains the
model definitions and results; the [three-way causal comparison](experiments/hamiltonian_study/CAUSAL_COMPARISON.md)
compares QCSE encoding, fixed Hamiltonian encoding, and Hamiltonian encoding
followed by the QCSE ansatz.

Start with [retained findings](docs/findings.md) for the robust benchmark's key
results and the investigation of jumps/plateaus. It records what was measured,
what remains unresolved, and how to recover the retired reports from Git.
[backlog.md](backlog.md) lists the remaining work.

The [natural-language Born follow-up](experiments/hamiltonian_study/LANGUAGE_BORN.md)
tests Hamiltonian-only training on real Tatoeba next-word prediction: eight qubits,
256 vocabulary entries, fixed four-word context encoding and 65,536 H coefficients.
Across three fresh seeds, held-out NLL improves 6.5861→3.5974 and top-1
2.57%→24.21%. It beats the retrained dephased control (3.6683 / 19.51%), including
on known words. The matched-input classical MLP remains stronger (2.6205 / 38.29%).
The report retains split-safe comparisons, sentence-group intervals and the exact
24-fit CPU run, completed in 4m20s. No learned classical encoder/head is in the H model.

A new [fixed-encoding/Born-readout experiment](experiments/hamiltonian_study/BORN_EVIDENCE.md)
isolates training inside the Hamiltonian, with no learned classical encoder or decoder.
Across five fresh seeds it reaches 100% on unseen 16-symbol compositions and 91.96%
on held-out UCI digits, versus 11.67% / 5.84% at initialization. The digit likelihood
also beats a theoretical bound for every completely dephased model with the same
readout. A similarly sized classical MLP reaches 95.43%: this establishes Hamiltonian
trainability and a coherence contribution within the tested architecture. Its language
follow-up is linked above; superiority over strong classical models remains open.

A runnable [hybrid context-correction proof of concept](experiments/hamiltonian_study/CONTEXTUAL_POC.md)
learns a correction to the bigram prediction from earlier ordered words, using
a classical encoder/controller, latent Hamiltonian and classical word head.
On 4,096 existing Tatoeba validation targets, matched tuning gives mean NLL
2.7940 / 2.7816 and top-1 36.17% / 36.03% for Hamiltonian / classical context
models. Both improve on the initial bigram; classical has the lower loss.
The report includes the 30-fit search, controls and reproduction commands.
This is an exploratory hybrid result. It does not establish a useful
Hamiltonian contribution or validate the intended vocabulary-space model.

## Install

Python 3.11+ and uv are required. From the checkout:

```bash
uv sync --locked
uv run qcse --help
uv run python -m experiments.hamiltonian_study.run --help
```

The default sync installs CPU PyTorch and development tools, without NVIDIA/CUDA
packages. On macOS, the default PyTorch build also supports MPS.
On an NVIDIA GPU machine, replace the default `cpu` dependency group with `cuda`:

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync python scripts/check_backend.py --device cuda
```

The groups are mutually exclusive. Use `uv run --no-sync` after the CUDA sync
to keep that environment, or pass `--no-group cpu --group cuda` to `uv run` too.
Plain `uv sync --locked` switches back to CPU and removes CUDA packages.
`--device cuda` requires an actual GPU allocation; `--device mps` selects an
available Apple GPU. No IBM account, sampling backend or external pretrained
model is required.

Add runtime packages with `uv add PACKAGE`; development tools with
`uv add --dev PACKAGE`.

## Work on the Hamiltonian model

Read the [reproduction commands](experiments/hamiltonian_study/REPRODUCE.md),
[protocol](experiments/hamiltonian_study/PROTOCOL.md), and
[theory](experiments/hamiltonian_study/THEORY.md). The study uses `data/tatoeba.csv`,
a training-only vocabulary and graph, and sentence-grouped partitions. It has
its own runner and checkpoints. Its default experiments use local CPU simulation.

The alternating graph/context walk and the fixed graph encoder followed by
QCSE are separate models with different readouts; the reports distinguish them.
Keep their classical graph/bigram and identity controls when extending either.
Existing source, tests, selection records and results remain under
`experiments/hamiltonian_study/`.

## Supporting code

- `src/qcse/`: production QCSE simulator, data/training/resume pipeline and
  reusable quantum/classical comparison models. [Model map](docs/models.md).
- `experiments/encoding_study/` and `experiments/measurement_basis/`: isolated
  encoding/readout controls, including the simulator used by the Hamiltonian model.
- `scripts/`: backend checks, reusable runners/probes and inherited cluster
  support. Completed cohort launchers and report collectors have been retired.
- [Benchmark protocol](docs/protocol.md), [compute notes](docs/optimizations.md),
  and [DGX operations](docs/dgx-a100.md) document retained supporting workflows.

The production CLI uses `phrases.csv` and `cleaned_sentences.csv`, with the
complete shared vocabulary before subset selection. Its checkpoints and scores
are separate from the Hamiltonian study. A small production smoke run is:

```bash
uv run qcse train --epochs 2 --max-examples 32 --output outputs/smoke
```

New runs print a unique directory. `qcse continue RUN/run.npz --epochs 10`
restores the production optimizer, RNG and split. Generic robust comparison
commands remain available with `uv run python scripts/robust_benchmark.py --help`;
use an explicit protocol for new work and a historical checkout for frozen runs.
Raw outputs and conversation exports stay local and ignored. The retired
benchmark reports are summarized in [findings](docs/findings.md#evidence-and-recovery).

## Development

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run ruff format .
uv run ruff check .
uv run pyright .
uv run pytest -q
uv run python scripts/check_backend.py --device cpu
```

[Validation](docs/validation.md) records actual results and hardware limits.
