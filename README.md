# Quantum semantics

QCSE circuits, learned quantum/classical context models and reproducible NLP
comparisons. Start with [backlog.md](backlog.md) for branch progress, findings
and remaining work. This README covers using the code.

The implementation is inspired by
[QCSE v2](https://arxiv.org/abs/2509.05729v2). It trains its own weights; the
authors' pretrained weights are not included. Paper-style bit metrics and
modern word-softmax benchmarks are separate experiments.

## Install and inspect

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. From the checkout:

```bash
uv sync --locked
uv run qcse --help
uv run qcse prepare
uv run python scripts/robust_benchmark.py --help
```

CPU is the default. `--device cuda` requires CUDA-enabled PyTorch and an actual
GPU allocation; `--device mps` selects an available Apple GPU. No IBM account,
sampling backend or external pretrained model is required.

Add runtime packages with `uv add PACKAGE`; development tools with
`uv add --dev PACKAGE`.

## Production QCSE

Small pipeline check with the complete vocabulary and a bounded example count:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run qcse train --epochs 2 --max-examples 32 --output outputs/smoke
```

The command prints a unique run directory. Use that path for inference/resume:

```bash
uv run qcse embed --model outputs/smoke/RUN/model.npz "the river moved slowly"
uv run qcse continue outputs/smoke/RUN/run.npz --epochs 10
uv run qcse train --help
uv run qcse continue --help
```

`RUN` means the printed directory name. New training creates unique folders;
continuation restores optimizer/RNG/split state. `--batch-size` changes Adam's
batch; `--simulation-batch-size` bounds simultaneous tensor contexts.
`--state-cache-mib` bounds retained encoded states. See [models](docs/models.md)
for encoding/readout and [optimizations](docs/optimizations.md) for memory limits.
The production CLI supports `--measurement-basis z|learned_y|learned_xyz`.
Learned readouts start at the exact Z-basis behavior and add one or two trainable
angles per qubit.

## Data

Default sources are `phrases.csv` and `cleaned_sentences.csv`; Tatoeba is
excluded. Named subsets retain vocabulary IDs from both complete active sources
before filtering/splitting. `--data` defines a custom corpus/vocabulary.

```bash
uv run qcse prepare --datasets phrases
uv run qcse prepare --cleaning strict --max-sentences 128
uv run qcse prepare --data my_sentences.csv
```

Default `dedupe` cleaning merges identical tokenized sentences across sources;
`basic` keeps duplicates and `strict` adds numeric/URL/length filters. Duplicate
sentences and their windows stay in the same split. Sentence caps and source
selection change the experiment; the robust studies keep the full eligible pool.
Corpus provenance, transductive vocabulary and target populations are detailed
in the [protocol](docs/protocol.md).

## Focused circuit studies

The [encoding study](experiments/encoding_study/REPORT.md) compares context
encodings and checks its batched simulator against Qiskit. The
[measurement-basis study](experiments/measurement_basis/REPORT.md) compares Z,
learned Y and two-angle local readouts for the causal objective. These studies
have their own runners and reports under `experiments/`.

## Other model families

| Task | Interface | Details |
| --- | --- | --- |
| Semantic decoder / quantum attention | `uv run python scripts/semantic_experiment.py --help` | [Models](docs/models.md); `--resume` uses a total epoch target. Chronological checkpoint and selected inference state are separate. |
| Robust QCSE/hybrid/MLP/BERT comparison | `uv run python scripts/robust_benchmark.py --help` | [Protocol](docs/protocol.md), [runs](docs/runs.md), [DGX](docs/dgx-a100.md). |
| Historical focused matrix | `uv run python scripts/focused_sweep.py --list` | [Archived protocol](docs/archive/FOCUSED_BENCHMARK.md); not the current 100×20k study. |
| New causal GPT / classical feature controls | `scripts/research_model_controls.py` | CPU-tested prototypes; runner integration is pending. |

Robust commands are `prepare`, `fit`, `status`, `select`, `refit` and `report`.
Preparation writes immutable manifests/arrays/source snapshots. Inspect an
existing study with `status STUDY`; use its declared protocol for fitting or
selection. Generic CLI defaults are historical, so use explicit settings for
a new fixed-budget experiment. Current next-stage design is in
[next experiments](docs/next-experiments.md); its launcher is not ready yet.

## DGX and results

Use the [DGX guide](docs/dgx-a100.md) for relative-path commands, existing-study
inspection, source guards, resume and export. `prod10` accepts `sbatch` only.
Supported current controllers allow at most four one-10GB MIG jobs. Frozen
studies use source 483bea6; do not change that checkout to read new notes.

Generated raw runs live in ignored `outputs/`; committed reports/CSV/figures are
under `results/`. [Runs](docs/runs.md) distinguishes complete, interrupted,
historical and diagnostic evidence. Conversation exports stay local and ignored.

## Development checks

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-sync pytest -q
uv run --no-sync ruff check src scripts tests
uv run --no-sync pyright
```

[Validation](docs/validation.md) records the latest scope and hardware gaps.
For research assumptions and primary sources, read [literature](docs/literature.md).
Older commands/decisions remain in the [archive](docs/archive/README.md).
