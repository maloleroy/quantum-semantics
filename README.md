# Quantum language experiments

A small CPU reference implementation for Hamiltonians, QCSE-style circuits, classical
controls and encoder/decoder experiments. The goal is a useful quantum advantage,
not just a trainable quantum component.

Read [research direction](docs/research.md) and the
[quantified findings](docs/experiments/findings.md). The retained Tatoeba H model has
**6.76% ideal one-shot success**, versus **24.77% for classical MLP sampling**.
Its **24.21% argmax accuracy is a different inference protocol**. No practical
quantum advantage has been demonstrated.

For connected text, use the downloaded **WikiText-2 Raw** corpus. See the
[dataset recommendation, local timing estimates and preparation recipe](docs/datasets.md).
Passing `--data data/wikitext-2-raw-v1` preserves official splits and article context.

## Layout

```text
src/                 typed model, data, physics, scoring and training modules
data/                source corpora and download manifests
tests/              small scientific logic tests
docs/research.md     shared research contract and architecture notes
docs/experiments/    one compact quantified findings report
output/             generated datasets, CSV histories, NPZ weights, figures and local tools
```

All project Python code is directly under `src/`; import it as `src.*`. There is no
nested `qcse` package, separate scripts tree or experiment-specific runner tree.
The virtual environment and tool caches use their default locations.

## Setup

```bash
uv sync --locked
uv run qcse --help
```

The runtime dependencies are NumPy, CPU PyTorch and Matplotlib. There is no model
weight download or required Qiskit/Transformers stack. This code is a simulator;
it is not a hardware backend. Add packages with `uv add PACKAGE` or
`uv add --dev PACKAGE`.

## Run an experiment

```bash
uv run qcse prepare --data data/phrases.csv --vocabulary 64 --window 4
```

Use the printed dataset path below. Preparation splits normalized sentence groups
before fitting vocabulary or statistics. A limit of zero keeps all eligible targets.
If a corpus has too few words, the vocabulary shrinks to a supported power of two.

```bash
uv run qcse train --dataset output/dataset-.../dataset.npz \
  --models hamiltonian dephased mlp bigram --epochs 30 --seeds 11 22 33
```

Training creates a new run directory and locks selected weights **without scoring
its test partition**. The default is NLL training with validation selection by
expected sampling success. `--loss success` and `--selection nll` are explicit options.
Use a different output run for different model configurations/tuning recipes.

```bash
uv run qcse evaluate --run output/run-...
uv run qcse predict --run output/run-... --fit hamiltonian-s11 --text "the bird is"
uv run qcse plot --history output/run-.../hamiltonian-s11/history.csv
```

`evaluate` scores all locked fits once, then refuses to overwrite their test results.
`predict` draws one ideal simulator output; `--argmax` requests a separate diagnostic.
Checkpoints are NPZ inference weights; optimizer resume is not implemented.

Model choices include `qcse`, `hybrid`, `classical_hybrid`, `calibrated_bigram` and
`unigram`. `--encoder amplitude|qcse` chooses fixed loading; QCSE defaults to its
original encoding. `--context-method` retains exponential, diagonal, phase, hash
and angular context matrices. `--decoder rotated` learns local measurement-basis
rotations; `--decoder marginal` is a classical independent-bit reconstruction,
not the circuit's one-shot word distribution. Hybrid models have their own learned
preparation and conditional classical decoder. `--freeze-core` supports attribution
controls. Use `--threads` to control Torch's CPU thread count.

`conditional` learns small token-angle tables controlling local gates between
joint Hamiltonian blocks, with direct word-register measurement. Use
`conditional_fixed` to freeze the angle table, `conditional --freeze-core` to
freeze the Hamiltonians, and `conditional_mlp` for the classical feature control.
See the [context-gate protocol and results](docs/experiments/conditional.md).

The tracked source corpus is `data/phrases.csv`. The local WikiText snapshot in
`data/wikitext-2-raw-v1/` is ignored and reproducible from its tracked manifest.
Select sources explicitly; do not concatenate related sentence variants as
independent evidence. Generated prepared datasets and weights belong in `output/`,
not `data/`.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run pyright .
uv run pytest -q
```

Pyright is strict for all code and tests. Keep tests focused on data leakage,
physical/gradient identities, metric semantics and a short training/checkpoint path.
Arrays and weights use NPZ; histories/tables use CSV; JSON is reserved for metadata
under 64 KiB. `output/retained/` holds compact historical evidence, not active CLI
runs. Its bundle keys and provenance are documented in the findings report.
