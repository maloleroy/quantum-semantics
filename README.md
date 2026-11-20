# Quantum language experiments

A small CPU/CUDA simulator for Hamiltonians, QCSE-style circuits, classical
controls and encoder/decoder experiments. The goal is a useful quantum advantage,
not just a trainable quantum component.

For an accessible overview, read the [research summary](docs/research-summary.md),
then the [successful steps](docs/experiments/successful.md) and
[negative experiments](docs/experiments/unsuccessful.md).

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
docs/experiments/    quantified studies and experiment summary inventories
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

For Slurm submission on the DGX A100, see the [DGX guide](docs/dgx.md).

The default laptop dependencies are NumPy, CPU PyTorch and Matplotlib. The DGX
uses the optional CUDA build; see the [setup and measured speedups](docs/dgx.md).
`train`, `evaluate`, `predict` and `generate` accept `--device cpu|cuda`. The
default is CPU unless `QCSE_DEVICE` is set; `dgx.sbatch` sets it to CUDA and checks
GPU availability before execution. Experiments need no model weight download and
no Qiskit stack. Transformers is used only by the demo's pretrained reference
(dependency group `demo`; skip it with `--no-group demo`). This code is a simulator;
it is not a hardware backend. Add packages with `uv add PACKAGE` or
`uv add --dev PACKAGE`.

## Run an experiment

```bash
uv run qcse prepare --data data/wikitext-2-raw-v1 --tokenizer byte-bpe \
  --vocabulary 512 --window 128
```

Download the pinned WikiText snapshot using its [instructions](data/wikitext-2-raw-v1/README.md),
then use the printed dataset path below. Preparation preserves official article
partitions and fits the tokenizer and statistics on training articles only.
A limit of zero keeps all eligible targets. `--data` is required.

```bash
uv run qcse train --dataset output/dataset-.../dataset.npz \
  --models adaptive adaptive_mlp --rank 128 --recent-window 16 \
  --epochs 30 --batch-size 512 --loss nll --selection nll --seeds 11 22 33
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
Training defaults to a constant learning rate; `--lr-schedule cosine --min-lr-ratio .1`
opts into epoch-wise decay from `--lr` to one tenth of it. See the
[cosine comparison](docs/experiments/adaptive-cosine.md).
`--older-context-dropout .25 --recent-window 16` optionally hides the older branch
for 25% of training examples while preserving the latest 16 tokens. It defaults to
zero; validation, generation and stored datasets remain intact. See the
[context-masking protocol](docs/experiments/context-dropout.md).
`--factor-norm 1.4142135623730951` optionally normalizes each trainable complex
Hamiltonian factor column before training and after each Adam update. Positive
column scaling preserves its QR-based evolution; Adam moments remain unchanged.
This is a different optimization recipe, not an equivalent Adam trajectory.
It applies only to models with trainable spectral factors; the default is disabled.
Enabled fits record column-norm extrema/means at each validation checkpoint.

`--epoch-examples 524288` optionally samples that many rows without replacement
from the training pool each epoch, keeping a fixed update budget as the pool grows.
It defaults to all rows. Validation still runs once per epoch; see the
[coverage protocol](docs/experiments/depth-coverage.md).

`conditional` learns small token-angle tables controlling local gates between
joint Hamiltonian blocks, with direct word-register measurement. Use
`conditional_fixed` to freeze the angle table, `conditional --freeze-core` to
freeze the Hamiltonians, and `conditional_mlp` for the classical feature control.
See the [context-gate protocol and results](docs/experiments/conditional.md).
`product` instead prepares a learned local-rotation product state before the
Hamiltonians. `product --decoder rotated` learns a local token-qubit measurement
basis; see the [readout comparison](docs/experiments/product-readout.md).
`product_fixed` and `product_mlp` provide controller/classical
controls; see the [product-state comparison](docs/experiments/product.md).
`sequence` applies token-ordered local RY/RZ preparation, with `sequence_pooled`,
`sequence_fixed` and `sequence_mlp` controls. See the
[sequential preparation protocol](docs/experiments/sequence.md).
`stream` interleaves four ordered context blocks with joint quantum evolution;
`stream_pooled`, `stream --freeze-core` and `stream_mlp` provide controls.
See the [streaming protocol and results](docs/experiments/stream.md).
`adaptive` adds context-dependent Hamiltonian eigenvalues; `adaptive_mlp` matches
its input features. For `adaptive --freeze-core`, bases/base energies are fixed
while context-dependent shifts remain trainable. See the
[adaptive-spectrum protocol](docs/experiments/adaptive-spectrum.md).
`prior` starts from fixed training-bigram amplitudes, then learns contextual
Hamiltonian refinements before direct word measurement. `prior_unigram` changes
the fixed input distribution; `prior_mlp` supplies a matched classical correction.
`prior --freeze-core` fixes bases and base energies while still learning contextual
energy shifts. See the [count-initialized comparison](docs/experiments/prior-screen.md).

The local WikiText snapshot in `data/wikitext-2-raw-v1/` is ignored and reproducible
from its tracked manifest. Tests generate their own tiny corpora in temporary directories.
Select sources explicitly; do not concatenate related sentence variants as
independent evidence. Generated prepared datasets and weights belong in `output/`,
not `data/`.

For open-vocabulary prediction, `prepare --tokenizer byte-bpe` fits byte-pair
merges on WikiText training articles only and keeps case, numbers and punctuation.
Set `--vocabulary 512 --window 128` for the declared subword screen. See the [manual subword screen](docs/experiments/byte-bpe.md).
Use `uv run qcse generate --run RUN --fit FIT --text "The history of " --steps 64`
for repeated direct token draws. Subword and word scores use different units.

## Autocompletion demo

`src/demo.py` serves a local page (`src/demo.html`, no front-end dependencies).
Typing a prompt shows each model's 16-token continuation next to its name.
Hovering a name shows validation metrics, parameters, context, readout and
checkpoint. Hovering a continuation shows the five most likely next tokens and
the latency. The selector switches between the most likely continuation and
direct sampling; sampling uses seed 42, as `qcse generate` does. Each
continuation is computed by `generate`.

The retained comparison uses seed 22 of each model. Copy the five locked runs
from the DGX into the ignored `output/demo/` (about 670 MB):

```bash
R=quantum-semantics-scale-12-4-replication-20261118/output/scale-12-4-replication
C=quantum-semantics-coverage-seeds-20261110/output/coverage-seeds/results
rsync -a dgx:$R/qcse/s22/runs/run-20261118T125256Z-171c2b57/ output/demo/qcse-4096-s22/
rsync -a dgx:$R/adaptive/s22/runs/run-20261118T101237Z-db115cec/ output/demo/hamiltonian-4096-s22/
rsync -a dgx:$R/adaptive_mlp/s22/runs/run-20261118T101232Z-dded9ea9/ output/demo/mlp-4096-s22/
rsync -a dgx:$C/quantum_s22/run-20261110T120211Z-0fd9d9b4/ output/demo/hamiltonian-512-s22/
rsync -a dgx:$C/mlp_s22/run-20261110T133847Z-1ed4ea9f/ output/demo/mlp-512-s22/
uv run python -m src.demo "Raw QCSE=output/demo/qcse-4096-s22" \
  "16-qubit Hamiltonian=output/demo/hamiltonian-4096-s22" \
  "Matched MLP=output/demo/mlp-4096-s22" \
  "10-qubit Hamiltonian=output/demo/hamiltonian-512-s22" \
  "MLP control=output/demo/mlp-512-s22" \
  --reference "Pythia-31M=EleutherAI/pythia-31m@e556ace21b489575e94e9d50b6dad2fcc7419679"
```

Then open <http://127.0.0.1:8000/>. Each argument is `LABEL=RUN`, or `RUN` to use
the fit name. Each run directory must hold exactly one locked fit. `--steps`,
`--port`, `--threads` and `--device` are optional. `--reference LABEL=NAME@REVISION`
adds a pretrained Hugging Face causal language model; it can be repeated.

| Label | Run | Vocabulary | Parameters | Validation NLL |
|---|---|---:|---:|---:|
| Raw QCSE | `qcse`, 12 qubits, latest 16 tokens | 4,096 | 2,240 | 8.173 |
| 16-qubit Hamiltonian | `adaptive`, 12 + 4 qubits | 4,096 | 35,520,800 | 3.188 |
| Matched MLP | `adaptive_mlp` | 4,096 | 35,463,168 | 3.156 |
| 10-qubit Hamiltonian | `prior_content`, 9 + 1 qubits | 512 | 1,073,900 | 2.471 |
| MLP control | `adaptive_mlp` | 512 | 750,460 | 2.228 |
| Pythia-31M (reference) | GPT-NeoX, 6 layers, pretrained | 50,304 | 30,494,720 | not comparable |

Pythia-31M is shown for reference only and sits outside the comparison. It was
pretrained by EleutherAI on the Pile with its own tokenizer, without our
training split, update budget or validation selection. The Pile includes English
Wikipedia, so the model may have seen these articles. The revision is pinned
because the repository's weights were replaced on 27 February 2026. Its first
launch downloads the weights into the Hugging Face cache. Its tokenizer attaches
a space to the following word, so trailing spaces are removed before encoding
and from the start of its continuation.

The 512-token runs were trained with an earlier `models.py`, `training.py` and
`cli.py`, so `load_predictor` refuses them. Instead, the demo checks dataset
and weight hashes and requires the current code to reproduce the first eight
saved validation probabilities (relative tolerance 1e-6). All five runs pass
this check. The Hamiltonian bases depend only on weights. For `adaptive` fits,
the demo orthonormalizes them once at load time instead of at every token.
Recomputing the 65,536 × 128 QR took about 2.4 s per token on an Apple M2 CPU.
Continuations are unchanged.

Measured on that CPU with four threads, a 16-token continuation takes about
0.8 s for raw QCSE, 0.7–1.0 s for each Hamiltonian, 0.15 s for the matched MLP,
0.01 s for the MLP control and 0.1 s for Pythia-31M. The server handles one
request at a time. Each row updates when its model finishes, and the page sends
at most one pending request per model. These are seed-22 validation scores on
reused articles; the test partitions remain unscored. Continuations from the
compared models remain incoherent; see the
[research summary](docs/research-summary.md#what-the-model-actually-generates).
New runs record the hash of `src/demo.py`, like every other module in `src/`.

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
