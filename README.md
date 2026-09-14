# QCSE with Qiskit and PyTorch

Implementation of **QCSE: A Pretrained Quantum Context-Sensitive Word Embedding
for Natural Language Processing**, [arXiv:2509.05729v2](https://arxiv.org/abs/2509.05729),
using the local [`../References/2509.05729v2.pdf`](../References/2509.05729v2.pdf).
This project trains its own weights on `phrases.csv` and
`cleaned_sentences.csv`; it does not include the
paper authors' pretrained weights or claim to reproduce their reported accuracy.

## Run

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
uv run qcse prepare
# Quick end-to-end training check, with the full vocabulary and a random subset:
uv run qcse train --epochs 2 --max-examples 64 --output outputs/smoke
uv run qcse embed --model outputs/smoke/<run>/model.npz "The river moved slowly under the bridge."
# Full corpus: potentially slow on a CPU; progress is printed after each epoch.
uv run qcse train --epochs 50 --layers 2
uv run python scripts/plot_training.py outputs/<run>/run.npz -o outputs/<run>/training_metrics.png
# Add epochs to an interrupted or completed run:
uv run qcse continue outputs/<run>/run.npz --epochs 25
uv run pytest
uv run ruff check .
uv run pyright
```

The default dataset is resolved relative to the source checkout, not the working
directory; no file selection or download is needed. From elsewhere, use
`uv run --project /absolute/path/to/QCSE qcse prepare`. `--data` and `--output`
accept other paths. `qcse prepare --help` / `qcse train --help` list all settings.
No IBM account, backend credentials, Aer, GPU or classical pretrained embedding
is needed. Qiskit builds the circuits and fixed context states; PyTorch computes
their statevector marginals in batches.

## Data selection and cleaning

The active sources are `phrases.csv` and `cleaned_sentences.csv`. `phrases` is
our original corpus (there is no separate `sentences.csv`). Both are used by
default. The old `tatoeba.csv` file remains in the checkout but is excluded from
training and vocabulary construction because `cleaned_sentences.csv` is its
curated version. Choose either source or both for custom training:

```bash
uv run qcse train --datasets phrases --epochs 10 --max-sentences 128
uv run qcse train --datasets phrases cleaned --sampling balanced --max-sentences 128
uv run qcse prepare --cleaning strict --max-sentences 128
uv run qcse train --data my_sentences.csv other_sentences.csv --epochs 10
```

For named datasets, the vocabulary always comes from **both complete active input
files**, before filtering, sentence sampling, example sampling, or train/test
splitting. Word IDs therefore remain identical across ablations. `--data` defines
a custom corpus instead, using the complete vocabulary of those explicit files.
Missing sources fail visibly; the vocabulary never silently shrinks.
New runs use 10,864 words (14 qubits); old checkpoints retain their saved vocabulary
and word IDs when resumed.

The loader recognizes first-row `Cleaned_Sentence`, `sentence`, `sentences`, and
`text` headers, handles quoted fields and unquoted sentence commas, normalizes
Unicode with NFKC, lowercases, and preserves internal apostrophes. Cleaning modes:

- `basic`: tokenization and removal of rows with fewer than two tokens; keep duplicates.
- `dedupe` (default): also merge identical tokenized sentences across sources.
- `strict`: dedupe, plus reject rows with digits, URLs, email markers, or over 40 tokens.

`--max-sentences N` samples before constructing windows. `--sampling uniform`
uses seeded random rows; `balanced` draws round-robin from shuffled source pools,
redistributing exhausted pools and selecting a shared deduplicated sentence only
once. These are overlapping sources, so source membership totals can exceed the
sample size. Identical sentence text always stays on the same side of the split,
including in `basic` mode. `--max-examples N` can further cap examples after splitting.

Each preparation/training directory includes the selected normalized `sentences.csv`,
`sentence_sources.json`, a complete `vocabulary.json`, and source hashes, cleaning,
and sampling information in `summary.json`. Original input files are preserved.
These are user-supplied sentence snapshots; the cleaned file overlaps Tatoeba and
should not be treated as an independent benchmark.

## Batched simulation

The default `--device cpu` uses float64 tensors. Use `--device cuda` for an NVIDIA
GPU with CUDA-enabled PyTorch, or `--device mps` for an Apple GPU. GPU simulation
uses float32 real/imaginary pairs. Device selection is explicit; requesting an
unavailable GPU reports an error.

```bash
uv run qcse train --device mps --batch-size 128 --simulation-batch-size 256
uv run qcse train --device cuda --batch-size 128 --simulation-batch-size 256
uv run qcse continue outputs/<run>/run.npz --epochs 10 --device mps
uv run qcse embed --device mps --model outputs/<run>/model.npz "the river moved"
```

`--batch-size` controls Adam's mini-batch and therefore the training trajectory.
`--simulation-batch-size` (default 256) limits the number of contexts simulated
at once **per weight vector**, including evaluation, without changing Adam's
batch. The two SPSA perturbations run together. `--state-cache-mib` (default 256)
bounds retained encoded states. Small corpora are packed onto the selected device;
larger corpora use a bounded host cache and transfer simulation-sized batches.
Evicted contexts are re-encoded with the same Qiskit circuit when needed. This
bounds statevector memory without dropping examples; the sentence/example arrays,
embeddings and checkpoint serialization still grow with corpus size on the host.

Diagonal RZ/CRZ gates are combined per layer, and the final diagonal gates are
skipped when computing marginals. Exported Qiskit circuits retain every gate.
One prediction per unique context supplies epoch metrics, checkpoint embeddings,
and final output. Small batches may run as fast or faster on CPU because GPU
dispatch has overhead.

Device and simulation batch size are runtime options, also accepted by
`QCSEModel(..., device="mps", simulation_batch_size=256)` and `QCSEModel.load(...)`.
Existing checkpoints remain readable and can move between devices; specify the
device again when resuming. Floating-point precision and device differences mean
cross-device runs need not be bit-for-bit identical. The optimizer and RNG state
remain resumable. Backend details: [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html)
and [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Cluster sweep: 45 causal configurations in nine jobs

From the cluster checkout on the new branch:

```bash
git switch corpus-training-sweep
uv sync --locked
mkdir -p outputs
uv run --no-sync python scripts/training_sweep.py --list > outputs/sweep-manifest.json
bash scripts/submit_sweep.sh
```

If updating a checkout from before the cross-validation history cleanup, fetch
and check out the rewritten remote tip instead of merging the old history:

```bash
git fetch origin
git switch --detach origin/corpus-training-sweep
uv sync --locked
bash scripts/submit_sweep.sh
```

Do this between sweeps: active jobs use the shared checkout and environment.

The submitter creates **nine independent jobs**, each running
**five configurations sequentially**: 9 × 5 = 45. Each configuration runs two
shuffle-split fits followed by one final refit, for **135 fits in total**. The array
uses task IDs 0–8 with a ten-job concurrency ceiling. Each job has
the supplied 12-hour limit, four CPUs, and one named MIG GPU in `prod10`.
A failed experiment stops its group. Already completed experiment folders remain intact.

All configurations use the **causal objective**, both active datasets, and dedupe
cleaning. The matrix has three focused groups:

| IDs | Runs | Variation | Fixed reference |
| --- | ---: | --- | --- |
| 0–26 | 27 | Alpha 0.1/1/3 × LR 0.0001/0.0003/0.001 × window 2/4/8 | 8 layers, batch 64, full corpus |
| 27–38 | 12 | (layers, batch): (2,16), (2,64), (8,16), (8,256), (64,64), (64,256), each at windows 2/8 | Alpha 1, LR 0.0003, full corpus |
| 39–44 | 6 | Sentence pools of 5,000/20,000/50,000 × uniform/balanced sampling | Alpha 1, LR 0.0003, window 4, 8 layers, batch 64 |

Simulation batch size follows Adam batch size. The main grid includes the shared
reference for every depth/batch comparison. Balanced sentence sampling redistributes
unused quota when the smaller `phrases` source is exhausted. CBOW remains available
for manual training but is excluded from this sweep.

Every fit defaults to **10 epochs** for pipeline validation and seed 42. Each epoch draws **5,000 token examples
with replacement** from its training split, independently of batch size. This is
an ongoing random stream: each epoch draws again from the entire eligible pool,
so there is no permanent 5,000-example training subset. An epoch is a fixed sample
budget rather than a full pass, and coverage of every example is not guaranteed.
The 39 full-data runs use all **202,172 deduplicated sentences**; only the six
sampling comparisons cap their sentence pools. All runs share the full
**10,864-word vocabulary (14 qubits)**. Outputs go to
`outputs/sweep/experiment-NNN/<unique-run>/`.

The 5,000-example budget gives 313 updates with batch size 16, 79 with batch size
64, and 20 with batch size 256. The final batch is smaller when necessary; every
fit defaults to 10 epochs (50,000 training draws). Override epochs with `--epochs N` and the draw budget with
`--samples-per-epoch N`; plain `qcse train` retains full passes when this option
is omitted. The sweep sets 5,000 explicitly.

Per-epoch train/validation curves use fixed, seeded samples of up to **2,048
examples per split**, controlled by `--eval-examples`. These monitoring samples
use a separate RNG and do not restrict the training pool. Histories record their
sizes, and logs/plots label sampled metrics. Sampled-epoch checkpoints omit
per-example embeddings to avoid a full-corpus prediction pass at every save;
model, optimizer, sampling RNG and split state remain resumable. The full
validation fold is evaluated once after its fit; final CV summaries use those
full scores. The final held-out test also remains complete.

Each configuration reserves **20% of sentence groups for test**, then performs
**two independent shuffle-splits within the remaining 80%**, holding out 20% of
development groups each time. Validation groups can overlap between repetitions;
this is not exhaustive two-fold CV. Duplicates and windows from one sentence stay
together. Each fit uses approximately 64% train / 16% validation / 20% held-out
test by sentence-group count (token-example proportions may differ). Both
fits start from the same seeded weights with fresh optimizer state. Their test
examples are excluded entirely. A fresh model then trains on all 80% of
development data for the same epoch target; the held-out test is scored once after this
refit. Compare configurations using validation metrics, not test scores.

The default state cache retains at most 256 MiB of host Qiskit states; one
two-lane, 256-context float32 simulation buffer at 14 qubits occupies 64 MiB,
with additional intermediates and library overhead. The small-corpus device
cache also stays within the configured budget. CUDA peak memory is unmeasured.
Corpus preparation, checkpoint serialization, and final full validation/test
scoring still scale with data size. Evicted contexts require Qiskit encoding;
135 fits are not guaranteed to finish within the 12-hour job limits. Checkpoints
save every complete epoch; resume an interrupted configuration with `resume-cv`
below. Increasing GPU memory alone does not remove that encoding cost.

Before its five experiments, every Slurm job runs `check_backend.py` against
Qiskit at 14 qubits/64 layers, then the CUDA regression tests including training,
CV, bounded-cache parity, resume, and both objectives. Missing CUDA fails before
training. `uv sync --locked` installs this checkout's Linux CUDA dependencies once before submission; array
jobs use `--no-sync` so they never race to modify the shared environment. No
activation of an unrelated parent `venv` is needed. The lock contains CUDA 13.0
runtime packages, compatible in principle with the supplied 580-series driver
([NVIDIA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).
CUDA execution still needs verification on that allocation.

The supplied `prod10` partition exposes the named
`gpu:nvidia_a100_1g.10gb:1` MIG resource,
which is set in `slurm-prod10.sbatch`. Adapt the GRES type only if your site
differs. Pass site overrides to the wrapper, for example
`bash scripts/submit_sweep.sh --partition=prod20 --gres=gpu:1`. Keep Slurm's
`CUDA_VISIBLE_DEVICES` unchanged, including a MIG UUID
([NVIDIA MIG guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html)).

```bash
# One group (five experiments), inside an existing GPU allocation:
uv run --no-sync python scripts/training_sweep.py --group-id 0 --device cuda
# Rerun one failed experiment into a fresh folder:
uv run --no-sync python scripts/training_sweep.py --experiment-id 12 --device cuda
# Resume its existing repeated-split run to the original epoch target, skipping completed fits:
uv run --no-sync qcse resume-cv outputs/sweep/experiment-012/<run> --device cuda
# Local coverage: seven causal configurations, 16 sentences, 32 draws per epoch, two epochs per fit:
uv run python scripts/training_sweep.py --smoke --device mps --output outputs/pipeline-smoke
# A short cluster trial before committing to the full dataset:
uv run --no-sync python scripts/training_sweep.py --experiment-id 0 --device cuda --max-sentences 32 --epochs 2
# Optional longer training with exhaustive five-fold CV and both full sources:
uv run --no-sync qcse train --device cuda --folds 5 --epochs 150 --samples-per-epoch 5000
```

## Pipeline and paper mapping

1. `data.py` loads and cleans the selected sentence sources as described above.
   Frequency descending and alphabetical ties across the full source vocabulary
   give stable zero-based IDs. `1-3000.csv` is a source vocabulary list, not an
   input requirement. There is no lemmatization or stopword removal.
2. A CBOW example contains up to two words before and two after the center
   (`--window 4`), in sentence order, excluding the center. At boundaries the
   context shrinks; it never crosses a sentence. Singleton phrases yield no
   examples. The vocabulary dictionary is constructed across the corpus before
   splitting (a transductive ID mapping, with no learned test-set statistics).
   The default `--objective causal` is GPT-like: each target is the next word and
   its context is up to `--window` preceding words only. Use `--objective cbow`
   for the BERT-like bidirectional behavior from `main`. A causal model can be
   queried with `qcse embed --max-new-tokens 5 "the river"`.
3. `context.py` implements Eq. (24), the default exponential sinusoidal matrix:
   `C[i,j] = exp(-alpha*abs(i-j))*sin(omega*theta[i])*cos(omega*theta[j])+theta[i]`,
   with `theta[i] = 2*pi*word_id[i]/vocabulary_size`. Positions are zero-based
   within the center-excluded context, without leaving a gap for the center.
   `--method diagonal|phase|hash|angular` selects Eqs. (25)-(28).
4. Row-major flattening, trailing zero padding and sequential chunks yield
   `(L, m, 2)` angles, with `L = ceil(matrix.size/(2*m))`. Each pair feeds RX then
   RZ on one qubit; the equivalent paper matrix is `(2*m, L)`, with chunks as
   columns. No resizing, interpolation, normalization or features are discarded.
   Angular-vector encoding first pads to the next square as the appendix requests
   a square matrix, then uses the same circuit padding.
5. `circuit.py` applies H to every qubit once (Eq. 10), then RX/RZ and an adjacent
   CNOT cascade **per encoding layer** (Eqs. 11-14). Each of M ansatz layers applies
   RX/RZ on each qubit and adjacent CRZ gates (Eqs. 15-18). There are
   **M*(3*m-1)** trainable angles. The complete circuit is an ordinary Qiskit
   `QuantumCircuit`; `model.circuit(context_ids, measured=True)` adds measurements
   for execution on a sampling backend.
6. `model.py` returns the contextual vector `P(q=1) = (1-<Z_q>)/2` (Eqs. 19-20).
   These m numbers are **marginal bit probabilities**, not vocabulary softmax
   probabilities and not the full complex quantum state. Every occurrence has
   its own context and embedding. Targets encode the center word ID in binary.
   Vectors are in **q0-first / least-significant-bit-first** order; Qiskit printed
   bitstrings conventionally use the reverse display order. Threshold decoding
   can yield an unused vocabulary ID; `predicted_word` is then null.
7. `training.py` minimizes mean bitwise binary cross-entropy plus
   `lambda*sum(weights**2)`. Defaults follow the paper's 50 epochs, learning rate
   0.0003, and lambda 0.001. Mini-batch Adam uses seeded SPSA gradient estimates
   (two perturbed weight vectors evaluated in parallel per batch) to keep
   simulation practical. Context states are cached; their encoding is fixed.
   This optimizer, batch size 32,
   uniform [-pi, pi] initialization and perturbation schedule are explicit
   implementation choices, not a claimed exact reconstruction of the authors'
   unspecified training code. Tensor marginals are tested against Qiskit circuit
   outputs.

`--alpha 1 --omega 1 --delta 1 --prime 31 --hash-size 997` are explicit defaults
where the paper gives formulas but no values. The hash is the paper's integer
modular formula, not Python's randomized string hash. Train/test splitting uses
80/20 **sentence groups**, keeping duplicate phrases and overlapping windows in
the same partition. This is more conservative than an example-level split;
the paper does not specify split granularity. A seed controls initialization,
splitting, optional subset selection and training perturbations.

## Ambiguities in the paper

- Figures 2-3 draw controls below targets, while the prose/equations describe
  q -> q+1. The default `--direction forward` follows the equations; `reverse`
  follows the figures' control direction, retaining ascending adjacent-pair order.
- Tables report 3*m parameters per layer (21 for seven qubits), but the explicit
  ansatz specifies 2*m rotations and m-1 CRZ parameters, i.e. 3*m-1 (20 for seven
  qubits). This implementation follows the stated ansatz without inventing an
  extra ring-closing gate.
- Flattening/angle assignment and boundary handling are not fully specified.
  The conventions above are fixed, documented and tested.
- The final ansatz RZ/CRZ gates are diagonal and cannot change the measured Z
  marginals. They are retained to implement the circuit faithfully; consequently
  some last-layer parameters have no effect on the data loss. Also the very first
  encoding RX gates act on |+> and initially contribute only a global phase.
- The reported accuracy accepts a word when **at least half its bits match**.
  This is not exact word prediction and can be high by chance. History therefore
  reports this `paper_similarity_accuracy` alongside bit accuracy, exact-word
  accuracy, and unregularized BCE. Test data are evaluated but do not select
  checkpoints or contribute gradients. A smoke run only checks execution.

## Outputs and Python use

`prepare` creates `vocabulary.json`, `summary.json`, `example.json` (the first
context matrix and both angle layouts), `contexts.npz` (every context matrix and
its IDs/position), and an initialized `circuit.txt` / OpenQASM 3 `circuit.qasm`.
Ragged context matrices use flat arrays, offsets and shapes, without pickle:

```python
import numpy as np

with np.load("outputs/<prepare-run>/contexts.npz") as data:
    k = 0
    lo, hi = data["matrix_offsets"][k : k + 2]
    matrix = data["matrix_values"][lo:hi].reshape(data["matrix_shapes"][k])
```

With the legacy defaults `--folds 1` and full-pass epochs, `train` saves a canonical, resumable `run.npz`
archive after epoch 0 and after every completed epoch. It contains the model weights, optimizer state, random
number generator state, complete metric history, latest embeddings/results and
the data split, so `qcse continue outputs/<run>/run.npz --epochs N` adds N
epochs even after an interrupted run. It also writes `model.npz`,
`training_config.json`, `history.json` (epoch 0 and all trained epochs),
`embeddings.npz` and a trained example circuit. The embeddings archive contains
`probabilities`, `pauli_z`, `target_ids`, `sentence_ids`, `positions`,
`original_example_ids`, `train_ids`, `test_ids`. Split indices address rows of
that archive; original IDs address the complete corpus example list. With
`--max-examples`, only the sampled examples are exported; otherwise all are.
Every `train` and `prepare` invocation creates a unique timestamped subfolder
inside `outputs/`. `--output PATH` changes the parent directory, so repeated
invocations never replace a previous run. The exact directory is printed before
work starts. `run_info.json` records arguments, device/library versions, timestamps,
and completion/failure status. JSON and checkpoint archives use atomic replacement;
a failed write preserves the last complete checkpoint. The canonical `run.npz`
also contains corpus provenance, so a copied archive remains resumable.

`continue RUN/run.npz --epochs N` resumes in the same directory. Add `--output
outputs/resumed` to continue into a **new** subfolder without modifying the source
archive. `embed` requires an explicit `--model`; there is no ambiguous latest-run
selection. In example commands, replace `<run>` with the printed directory name.

With `train --folds 5 --epochs 150`, the run directory instead contains:

- `splits.npz`: development/test and all fold train/validation indices, addressing
  the full example list; `original_example_ids` maps back to corpus examples.
- `fold-01/` through `fold-05/`: each has `run.npz`, `model.npz`, and `history.json`.
  These histories contain train/validation metrics. Fold archives carry only
  development examples and use local split indices; their original IDs map to
  the full corpus. They contain no test examples.
- In sampled-epoch mode, each fold also has `full_validation.json`, scored on
  every validation example after training. Checkpoint embeddings are empty in
  this mode, and standalone sampled `train`/`continue` omit `embeddings.npz`;
  use the saved model for inference. CV still exports the full held-out test.
- `cv_history.json`: per-epoch train/validation metric means and sample standard
  deviations across folds (monitoring samples when sampled epochs are enabled).
- `refit/`: the final model and resumable checkpoint, with train-only history.
- `cross_validation.json`: final fold metrics, validation mean/std, and the single
  held-out test result; `test_embeddings.npz` exports held-out probabilities and IDs.

Use `qcse resume-cv outputs/<run> --device cuda` to resume this entire workflow
from its saved sentence/vocabulary/split snapshots to its original epoch target.
Completed fits are skipped. Use `refit/model.npz` for inference and plot a fold's
`run.npz` with `scripts/plot_training.py` to see validation curves. BCE progress
prints eight decimal places to expose small changes; flat exact-word accuracy can
persist when probabilities improve without crossing their decoding thresholds.
Resume preserves the saved sample budget and RNG stream; old full-pass runs keep
their original schedule. Start a new run to use the sampled-epoch protocol.

```python
from qcse import QCSEModel

model = QCSEModel.load("outputs/<run>/model.npz")
occurrences = model.embed_phrase("The river moved slowly under the bridge.")
print(occurrences[1]["embedding"])
lookup = {word: i for i, word in enumerate(model.vocabulary)}
context = [lookup[w] for w in ["the", "moved", "slowly"]]
circuit = model.circuit(context)
state = model.encode(context).evolve(model.bound_ansatz())  # full quantum state
print(circuit.draw())
```

Inference rejects unknown words rather than silently mapping them to another
word; retrain with an expanded dataset to add vocabulary. Simulation costs grow
exponentially with qubit count, despite the circuit's logarithmic qubit scaling.

Qiskit API reference: [Statevector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).
