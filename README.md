# QCSE in Qiskit

Implementation of **QCSE: A Pretrained Quantum Context-Sensitive Word Embedding
for Natural Language Processing**, [arXiv:2509.05729v2](https://arxiv.org/abs/2509.05729),
using the local [`../References/2509.05729v2.pdf`](../References/2509.05729v2.pdf).
This project trains its own weights on `phrases.csv`; it does not include the
paper authors' pretrained weights or claim to reproduce their reported accuracy.

## Run

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
uv run qcse prepare
# Quick end-to-end training check, with the full vocabulary and a random subset:
uv run qcse train --epochs 2 --max-examples 64 --output outputs/smoke
uv run qcse embed --model outputs/smoke/model.npz "The river moved slowly under the bridge."
# Full corpus: potentially slow on a CPU; progress is printed after each epoch.
uv run qcse train --epochs 50 --layers 2
uv run python scripts/plot_training.py outputs/train/run.npz -o outputs/train/training_metrics.png
# Add epochs to an interrupted or completed run:
uv run qcse continue outputs/train/run.npz --epochs 25
uv run pytest
uv run ruff check .
uv run pyright
```

### Parameter-grid visualization

The parameter sweep in [`data/grid.txt`](data/grid.txt) can be rendered as
annotated heatmaps. The generated image is kept under the ignored `outputs/`
directory:

```bash
uv run python scripts/plot_grid.py data/grid.txt -o outputs/grid_heatmap.png
```

The default dataset is resolved relative to the source checkout, not the working
directory; no file selection or download is needed. From elsewhere, use
`uv run --project /absolute/path/to/QCSE qcse prepare`. `--data` and `--output`
accept other paths. `qcse prepare --help` / `qcse train --help` list all settings.
No IBM account, backend credentials, Aer, GPU or classical pretrained embedding
is needed. Qiskit's exact `Statevector` simulator computes the marginals.

## Pipeline and paper mapping

1. `data.py` reads the headerless, one-phrase-per-line `phrases.csv`, lowercases,
   removes punctuation, preserves internal apostrophes, and derives a vocabulary
   from the observed words. Frequency descending and alphabetical ties give
   stable zero-based IDs. `1-3000.csv` is a source vocabulary list, not an input
   requirement: unused words are not allocated IDs. There is no lemmatization or
   stopword removal. The supplied corpus gives 1,000 phrases, 7,235 tokens,
   **729 words and 10 qubits**, with one example per token.
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
   (two perturbed evaluations per batch) to keep simulation practical. Context
   states are cached; their encoding is fixed. This optimizer, batch size 32,
   uniform [-pi, pi] initialization and perturbation schedule are explicit
   implementation choices, not a claimed exact reconstruction of the authors'
   unspecified training code. Embeddings remain Qiskit circuit outputs.

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

with np.load("outputs/prepare/contexts.npz") as data:
    k = 0
    lo, hi = data["matrix_offsets"][k : k + 2]
    matrix = data["matrix_values"][lo:hi].reshape(data["matrix_shapes"][k])
```

`train` saves a canonical, resumable `run.npz` archive after epoch 0 and after
every completed epoch. It contains the model weights, optimizer state, random
number generator state, complete metric history, latest embeddings/results and
the data split, so `qcse continue outputs/train/run.npz --epochs N` adds N
epochs even after an interrupted run. It also writes `model.npz`,
`training_config.json`, `history.json` (epoch 0 and all trained epochs),
`embeddings.npz` and a trained example circuit. The embeddings archive contains
`probabilities`, `pauli_z`, `target_ids`, `sentence_ids`, `positions`,
`original_example_ids`, `train_ids`, `test_ids`. Split indices address rows of
that archive; original IDs address the complete corpus example list. With
`--max-examples`, only the sampled examples are exported; otherwise all are.
Repeated runs to the same output directory replace those generated files.

```python
from qcse import QCSEModel

model = QCSEModel.load("outputs/train/model.npz")
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
