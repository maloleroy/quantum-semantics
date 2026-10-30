# Learned product-state preparation: manual comparison

30 October 2026. Implemented and scientifically tested; training is **unrun**.
The completed [byte-token study](byte-bpe.md#completed-results) achieved input
coverage but poor prediction/generation. This experiment changes the initial
state while keeping the Hamiltonian core and direct vocabulary measurement.

## Hypothesis and physical model

The existing amplitude input places population on observed context token IDs.
Its quantum evolution must turn that representation into a predictive distribution.
This may be a limiting inductive bias; the completed results do not establish it
as the sole cause of poor performance. More training exposure is another unresolved
axis. This screen tests a different preparation rather than scaling the same fit.

For 512 tokens plus one auxiliary qubit, prepare ten zero qubits, then apply:

```
local RY(alpha(context)) -> exp(-i H0) -> local RZ(beta(context))
                        -> exp(-i H1) -> measure nine token qubits
```

Ignore the auxiliary result and retain every token outcome. Initial RY rotations
act on all ten qubits, including the auxiliary. The input is a separable product
state; the learned joint Hamiltonians may entangle it. There is no learned
classical vocabulary decoder or supplied count prior in this prediction path.

Both angle vectors use separate learned token tables. For each qubit, sum the
last-token entry, a .7-decayed normalized mean over the preceding 15 tokens,
and a .97-decayed normalized mean over older tokens. Add a learned global offset.
Tables start as independent Gaussian entries with standard deviation .1; offsets
start at zero. Padding contributes zero. The pooled lookup/mean operations are
classical controls, whose learning and inference costs must be counted.

For greater depth, the first table always controls RY preparation and each later
table controls RZ between Hamiltonian blocks. This screen uses two rank-64 blocks
and one interleaving stage. `--phase-scale` multiplies pooled angles before the
global offsets; at zero scale the predictor is context-independent. Product
models require `--encoder auto --decoder born`. `--amplitude-decay` is unused
by these models; the runner retains .9 in configuration for the historical
amplitude reference, not as an active product-state hyperparameter.

Input preparation now needs ten local RY gates with classically computed angles,
instead of arbitrary context-amplitude loading. This is a preparation change,
not an established end-to-end hardware speedup: generic learned low-rank
Hamiltonian compilation, optimization/shot costs and noise remain unresolved.

## Parameters and independently trained controls

The two Hamiltonians retain 262,272 real parameters. Two angle tables add 30,720
parameters and global offsets add 20, for **293,012** total. About 89.5% are in
the quantum dynamics; parameter placement alone does not prove useful quantum
computation. The previous amplitude model has 277,632 parameters, so comparison
with it changes preparation and controller capacity, not only state geometry.

Five new fits:

- `product`, full 128-token context: train preparation, later angles and dynamics.
- `product`, independently retrained on the existing 16-token masked dataset.
  Allocated parameter count matches; older-context table entries are inactive.
- `product_fixed`: freeze both angle tables and offsets; train only dynamics.
  All 30,740 fixed controller entries remain stored and counted as resource costs.
- `product --freeze-core`: freeze both Hamiltonians; train only the 30,740
  controller/offset parameters. This tests whether trained dynamics contributes.
- `product_mlp`: independently learn the same angle/preparation family and replace
  quantum evolution with a tanh MLP receiving the 1,024 real prepared amplitudes
  and ten later RZ angles. Total **292,695** parameters. It has the same feature
  family and information access; its fitted tables need not equal the quantum ones.

The classical control also pays for constructing its prepared-state features.
It is an approximately parameter-matched replacement, not a claim of equal
hardware compute. The fixed/frozen controls have different trainable budgets;
interpret their specific attribution questions rather than generic superiority.

## Declared protocol and frozen reference

Reuse the exact byte-BPE full/short datasets from
`output/byte-bpe-screen/batch-IBqf0v/data/`: 32,768 training and 4,096 validation
targets, the same tokenizer, official article partitions, train-only statistics,
and target byte lengths. No tokenizer refit. No test scoring.

All new fits use seed 11, 30 epochs, batch 512, Adam LR .003, clipping 5,
NLL training and validation NLL checkpoint selection including epoch zero,
float64/complex128, one Torch thread each. No LR search is planned in this screen.
Report token NLL, sampling success, argmax and sampled-target bits per byte using
the same denominator and limitations as the byte-token protocol. These are
single-seed adaptive validation fits, not fresh held-out confirmation.

Reuse the already verified amplitude-model prediction arrays as a fixed reference:
`batch-IBqf0v/learned/run-20261030T065359Z-d2277bd0`, NLL 4.277301 and sampling
5.6766%. Its weights reproduced these arrays before source changes, recorded in
commit `c08e614`. The generated runner pins its selection, weights, predictions
and metadata. A source snapshot from commit `1de6955` matches all hashes recorded
in its selection. The preflight checks that snapshot, its dataset and weight
locks. It does not re-score old weights with changed model source or bypass the
current-source loader. This avoids repeating an already completed fit while
preserving its provenance. New fits use the normal current-source selection lock.

The summary compares paired targets/groups/byte lengths, recomputes all new
validation predictions, and includes all-training unigram and bigram references.
Their count exposure exceeds the gradient target subset. Bootstrap intervals
are descriptive article-group intervals conditional on this seed and reused
validation, with model-selection uncertainty excluded.

Generate 64 direct draws for each new model and count reference from the same
three prompts/seeds declared in the byte-token screen. Preserve exact token IDs
in NPZ and rendered text in CSV; no sample filtering/retries/temperature changes.
The prior amplitude samples remain in their original batch. Qualitative samples
are diagnostics, not a text-quality benchmark or proof of useful long context.

## Verification and manual handoff

All 51 tests, strict Pyright and Ruff pass. New scientific checks compare product
preparation to explicit little-endian RY gates, verify its initial separability,
and match full probabilities and gradients to dense Hamiltonian/local-rotation
references. Checks also cover fixed/frozen controls, parameter budgets and
context independence at zero phase scale. Generic checks exercise probability
normalization and portable checkpoints for all three new model kinds.

Generated tools and the archived reference source live in `output/product-screen/`.
Runner syntax, imports, source/data/reference checks and the no-fit preflight
were checked. The full training/summary pipeline has not been run. No new fits
were launched or monitored by the assistant.

```bash
bash output/product-screen/run.sh
```

Allow roughly 10–15 minutes for five concurrent one-thread fits and summaries.
This is an unmeasured estimate based on the previous six-minute fits, with extra
preparation/control computation; it is not a timing result for the new model.
A checksum manifest, lock and process guard protect the run. If interrupted,
resume with its printed batch path to preserve completed fits and restart only
missing fits from initialization:

```bash
bash output/product-screen/run.sh output/product-screen/batch-XXXXXX
```

Competitive generation and a justified quantum advantage remain unproven.
