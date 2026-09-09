# Experiment 5 — semantic bit families and actual quantum proximity

**Question clarified by the user:** can related words have nearby codes, potentially
with some bits representing semantic attributes?

A simple balanced recursive partition of normalized Qwen word vectors assigns unique
10-bit codes. At each node, two far-apart words define a projection; split its sorted
scores in half, placing the decision in the next high bit. There is no supervised
attribute label, nearest-neighbor optimization, or search over codebooks.

Compare with alphabetical IDs and a random reassignment of the EXACT same semantic
code set. Codes are external pretrained information. For the predictive experiment,
only center-word TARGET codes change; QCSE context input values remain alphabetical.

## Static lexical geometry

| Codebook | Qwen nearest-5 mean Hamming | Neighbors within 3 bits | Random-pair Hamming | Unique codes |
|---|---:|---:|---:|---:|
| alphabetical | 4.242 | 34.9% | 4.881 | 729 |
| semantic | 4.370 | 31.9% | 4.874 | 729 |
| shuffled_semantic | 4.874 | 18.9% | 4.866 | 729 |

Semantic codes improve locality over shuffled codes, but **do not beat alphabetical
IDs**. Only 31.9% of the five nearest Qwen neighbors land within three bits, below
alphabetical's 34.9%. Alphabetical ordering can retain word-form/morphology structure.
This is a sanity check against the SAME pretrained space used to build the codebook,
not independent semantic validation. No claim that bit 6 means gender/species/quality
is supported: naming bits requires independent labels and held-out attribute probes.

## Context prediction of semantic bits

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| semantic-codes | 0.7285 | 0.7235 ± 0.0127 | 48.8% | 0.0% | 0.0% |
| random-codes | 0.7342 | 0.7288 ± 0.0184 | 50.2% | 0.0% | 0.0% |

Semantic targets yield mean BCE 0.7235 versus shuffled-code 0.7288, with no exact
word predictions. Bit losses across different codebooks have different priors; consult
the codebook-specific [classical controls](../artifacts/pilot/classical.json).
This does not show that QCSE learned useful semantic families.

## Why Hamming-close is not automatically quantum-close

For distinct computational basis bitstrings, fidelity is zero regardless of whether
one bit or ten bits differ. Further, the shared QCSE ansatz U obeys

`|<Uψ | Uφ>|² = |<ψ | φ>|²`.

On all held-out context pairs, the largest change was 4.00e-15.
The correlation between target Qwen cosine and full-state fidelity was about −0.0434,
unchanged across codebooks/weights except tiny tie-rounding effects. Marginal-distance
correlations stayed close to zero (roughly −0.027 to +0.050). Shared pair endpoints
are not independent samples; no pairwise p-values are reported.

A separate genuine nonorthogonal representation is

`tensor_product_q Ry((pi/3) * bit_q) |0>`.

Its fidelity is `(3/4)^Hamming`. This formula was checked against Qiskit circuits.
Mean nearest-neighbor fidelity is 0.3392 for alphabetical codes, 0.3240 for semantic,
and 0.2724 for shuffled codes. This construction achieves actual graded proximity,
but is a separate product-state representation, not trained QCSE or quantum advantage.

**Decision:** the semantic-bit idea needs a better code objective and independent
attribute/semantic evaluation. If the goal is trainable full-state geometry, use a
context-dependent trainable encoder or interleaved data uploads and trainable blocks;
changing only this common terminal unitary cannot achieve it.

Artifacts: [code geometry](../artifacts/pilot/code_geometry.json),
[state geometry](../artifacts/pilot/state_geometry.json),
[diagnostic source](../research/state_geometry.py).

## Protocol and reading the numbers

177 train occurrences from 24 complete sentence-text groups; 57 test occurrences
from 8 disjoint groups. Full 729-word alphabetical vocabulary (transductive dictionary),
window 8, 10 qubits, 2 ansatz layers, 20 epochs, Adam-SPSA, learning rate 0.003,
L2 0.001, batch 32. Paired initialization/optimizer seeds 11, 23, 37. Settings were
fixed before outcomes; every run is retained. No test-selected checkpoints or tuning.

SD describes variability across only three runs on ONE split, not a confidence
interval or evidence of population significance. There are 99 distinct training targets
and 37 test targets; 26/57 test occurrences have targets absent from the selected
training subset. No identical context tuple crosses the split. Results are a sensitivity
pilot, not converged language performance or a replication of the paper's corpus.

BCE is mean binary cross-entropy (lower is better). Exact words require every bit
correct. Valid-code word decoding ranks the allowed codebook using a product of
marginals; it is not joint Born-probability decoding. The canonical train-only bit prior
has test BCE **0.6758**. All quantum conditions here have 0% threshold exact-word accuracy.

Sources and reproduction: [protocol](../artifacts/pilot/protocol.json),
[all results](../artifacts/pilot/results.json), [runner](../research/run_suite.py),
[full progress log](../PROGRESS.md). Each run folder stores weights, codebook, raw
probabilities, initial weights, complete history, and train/test indices.
