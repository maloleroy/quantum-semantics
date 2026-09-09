# Experiment 2 — repeatedly change numerical word indices

**Question:** does the arbitrary numerical dictionary affect QCSE even when target
bitstrings and all training settings stay fixed?

Apply three independently seeded bijections (101, 202, 303) to INPUT word IDs before
constructing C. Each mapping is frozen across every context, epoch and train/test use.
This changes numerical angles, not sentence order. Keep output bits alphabetical.
A separate `joint-id-map` applies map 101 to targets too: it changes the prediction
code and must not be treated as a pure feature-order comparison.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| id-map-101 | 0.7170 | 0.7160 ± 0.0118 | 51.2% | 0.0% | 0.0% |
| id-map-202 | 0.7068 | 0.7032 ± 0.0051 | 53.6% | 0.0% | 1.2% |
| id-map-303 | 0.7119 | 0.7093 ± 0.0085 | 51.2% | 0.0% | 0.0% |
| joint-id-map | 0.7207 | 0.7171 ± 0.0031 | 51.2% | 0.0% | 0.0% |

All three input maps have lower mean test BCE than canonical in this small pilot.
Their means range from 0.7032 to 0.7160 versus 0.7214; no map was selected or reused
because of its score. None beats the canonical prior at 0.6758, and much of the gap
already exists before training. These results establish sensitivity to arbitrary IDs,
not a useful learned semantic ordering. Joint relabeling needs its own prior, stored
under `joint` in [classical controls](../artifacts/pilot/classical.json).

The original frequency dictionary is audited separately because it encodes full-corpus
frequency information; mixing it with these alphabetical-target comparisons would
confound label-bit imbalance and input geometry.

**Decision:** retain multiple fixed ID maps as a robustness check in future work.
Do not conduct a search for a lucky test mapping.

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
