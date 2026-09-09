# Experiment 4 — rotation and entangler execution order

**Question:** do noncommuting gate-order choices materially change the circuit?

`rz-before-rx` executes RZ then RX in BOTH encoding and ansatz, while retaining each
angle on its original axis. It is a joint architectural change; this pilot does not
isolate encoding order from ansatz order. `reverse-cnot` reverses execution along
the same directed edges (q→q+1); it does not reverse control/target roles or add gates.
Parameter count, inputs and number of operations stay fixed.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| rz-before-rx | 0.7457 | 0.7308 ± 0.0265 | 50.4% | 0.0% | 0.0% |
| reverse-cnot | 0.6990 | 0.6973 ± 0.0033 | 50.3% | 0.0% | 0.0% |

Reverse CNOT order has test BCE 0.6973 versus 0.7214. But its initial mean is already
0.6990, so most of this difference is an initialization/feature-map effect. It remains
worse than the prior at 0.6758 and yields no exact predictions. Reversing RX/RZ is
more variable and worse on average (0.7308); it exposes first-upload RX information
but that alone does not guarantee useful learning.

A control reversing only CRZ gates within each uninterrupted CRZ block agrees in
full state to 5.87e-17. Diagonal CRZ gates commute; an apparent large
advantage from their order alone would indicate a changed edge-angle assignment,
other intervening gates, a simulator error, or an uncontrolled experiment.

**Decision:** gate order changes behavior as expected, but no evidence here justifies
calling the better initial BCE an improved semantic model. No test-driven gate search.

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
