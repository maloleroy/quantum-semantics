# Experiment 7 — bit-prior controls and shuffled training labels

These were preregistered controls for the other experiments, not a replacement for
the user's semantic-bit request. Canonical output codes and the same held-out split:

| Classical method | Test BCE | Bit accuracy | Exact words | Trainable parameters |
|---|---:|---:|---:|---:|
| train_bit_prior | 0.6758 | 56.8% | 0.0% | 10 |
| context_bit_mean | 1.4264 | 46.7% | 0.0% | 0 |
| linear_bit_mean | 0.6957 | 59.3% | 1.8% | 110 |
| linear_qwen_projected | 0.6819 | 60.2% | 14.0% | 210 |

The prior is add-one-smoothed training target frequency per bit, constant across
contexts. Raw context-bit means are untrained and can be exactly 0 or 1; BCE uses
the reference clipping epsilon 1e-10, so confidently wrong bits receive large loss.
Both logistic heads use 400 deterministic full-batch Adam updates at 0.05 and L2
0.001 on weights only. Means/standard deviations and gradients use training data only.
No classical hyperparameters were selected using test data. Parameter counts are not
matched to the quantum model, and no optimization-budget equivalence is claimed.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| shuffled-labels | 0.7256 | 0.7203 ± 0.0101 | 51.6% | 0.0% | 0.0% |

The shuffled-label experiment permutes center targets only within the training set,
keeping contexts, label frequencies and the entire test set unchanged. All final
scores above are against real labels. Its internal training history intentionally
measures shuffled training labels, clearly flagged in the run configuration.

Shuffled-label test BCE (0.7203) is essentially tied with real-label training (0.7214).
This is strong reason not to interpret the pilot's small BCE differences as learned
contextual semantics. The lower constant-prior loss reinforces that limitation.

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
