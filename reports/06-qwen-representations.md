# Experiment 6 — local Qwen geometry, layout-only use and direct representations

**Question:** does pretrained semantic information help through placement alone, or
through direct numerical representations?

## Local model and extraction

Used `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` on the local Mac, CPU float32, four
threads, batches of 16. Last nonpadding token with left padding, L2 normalized,
no prompt. 729 isolated words → 729×1024 vectors, maximum four tokens per word.
These are pretrained word vectors, not occurrence-contextual sentence embeddings.

Download/load took 49.53 s; inference 7.91 s; peak process RSS was 1.88 GiB. No remote
inference API was used. Followed the [official Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
for model size and pooling. [Metadata](../artifacts/qwen/metadata.json) records vector
hashes, immutable revision, device and extraction details.

## Layout-only experiment and topology control

Compute original C first, retaining its original linguistic distances. A greedy
nearest-neighbor path through context-word Qwen vectors reorders its rows AND columns
before serialization. Every numerical C entry is preserved. Pretrained vectors decide
placement only. A shuffled word-to-vector assignment uses the same vector distribution.
The paired no-entanglement controls remove all encoding CNOT and ansatz CRZ gates.
They have 40 circuit parameters versus 58, while retaining 58 optimizer slots for
paired initialization. This is a natural topology ablation, not a capacity-matched one.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| qwen-layout | 0.7201 | 0.7172 ± 0.0097 | 49.6% | 0.0% | 0.0% |
| shuffled-qwen-layout | 0.7236 | 0.7186 ± 0.0187 | 50.2% | 0.0% | 0.6% |
| qwen-layout-no-entanglement | 1.0465 | 1.0065 ± 0.0486 | 49.5% | 0.0% | 0.0% |
| shuffled-layout-no-entanglement | 1.0043 | 0.9705 ± 0.0199 | 50.2% | 0.0% | 0.0% |

Define benefit as shuffled BCE minus semantic BCE (positive favors semantics).
The local-chain benefit is +0.0014; without entanglers it is
-0.0360. The paired difference is +0.0374 with SD
0.0677. This tiny, three-seed pilot cannot establish a topology mechanism:
semantic layout is nearly tied with shuffled layout on the chain, and both
no-entanglement versions are much worse. The comparison also changes expressivity
and phase-to-amplitude mixing, not only a notion of semantic locality.

## Direct representations and classical comparison

Mean each context's Qwen word vectors. Fit PCA on TRAIN context means only, retain
20 dimensions, divide by training standard deviations and multiply by 0.5 radians.
The quantum encoder consumes these 20 angles in one upload. A classical logistic
predictor receives exactly the same 20 features (400 full-batch Adam steps, rate 0.05,
L2 0.001, train-only standardization, 210 parameters). Quantum has 58 slots.
Budgets and parameter counts differ and are disclosed; no equal-compute claim.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| qwen-direct | 0.7197 | 0.7061 ± 0.0205 | 53.3% | 0.0% | 0.0% |

The classical projected-Qwen head has test BCE **0.6819**, exact-word accuracy
**14.0%** (8/57), versus quantum BCE **0.7061 ± 0.0205**, exact 0%. The constant
bit prior is still slightly better on BCE (0.6758), while predicting no exact words
under alphabetical IDs. These metrics answer different questions.

Direct Qwen is not a clean replacement isolating just semantics: it averages positions,
compresses features, and has one upload versus up to four for native QCSE. Also the
reference first-upload RX gates discard half of these angles as global phase. The
classical control exposes information potentially lost in this quantum map; no
improvement should be attributed automatically to quantum processing.

**Decision:** no positive evidence for Qwen layout-only benefit here. Direct classical
Qwen features are more promising on exact words, but need a larger held-out sample.
Prioritize a small learnability/feature-retention test over another layout sweep.

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
