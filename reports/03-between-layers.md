# Experiment 3 — layouts and ID maps between upload layers

**Question:** does changing the layout or dictionary between uploads help beyond
using one fixed change everywhere?

- `fixed-pairs`: one seeded qubit permutation reused on every upload; RX/RZ pairs
  stay together, including padded zero pairs.
- `layer-pairs`: independently seeded pair placement per upload (101+layer).
- `reverse-uploads`: reverse complete upload chunks, preserving every padded value.
- `layer-id-maps`: recompute C with a frozen dictionary seeded 101+layer and take
  that layer's original chunk. Compare directly to `id-map-101`, whose first upload
  is identical. This changes numerical features, not just a permutation of C.

All maps remain fixed during optimization and inference, so caching is valid. These
are between-layer changes, not resampling every batch or epoch.

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| fixed-pairs | 0.7204 | 0.7165 ± 0.0088 | 49.9% | 0.0% | 0.6% |
| layer-pairs | 0.7198 | 0.7163 ± 0.0097 | 49.7% | 0.0% | 0.6% |
| reverse-uploads | 0.7163 | 0.7128 ± 0.0052 | 52.7% | 0.0% | 0.0% |
| id-map-101 | 0.7170 | 0.7160 ± 0.0118 | 51.2% | 0.0% | 0.0% |
| layer-id-maps | 0.7147 | 0.7137 ± 0.0091 | 53.0% | 0.0% | 0.0% |

Actual upload counts across 234 examples: 64 have one, 143 have two, 24 have three,
and 3 have four. Thus 170/234 examples genuinely exercise multiple uploads; effects
are diluted by the 64 one-upload cases. No extra uploads or features were invented.

Layer-specific pair placement changes BCE by -0.000156 versus fixed
placement (paired SD 0.000889); that difference is negligible here.
Reverse uploads average 0.7128, and layer-specific dictionaries 0.7137. Neither beats
the prior. First-upload RX values are invisible, so reversing chunks also changes
which features survive; temporal-memory claims would require separating this effect.

**Decision:** no reason from this pilot to pursue extensive layer-layout searches.
The fixed-vs-layer-specific control is worth keeping once a model learns reliably.

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
