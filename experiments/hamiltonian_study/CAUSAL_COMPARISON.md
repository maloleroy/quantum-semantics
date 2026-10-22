# Three-way causal encoder comparison

This follow-up answers the requested comparison, with exactly three plotted systems:

1. QCSE encoding + the existing causal QCSE ansatz.
2. Fixed Hamiltonian encoding, with no trainable ansatz.
3. The same fixed Hamiltonian encoding + the same causal QCSE ansatz.

In the repository, `objective="causal"` means past-only context construction and
next-word prediction. It does not select a different gate topology in
`ansatz_circuit`. All three systems here use that past-only objective, window 8,
and the same saved Tatoeba split, vocabulary, and 8,192 test examples. No CBOW
or future-word context is used.

To isolate the encoder and the added circuit, all three use the existing Z-marginal
decoder: independent-bit probabilities normalized over valid vocabulary entries.
None uses joint Born word readout, learned measurement axes, or unigram mixing.
Consequently, the Hamiltonian-only score is a new matched comparison and is not
the differently decoded static-walk score in the main report.

The fixed encoder loads square roots of recency-weighted counts (decay 0.7), then
applies `exp(-i t H)` using the training-only count graph with 24-neighbor threshold
pruning and symmetric normalization. The no-ansatz validation NLL chooses `t=16`
from the declared grid 0, 0.25, 0.5, 1, 2, 4, 8, 16. This identical frozen encoder
is used in both Hamiltonian arms. It is not the report's jointly trained,
alternating graph/context circuit.

Both trained arms use 16 forward RX/RZ/CRZ layers (416 parameters), seeds 11/22/33,
30 epochs over the same 8,192 training targets, Adam at 0.01, batch size 128,
categorical NLL plus the existing 0.001 angle L2 penalty, and norm clipping at 5.
Validation NLL at epoch 0 and every five epochs selects checkpoints. The QCSE
arm reuses the existing `main/qcse_nll` causal runs; the Hamiltonian+ansatz arm
is trained afresh. Random initial weights and optimization settings match by seed.

The selection manifest is written before this follow-up's test evaluation. The
test set was already used in the main study, so this is a follow-up comparison,
not an independently untouched replication. Error bars show seed standard
deviation, not confidence intervals; the fixed no-ansatz arm is deterministic.

Reproduce from the QCSE repository:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python -m experiments.hamiltonian_study.causal_comparison prepare
uv run python -m experiments.hamiltonian_study.causal_comparison train
uv run python -m experiments.hamiltonian_study.causal_comparison figure
uv run python -m experiments.hamiltonian_study.causal_comparison render --language fr
uv run python -m experiments.hamiltonian_study.causal_comparison render --language fr --transparent
```

The figure and compact metric records are under `results/causal_comparison/`.
Independent tests compare the fixed encoder to a matrix exponential and both
zero-layer and four-layer compositions to the production Qiskit circuit and an
independently calculated decoder.

## Held-out results

| Configuration | NLL (lower is better) | Top-1 | Top-5 |
|---|---:|---:|---:|
| Baseline QCSE encoding + causal ansatz | 5.182640 | 7.47% | 22.18% |
| Hamiltonian encoding without ansatz | 5.347671 | 5.07% | 15.59% |
| Hamiltonian encoding + causal ansatz | 5.216199 | 9.66% | 26.42% |

Adding the causal ansatz improves all three metrics over Hamiltonian encoding
alone. Compared with baseline QCSE, Hamiltonian plus the causal ansatz improves
top-1 by 2.19 percentage points and top-5 by 4.24 points, with NLL worse by 0.03356.
Better top-k rankings do not require better probabilities across the full vocabulary.
These statements describe this matched, marginal-readout experiment; they do not
replace the differently defined Hamiltonian models in the main report.
