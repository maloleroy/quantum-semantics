# Experiment 1 — paper fidelity and honest baselines

**Verdict:** the core implementation is a defensible equation-level implementation
of QCSE. The audit found no core formula or gate bug requiring a circuit rewrite.
It is not an exact reconstruction of unspecified author code and does not reproduce
the paper's performance. Documentation overstated the isolation of full-corpus
frequency-ranked IDs; that wording and the broken PDF path were corrected.

## Paper-to-code audit

Read the supplied [paper4.pdf](../../paper4.pdf), 15 pages, arXiv:2509.05729v2,
10 March 2026. PDF SHA256 `e01a1d2cb98c2ba74081597e0d5131ca977a0fe15cb1d1a37df805fd93160768`.

| Paper item | Implementation and verdict |
|---|---|
| Eq. 24, p. 7 | Exponential positional decay, sine/cosine word angles and additive identity term match. |
| Eqs. 25–28, pp. 13–15 appendix | Alternative context equations have direct formula tests; square padding for the angular vector is a documented convention. |
| Reshape, p. 4 | Row-major sequential chunks, trailing zeros; explicit convention where paper is incomplete. |
| Eqs. 10–14, pp. 4–5 | One initial H per qubit; RX then RZ then ascending adjacent CNOT cascade for every upload. Independently checked with dense matrices. |
| Eq. 13 | Display swaps the two angle indices relative to Eqs. 11–12. Implementation follows the preceding gate definitions. |
| Eqs. 15–18, p. 6 | RX/RZ then adjacent CRZ per ansatz layer; independent CRZ matrix test. |
| Figures 2–3 vs equations | Controls point the other way in figures; reference supports both, default follows equations. |
| Parameter count | Explicit ansatz has M(3m−1), so 58 at m=10, M=2. Tables use 3m; no invented ring gate added. |
| Gate count, Eqs. 21–23 | Formula omits the initial m Hadamards; actual circuit includes them. |
| Eqs. 19–20 | Exact Z marginals (q0 first), not softmax probabilities or a complete quantum state. |
| Section V.A, p. 7 | BCE and L2, rate 0.0003, 50 epochs match the miniature audit. SPSA/Adam details, initialization and batch size are implementation choices. |
| Data/split | Local 1,000-phrase corpus differs from paper's 110 English sentences/80 words. Sentence-text grouping is stricter than the unspecified paper split. |

The complete original test suite passed before changes. Expanded tests cover full
statevectors, canonical equivalence, ordering conservation, inactive gates, projection
isolation and the shared-unitary invariant. The pilot's largest independent Qiskit
marginal discrepancy was 1.22e-15.
The batch evaluator applies actual Qiskit gate matrices to complete complex states;
its speed comes from batching, not an approximation or learned shortcut.

## Complete miniature training

Used original frequency-ranked vocabulary, window 4, two ansatz layers, seed 11,
177 train / 57 test occurrences selected by whole sentence groups. Completed all
50 epochs at the paper's learning rate and penalty using the original Qiskit evaluator.
Wall time: 36.78 s, including diagnostic measurements.

| Model | Test BCE | Bit accuracy | Exact words | Paper half-bits metric |
|---|---:|---:|---:|---:|
| QCSE initial | 0.7283 | 47.4% | 0.0% | 54.4% |
| QCSE epoch 50 | 0.7200 | 47.9% | 0.0% | 56.1% |
| train_bit_prior | 0.5029 | 74.7% | 22.8% | 96.5% |
| context_bit_mean | 2.6731 | 64.4% | 0.0% | 93.0% |
| linear_bit_mean | 0.4964 | 73.5% | 15.8% | 94.7% |

The constant prior is estimated from training labels only, with add-one smoothing.
Its **96.5% paper metric** is not evidence of language understanding. For 10 independent
fair predicted bits, the expected half-bits success is 62.3047%, while exact matching
is only 0.09766%. Frequency-ranked IDs add substantial bit imbalance on this corpus.
This exposes why the paper metric cannot substantiate a quantum advantage.

## Structural limits checked numerically

- The first encoding RX on each |+> changes global phase only: half the first
  upload's assigned values are unobservable. Window 4 has one upload at 10 qubits.
- Final RZ and CRZ gates cannot affect Z marginals: at least 19/58 parameters are
  inactive for the data term in this two-layer reference circuit. L2 can still move them.
- A finite-difference probability Jacobian for one audit context had 31/58 nonzero
  columns; this single-context count is not a global parameter-rank estimate.
- Reversing the execution order of an uninterrupted CRZ block, retaining each edge's
  angle, changed the state by only 5.87e-17: those diagonal gates commute.
- A shared ansatz unitary preserves all pairwise full-state fidelities. This limits
  what “learning similar quantum states” can mean; see [experiment 5](05-semantic-bits.md).

**Conclusion:** suitable as a transparent research reference, with the documented
conventions. Predictive performance remains weak in this audit. Do not present weights
moving, falling training BCE, or the half-bits metric as proof of semantic learning.

Artifacts: [audit result](../artifacts/audit/result.json),
[50-epoch history](../artifacts/audit/history.json),
[raw predictions](../artifacts/audit/predictions.npz).
