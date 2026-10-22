# Research and literature

Primary sources reviewed on 2026-10-21/22, plus the branch's earlier paper audit.
This is a targeted review of this project's questions. Project deductions below
refer to our implementation, not automatically to every QCSE/QNLP architecture.
Numerical evidence lives in [runs](runs.md); executable work in
[next experiments](next-experiments.md).

## Hamiltonians and ansatz

[QCSE v2](https://arxiv.org/html/2509.05729v2) specifies numerical context
features, H/RX/RZ/CNOT encoding, RX/RZ/CRZ ansatz and marginal measurements.
Its context matrix is not a learned semantic Hamiltonian. The repository follows
the angle convention of equations 11–12 where equation 13 swaps labels. The
appendix leaves some padding/ordering choices unspecified; ours are documented
in [models](models.md), rather than presented as recovered author code.

For real angles, the gate generators used here are Hermitian:

```
RX(a)  = exp(-i a X/2)
RZ(b)  = exp(-i b Z/2)
CRZ(c,t,g) = exp[-i g (Z_t - Z_c Z_t)/4]
RZZ(g) = exp(-i g Z_c Z_t/2)
RXX(g) = exp(-i g X_c X_t/2)
```

CRZ includes a local Z field as well as ZZ. For the current directed chains
with free local RZ fields in every layer, commuting diagonal gates give

```
CRZ(c,t,g) = RZ_t(g/2) RZZ_ct(-g/2).
```

Keep RX angles, add half each incoming CRZ coupling to its target RZ angle and
halve/negate the edge angle. This invertible map yields the same whole unitary
at any input/depth. Our 18 float 64 state/input-gradient/parameter-gradient
checks pass with maximum discrepancy 3.55e−15. Default independent Gaussian
initializers do not map to the same distribution, and coordinate-wise Adam is
not invariant under the map. **Current ZZ/CRZ quality differences cannot show
greater expressivity.** Mapped initial functions are the useful next control.

Changing an RX/ZZ gate product into evolution under their summed generators
requires commutation or an explicitly controlled Trotter approximation. A
nonsymmetric numeric context matrix is not itself a Hermitian Hamiltonian.
The five context forms are feature encodings, not five established language
Hamiltonians. Circuit angles alone do not acquire a semantic interpretation.

A graph study is separate: construct a symmetric train-only co-occurrence graph,
define normalization/isolated nodes, then compare identity, `exp(-itL)`, heat
`exp(-tL)` and matched classical graph filtering. [Farhi & Gutmann 1998](https://arxiv.org/abs/quant-ph/9706062)
supplies a quantum-walk construction, not evidence that our graph helps NLP.
Use sparse exponential action, test Hermiticity/norm/t=0 and prevent held-out
edges; no full-corpus graph implementation or ETA exists yet.

## NLP and classical comparisons

QCSE's small English/Fulani experiments use a half-bits similarity criterion
and different classical/quantum training objectives. Their reported scores are
not exact-word top-1 on this corpus. Reproduce a paper metric separately from
the common valid-word endpoint; do not use either to rename the other.

| Source | Consequence for this benchmark |
| --- | --- |
| [Bengio et al.2003](https://www.jmlr.org/papers/v3/bengio 03a.html) | A learned lookup with feedforward conditional prediction is a serious NLP model. The MLP is a quality/cost reference, not a dispensable check. |
| [GPT generative pretraining](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) and [BERT 2019](https://aclanthology.org/N19-1423/) | Causal decoder and bidirectional masked encoder are different architectures/objectives. Our scratch word-ID adapters do not reproduce pretrained/tokenized GPT/BERT training. Left-input BERT is not GPT. |
| [Chen & Goodman 1996](https://aclanthology.org/P96-1041/) | Smoothing/backoff matters for count baselines. Compare normalized causal and gap-context references; unsmoothed failure would be a weak classical comparison. |
| [Quixer 2024](https://arxiv.org/html/2406.04305v1), sections 4/6 | An actual subsequent-word quantum Transformer experiment on Penn Treebank with matched context/stride and tuned classical references. Its Transformer reference wins the reported table. LCU/QSVT and hardware/simulation costs differ from our hybrid; its A100 duration is not a MIG forecast. |
| [Bowles, Ahmed & Schuld 2024](https://arxiv.org/abs/2403.07059) | Their binary-classification benchmark exposes sensitivity to design and strong classical baselines. This motivates controls; it does not determine an NLP ranking. |

The side-by-side study must separate equal exposure, equal tuning, equal model
capacity and equal compute. Current MLP-32/BERT have about twice the hybrid's
parameters. MLP-16 and the shared-encoder classical feature control address
that gap. Their [parameter/module differences](models.md#classical-references-and-the-gpt-gap)
remain explicit. Retain both causal and bidirectional research tracks; future
context is allowed only in the latter.

## Representation, readout and classical simulation

The independent-bit valid-word decoder is exactly a fixed binary-code head on
marginal log-odds. Identical marginals can arise from different joint quantum
states. A learned word head and a joint-Born valid-word distribution therefore
test different bottlenecks; the latter uses correlations without extra word-head
parameters. A final diagonal block remains invisible even to joint
computational-basis probabilities. Tests cover the current adapter's algebra;
the alternative heads still need implementation.

| Source | Consequence |
| --- | --- |
| [Schuld, Sweke & Meyer 2021](https://arxiv.org/abs/2008.08605) | Encoding gates constrain accessible Fourier frequencies. Vary encoding/readout and compare classical nonlinear features before adding depth. |
| [Schreiber, Eisert & Meyer 2023](https://arxiv.org/abs/2206.11740) | Classical surrogates motivate evaluating the prediction function, not just the quantum state dimension. This is not a theorem dequantizing every model. |
| [Kerenidis 2026, v2](https://arxiv.org/html/2607.24014v2), IV.1/Proposition 8 | Its passive-FLO/block-product construction has polynomial classical fixed-body readout evaluation. Sampling hardness does not establish readout hardness. The theorem is specific to that architecture, not our CRZ chain; older summaries miss the v2 qualification. |
| [Vidal 2003](https://arxiv.org/abs/quant-ph/0301063) | Restricted entanglement enables classical simulation. For our product-input CRZ/ZZ/XX chain, one rank-two gate crosses each cut per layer: Schmidt rank≤2^depth, hence≤4 at depth 2. Eighteen q8 state checks respect the bound. |

The chain bound gives an exact MPS candidate polynomial in qubits at fixed
depth, exponential in depth. It does not prove a measured speedup at q8 and
does not automatically cover the paper adapter's entangled encoder or rings.
No differentiable MPS backend has been implemented/profiled here.

## Plateau and optimization

The relevant term is **barren plateau**, but a flat score or harder optimization
at larger q is not a diagnosis. [McClean et al.2018](https://arxiv.org/abs/1803.11173)
studies gradient concentration for specified circuit ensembles;
[Cerezo et al.2021](https://www.nature.com/articles/s41467-021-21728-w) shows why
cost locality/depth assumptions matter. Those theorems cannot be transferred
unchanged to learned-head word cross-entropy.

[Larocca et al.2025](https://arxiv.org/html/2405.00781v2) distinguishes
probabilistic/deterministic concentration and the roles of data, circuit,
measurement and loss. Small variance is not enough when the mean gradient
remains useful. [Grant et al.2019](https://quantum-journal.org/papers/q-2019-12-09-214/)
offers identity-block initialization; our small Gaussian initializer is not
the same construction.

Our 9,216 initialization probes retain nonzero gradients through q12. Small-std
variance collapses much more than mean-gradient RMS. Recent q8 trained
histories have active layer gradients and improving monitors. These observations
neither establish nor exclude a later/larger-q plateau: the probe uses only
eight targets per batch and changing q changes encoder/head shape.

Next measure actual NLP loss and signed mean/variance/RMS at optimizer-sized
batches, with a controlled head/projection, multiple inputs/initializations and
trained checkpoints. Separate structurally dead final diagonal gates, encoder
saturation, batch cancellation, overfitting and poor exposure. Scaling fits over
three or four small q values are descriptive, not asymptotic evidence.

## Data mixture, sampling and generalization

The common training fold has 650 phrase-only sentences/4036 positive targets,
128739 cleaned-only/1010044 targets, and one shared sentence/five targets.
Phrases contribute about 0.4% of token mass; validation has only 159 phrase
sentences/998 targets. Uniform-sentence sampling is not balanced-source sampling.
Ten percent phrase mass is already about 25× the original weight;50% is about 125×.

[DoReMi 2023](https://arxiv.org/abs/2305.10429) motivates treating data weights as
an experimental variable, not importing its large-model optimum here.
Without importance weighting, a new sampler changes the training objective.
Handle shared membership fractionally, preserve nonzero eligibility and keep
evaluation fixed. Hash both probabilities and draw streams.

[Lee et al.2022](https://aclanthology.org/2022.acl-long.577/) motivates near-duplicate
audits beyond exact grouping. Existing source-capped experiments changed the
evaluation population and do not isolate mixture weights. A new source/template/
external test is needed for semantic generalization. Curated next-word scores
do not establish semantic understanding, NLI/STS quality or pretrained-LM parity.

## Statistical robustness

[Dror et al.2018](https://aclanthology.org/P18-1128/) and
[Bouthillier et al.2021](https://arxiv.org/abs/2103.03098) motivate matching
uncertainty analysis to the experimental unit and sources of variability.
Here sentence groups are paired; tokens/windows are not independent. Seed
resampling cannot replace training new seeds, and reused folds do not supply
an unseen population. Current intervals are conditional exploratory summaries.
See [protocol](protocol.md#metrics-and-uncertainty) for confirmation rules.

## Other QNLP formulations

[Meichanetzidis et al.](https://arxiv.org/abs/2005.04147) maps grammatical
DisCoCat diagrams to circuits; [lambeq](https://arxiv.org/abs/2110.04236) supplies
the parsing/rewriting pipeline. Our numeric context chain is not that model.
[Del Castillo, Zhao & Pei 2025](https://arxiv.org/pdf/2502.20744) compares ansätze
for text classification with tuning, not our word-prediction task.
[Sim et al.2019](https://arxiv.org/abs/1905.10876) provides expressibility and
entanglement descriptors; neither descriptor alone certifies useful language
representations. These sources inform controls, not transferable benchmark wins.
