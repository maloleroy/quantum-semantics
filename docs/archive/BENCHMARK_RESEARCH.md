# QCSE, Hamiltonians and classical NLP — research audit

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


## What the current model computes

[QCSE v2](https://arxiv.org/html/2509.05729v2) specifies a numerical context
matrix, H/RX/RZ/CNOT encoding, RX/RZ/CRZ ansatz and single-qubit Z readout.
It is not a learned semantic Hamiltonian. Equation 13 swaps the angle labels
used in equations 11–12; the repository follows 11–12 and forward chain edges.
The paper's bit similarity threshold is not exact vocabulary top-1. Its small
English/Fulani evaluations do not establish a matched BERT comparison.

The following are direct algebraic observations and code-audit findings:

Let P1(c)=(I-Zc)/2. With hbar=1,

```
RX(a) = exp(-i a X / 2)
RZ(b) = exp(-i b Z / 2)
CRZ(c,t; g) = exp(-i g P1(c) Z(t) / 2)
             = exp[-i g (Z(t)-Z(c)Z(t)) / 4]
RZZ(c,t; g) = exp(-i g Z(c)Z(t) / 2)
RXX(c,t; g) = exp(-i g X(c)X(t) / 2)
```

Thus CRZ couples a ZZ interaction to a local Z field. It is not RZZ; substituting
one changes the model. All displayed generators are Hermitian for real weights,
so their exponentials are unitary. A nonsymmetric real context matrix cannot
itself serve as a Hermitian Hamiltonian. An undirected graph construction could
use J=(C+C.T)/2 and H=sum J_ij Z_i Z_j plus local fields, but token-to-qubit
assignment, normalization and time would then be new design choices.

For any diagonal D, D† Z_q D=Z_q. The final RZ/CRZ/RZZ block therefore has zero
influence on Z-only marginals and zero exact gradient. Earlier diagonal layers
can matter through subsequent noncommuting rotations. XYZ readout retains phase
information. A product of RX and ZZ evolutions is an ordered variational circuit;
it equals evolution under their sum only under appropriate commutation or a
specified Trotter approximation. No such equality is claimed here.

The `H=J X⊗X` example in `../french/paper1.md` is a possible physical interaction,
not an empirically fitted language dynamics. RXX supplies a controlled executable
alternative; an XX-chain ansatz tests that interaction family without inventing
semantic time. Chain/ring CRZ, ZZ, XX, RY/CNOT, local-only and identity controls
separate generator and connectivity choices. CNOT is a fixed discrete entangler.

IDs are frequency-ranked in the full curated vocabulary. The QCSE trigonometric
formula treats those ranks as geometry, without proving semantic distance.
A fixed random ID permutation, applied to encoder IDs only, measures sensitivity
while preserving words, target IDs and partitions. The five context forms in
`src/qcse/context.py` are feature constructions, not five Hamiltonians.

## BERT comparison

[Devlin et al., BERT](https://aclanthology.org/N19-1423/) uses token, position and
segment embeddings, stacked bidirectional multihead attention with softmax,
feed-forward layers, residual connections, layer normalization and an MLM head.
Its original pretraining also includes next-sentence prediction and WordPiece.
CBOW context alone is insufficient to call a model BERT.

This benchmark instantiates the actual randomly initialized Hugging Face
`BertForMaskedLM` architecture using the installed locked dependency, without
downloading weights. [BERT API documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert)
documents the configuration and MLM interface. A target mask is inserted into
the shared word context. The causal track supplies only the prefix; the
bidirectional track supplies both sides of the absent center word. Both are
explicit controlled adaptations: no WordPiece, no external pretraining, no NSP,
and one prediction target per window rather than BERT's 15% corruption recipe.
Neither track is a reproduction of original BERT pretraining.

| Property | Paper QCSE adapter | Semantic hybrid | BERT from scratch |
| --- | --- | --- | --- |
| Input | Fixed rank-angle formula | Learned lookup and angle projection | Learned token/position/type embeddings |
| Context mixing | Unitary gates | Unitary gates | Softmax multihead attention + FFN |
| Readout | Z marginals | XYZ → tied word head | MLM transform → tied word head |
| Training | Bit BCE or valid-word CE arm | Word CE | Word CE |
| Capacity scaling | O(depth × log vocabulary) trainable angles | Lookup dominates | Lookup + Transformer blocks |
| Simulation | Exact statevector, exponential in qubits | Exact small register | Dense classical tensors |

Marginal bit probabilities do not determine joint quantum measurement
probabilities. The adapter's word distribution is explicitly an independent-bit
product normalized over valid vocabulary IDs; it is not statevector fidelity or
a Born joint distribution. BCE and word CE remain distinct objectives, even
when evaluated by the same top-k/NLL metrics. Both QCSE objectives are separate
manifest arms. Parameter counts, optimizer updates, processed examples, wall
time and device peak memory accompany every fit. The common batch budget is
exposure/update matched, not automatically parameter/FLOP matched. A classical
MLP and an equal-width Transformer complement the BERT reference. Report actual
capacity differences rather than declaring quantum advantage.

## Literature implications

- [Meichanetzidis et al., QNLP](https://arxiv.org/abs/2005.04147) maps grammatical
  DisCoCat diagrams to circuits. This differs from serializing fixed numeric
  context features; our chain is not a parsed sentence diagram.
- [Kartsaklis et al., lambeq](https://arxiv.org/abs/2110.04236) provides a modular
  parsing/rewriting/ansatz pipeline. No lambeq dependency is needed to test the
  present gate families, and this benchmark does not implement DisCoCat.
- [Sim et al.](https://arxiv.org/abs/1905.10876) studies expressibility and
  entangling capacity as circuit descriptors. These alone cannot certify useful
  NLP representations; evaluate the same task and retain classical controls.
- [McClean et al.](https://arxiv.org/abs/1803.11173) identifies vanishing-gradient
  regimes for broad circuit classes. More depth is not inherently better;
  log gradient norms and compare shallow/deeper circuits.
- [Dror et al.](https://aclanthology.org/P18-1128/) explains why the statistical
  procedure must fit the task and experimental unit. Here overlapping windows
  are dependent: paired bootstrap resamples sentence groups, not tokens.

Prediction on these curated corpora is a language-model benchmark. Downstream
semantic similarity, inference, hardware noise, quantum runtime and quantum
advantage require separate evidence and are not inferred from these scores.

The [trainability investigation](TRAINABILITY.md) distinguishes barren plateaus
from structural dead parameters, and specifies qubit/depth/initialization
controls and paired local/global initial-gradient probes. It records literature
assumptions and what remains unverified for our NLP objective.
