# Hamiltonian-Based Quantum Encoder for Context-Sensitive Word Embeddings

## 1. Motivation

The current QCSE-style encoding maps a context matrix to successive blocks of rotation angles. This makes the quantum state depend on **how coefficients are flattened, indexed, and injected into circuit layers**, even when those choices have no linguistic meaning.

The proposed alternative is to move the semantic structure from the **data-loading order** into a **Hamiltonian derived from the corpus graph**.

The core idea is:

> Represent words as orthogonal quantum basis states, represent corpus-level semantic relations as a weighted graph, and let a quantum evolution over that graph transform a context state into a contextual word representation.

This removes the need for a learned classical `one-hot → linear embedding` layer and gives the quantum component an essential role.

---

## 2. Main Construction

### Vocabulary graph

Let the vocabulary contain \(V\) words. Build a weighted undirected graph

\[
G=(\mathcal V,\mathcal E,W)
\]

where each node corresponds to a word and \(W_{ij}\) measures the relation between words \(i\) and \(j\).

Reasonable first choices are:

- windowed co-occurrence counts;
- PPMI;
- distance-weighted co-occurrence.

The important point is that \(W\) comes directly from corpus statistics, not from a learned classical embedding.

### Word states

Represent each word by an orthogonal state

\[
w_i \longleftrightarrow |i\rangle.
\]

With binary basis encoding this requires

\[
m=\lceil \log_2 V\rceil
\]

qubits.

The index itself carries no semantics. Semantic structure comes from the Hamiltonian.

### Graph Hamiltonian

A simple choice is

\[
H_G = D^{-1/2}WD^{-1/2}
\]

or a graph Laplacian such as

\[
H_G = I-D^{-1/2}WD^{-1/2}.
\]

Quantum evolution is then

\[
U_G(t)=e^{-itH_G}.
\]

This is closely related to a **continuous-time quantum walk**: amplitudes propagate between words according to corpus-level semantic connectivity.

---

## 3. Context Encoding

For a context window \(C\), prepare

\[
|\psi_C\rangle
=
\frac{1}{\sqrt Z}
\sum_{r\in C}
\sqrt{\alpha_r}\,|w_r\rangle,
\]

where \(\alpha_r\) can encode distance from the center word.

This immediately removes the arbitrary QCSE-style ordering of \(2m\)-coefficient blocks.

If word order must be preserved, introduce a separate position register,

\[
|\psi_C\rangle
=
\frac{1}{\sqrt Z}
\sum_r
\sqrt{\alpha_r}\,
|r\rangle_{\text{pos}}
|w_r\rangle_{\text{word}},
\]

rather than encoding position indirectly through the order in which rotation gates are applied.

A context-dependent diagonal Hamiltonian can also be introduced:

\[
H_C=\sum_i c_i |i\rangle\langle i|,
\]

where \(c_i\) reflects the presence or weight of word \(i\) in the current context.

A compact variational encoder is then

\[
U_\theta(C)
=
\prod_{\ell=1}^{L}
e^{-i\beta_\ell H_G}
e^{-i\gamma_\ell H_C}.
\]

This is analogous in spirit to a QAOA-like alternating evolution: \(H_G\) supplies global corpus structure while \(H_C\) injects the current context.

---

## 4. Why This Reduces Indexing Arbitrariness

Suppose the vocabulary is relabeled by a permutation matrix \(P\).

Then

\[
|\psi_C\rangle\rightarrow P|\psi_C\rangle,
\qquad
H_G\rightarrow PH_GP^\dagger,
\qquad
H_C\rightarrow PH_CP^\dagger.
\]

Therefore

\[
U_\theta\rightarrow PU_\theta P^\dagger.
\]

The model changes only by a change of basis. Its behavior does not depend on whether, for example, *cat* was assigned index 7 or index 932.

This is the desired property: **permutation equivariance rather than index-dependent geometry**.

---

## 5. Training Objective

For a skip-gram/CBOW-like task, exclude the center word from the input context and predict it from the evolved state.

\[
|\phi_C\rangle = U_\theta(C)|\psi_C\rangle.
\]

The probability of predicting word \(j\) is

\[
p(j|C)=|\langle j|\phi_C\rangle|^2.
\]

Train with

\[
\mathcal L=-\log p(w_{\text{target}}|C).
\]

Initially, keep the only learned parameters as

\[
\theta=\{\beta_\ell,\gamma_\ell\}.
\]

This is an important control: it prevents a large classical embedding layer from solving the NLP task before the quantum circuit is used.

---

## 6. Recommended Attack Plan

### Phase A — Validate the idea mathematically

1. Reproduce the existing QCSE baseline.
2. Build a small vocabulary graph from the same corpus.
3. Test several graph weights: co-occurrence, PPMI, normalized PPMI.
4. Verify numerically that random vocabulary permutations leave predictions unchanged after applying the same permutation to inputs and outputs.

This permutation test should become a unit test.

### Phase B — Classical simulation of the quantum model

Before compiling a circuit, simulate directly in the \(V\)-dimensional Hilbert space:

\[
|\psi\rangle\mapsto e^{-itH}|\psi\rangle.
\]

Use sparse matrices and Krylov/exponential-multiply methods.

Compare:

- QCSE;
- one-hot + classical linear embedding;
- graph Hamiltonian with fixed \(H_G\);
- graph Hamiltonian + variational \(\beta,\gamma\).

The crucial ablation is **removing \(H_G\)**. Performance should degrade if the quantum/graph evolution is genuinely carrying semantic information.

### Phase C — Quantum implementation

There are two useful implementations.

**1. Unary / single-excitation encoding — best scientific prototype**

Use one qubit per vocabulary item:

\[
|w_i\rangle = |00\ldots010\ldots0\rangle.
\]

Then graph evolution can be implemented naturally with an XY hopping Hamiltonian

\[
H_G
=
\frac12
\sum_{i,j}
W_{ij}
(X_iX_j+Y_iY_j).
\]

The dynamics remain inside the one-excitation subspace.

Advantages: very clean symmetry and direct graph interpretation.

Disadvantage: \(V\) physical qubits.

**2. Binary encoding — scalable target**

Use \(\lceil\log_2V\rceil\) qubits and map \(H_G\) to Pauli operators.

This uses far fewer qubits, but Hamiltonian compilation can become expensive because a generic graph Hamiltonian may require many Pauli terms.

Therefore: **prototype the scientific idea with unary encoding first; optimize binary Hamiltonian simulation only if the model works.**

---

## 7. Minimal Experimental Success Criteria

The Hamiltonian approach is interesting if it satisfies all three:

1. **Permutation robustness:** arbitrary vocabulary relabeling does not change performance.
2. **Quantum-component necessity:** removing the Hamiltonian evolution causes a measurable degradation.
3. **Context sensitivity:** the same word obtains measurably different output states in different contexts.

Useful diagnostics include fidelity,

\[
F(C_1,C_2)
=
|\langle\phi_{C_1}|\phi_{C_2}\rangle|^2,
\]

prediction accuracy / perplexity, and downstream similarity tests.

---

## 8. Main Risk

The conceptual model is simple; **Hamiltonian implementation is the hard part**.

A dense corpus graph may produce an expensive quantum simulation. The first experiments should therefore use:

- a small vocabulary;
- sparse \(k\)-nearest-neighbor co-occurrence graphs;
- shallow alternating evolutions;
- unary encoding as the clean reference implementation.

Only after demonstrating a useful effect should effort be spent on efficient binary Hamiltonian simulation.

---

## References / Starting Points

- Varmantchaonala et al., **QCSE: A Pretrained Quantum Context-Sensitive Word Embedding for Natural Language Processing** (2025): https://arxiv.org/abs/2509.05729
- Continuous-time quantum walks provide the basic interpretation \(U(t)=e^{-itH_G}\) for graph-based quantum propagation.
- Liu et al., **GraphQWalk: Learning Structural Node Embeddings via Continuous Quantum Walk**, *IEEE Transactions on Network Science and Engineering* (2026), DOI: 10.1109/TNSE.2025.3644803.
- Recent work on permutation-equivariant quantum graph neural networks supports the broader principle that graph relabeling symmetry should be built into the quantum architecture rather than learned from arbitrary node indices.

### Bottom line

The cleanest first model is:

\[
\boxed{
\text{context}
\rightarrow
|\psi_C\rangle
\rightarrow
\prod_\ell
e^{-i\beta_\ell H_G}
e^{-i\gamma_\ell H_C}
\rightarrow
\text{measurement over }|w_i\rangle
}
\]

with **no learned classical embedding layer**.

This directly tests whether corpus-derived Hamiltonian quantum dynamics can act as the semantic encoder.
