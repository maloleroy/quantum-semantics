# Research sources and scope

Sources checked on 2026-09-16. These references support the background; the algebra
and numerical findings in THEORY.md and REPORT.md are this study's analysis.

1. Varmantchaonala et al. **QCSE**, arXiv:2509.05729v2 (10 March 2026).
   [Paper](https://arxiv.org/abs/2509.05729v2).
   Local supplied PDF inspected, especially equations 10-28. It defines the
   context-matrix/rotation pipeline and marginal-bit readout. The repository is
   a causal adaptation, not a literal reproduction of its corpus or reported scores.
2. Mülken and Blumen. **Continuous-Time Quantum Walks: Models for Coherent Transport
   on Complex Networks**, Physics Reports 502 (2011).
   [Author preprint](https://arxiv.org/abs/1101.2572).
   Background on graph-generated unitary dynamics and coherent transport; it does
   not establish language-model gains for the present encoding.
3. Mernyei, Meichanetzidis and Ceylan. **Equivariant Quantum Graph Circuits**, ICML 2022.
   [Proceedings](https://proceedings.mlr.press/v162/mernyei22a.html).
   Relational symmetry and representation power motivate designing equivariance
   into the architecture. Their graph-function results are not a guarantee of
   prediction performance or absence of trainability problems in this model.
4. Liu et al. **GraphQWalk: Learning Structural Node Embeddings via Continuous Quantum
   Walk**, IEEE TNSE 13, 4779-4796 (2026).
   [Authors' implementation and citation](https://github.com/jitwooo/GraphQWalk).
   Uses average transition probabilities, spectral filtering and neighborhood
   aggregation to represent structural roles. This differs from claiming that
   evolved pure states themselves become closer in fidelity. The supplied plan's
   DOI was not independently verified against the publisher; the author repository
   verifies title, authors, venue, year, volume and pages.
5. Schuld, Sweke and Meyer. **The effect of data encoding on the expressive power of
   variational quantum machine learning models**, PRA 103, 032430 (2021).
   [Author preprint](https://arxiv.org/abs/2008.08605).
   Input encoding determines accessible functions; extra frequencies or depth
   alone do not establish task-relevant linguistic structure.
6. Pérez-Salinas et al. **Data re-uploading for a universal quantum classifier**,
   Quantum 4, 226 (2020). [Author preprint](https://arxiv.org/abs/1907.02085).
   Supports the general use of repeated input-dependent operations. The present
   alternating graph/context construction has different generators and task.
7. Levy and Goldberg. **Neural Word Embedding as Implicit Matrix Factorization**,
   NeurIPS 2014. [Paper](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html).
   Distributional-count/PMI structure motivates PPMI controls. PPMI is not inherently
   quantum; its computation and storage belong in the resource accounting.
8. Plesch and Brukner. **Quantum-state preparation with universal gate decompositions**,
   PRA 83, 032302 (2011). [Author preprint](https://arxiv.org/abs/1003.5760).
   State preparation is a substantive circuit resource even when amplitude
   encoding needs only logarithmically many qubits.
9. Berry, Childs and Kothari. **Hamiltonian simulation with nearly optimal dependence
   on all parameters**, FOCS 2015. [Author preprint](https://arxiv.org/abs/1501.01715).
   Sparse-Hamiltonian algorithmic guarantees assume the specified access model;
   they do not make an arbitrary dense vocabulary graph inexpensive to load.
10. Lu et al. **Chiral Quantum Walks** (2014).
    [Author preprint](https://arxiv.org/abs/1405.6209).
    Complex edge phases offer another way to study transport directionality.
    This study tests a source/target dilation instead; chiral language encoders
    remain untested and should not be presented as validated improvements.
11. Hill, Reichart and Korhonen. **SimLex-999: Evaluating Semantic Models with
    (Genuine) Similarity Estimation**, Computational Linguistics (2015).
    [Authors' dataset page](https://fh295.github.io/simlex.html),
    [paper](https://arxiv.org/abs/1408.3456).
    An external similarity probe, distinct from association or next-word prediction.
    Only pairs whose two words are in the training vocabulary are evaluated; this
    restricted subset must not be compared with published full-dataset scores.
