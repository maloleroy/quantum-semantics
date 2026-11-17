# Poster presentation: embeddings to Hamiltonian dynamics

Revision 3, 17 November 2026. The title is “Can ten qubits finish your sentence?”
Read down the left column for the model, then the right column for the evidence.

## Three-minute script

**0:00–0:35 — Question and contribution**

Can ten qubits finish your sentence? We started from Quantum Context-Sensitive
Embedding, or QCSE, which trains an embedding circuit to predict a word's binary
code. Our contribution changes what is learned: we train the Hamiltonian that
generates quantum evolution, and read token probabilities directly by measurement.

The tokenizer example is real. “Language” becomes four subword pieces. A
512-token vocabulary fits into nine token qubits, and we add one auxiliary qubit.

**0:35–1:30 — Hamiltonian**

The equation defines each stage's Hamiltonian. C is the preceding context, and
the stage index runs from one to three. Q is a learned matrix with 128
orthonormal columns in a 1024-dimensional space. The vector e contains learned
energies. Classical context pooling produces the energy shifts s. Theta
collects the trainable parameters. Diag places the energies on a matrix
diagonal, and the dagger means complex conjugate transpose.

We prepare a state from training bigram counts, apply three learned evolutions
separated by local phase rotations, and measure the token register. We ignore
the auxiliary outcome. The model has about 1.07 million trainable real parameters
and no classical vocabulary output head.

**1:30–2:05 — Training**

The plot shows actual training epochs for three seeds. We minimize negative
log-likelihood and select checkpoints on fixed validation targets. Validation
loss falls from 3.06 to 2.47 nats. The dashed curve is the online training loss,
and the band is one standard deviation across training seeds.

**2:05–3:00 — Evidence and scope**

The two case studies explain the move beyond our original embedding experiments.
The full QCSE study reached 1.07 percent exact-word test accuracy with a
10,864-word vocabulary and fourteen qubits. Our ten-qubit Hamiltonian reaches
44.10 percent most-likely-token accuracy on a 512-subword validation task.

Those percentages are not a controlled head-to-head comparison. The tasks,
vocabularies, contexts and evaluation splits differ. A matched comparison is
still needed. On the Hamiltonian's task, a classical multilayer perceptron
retains lower loss, 2.22 nats.

The result we establish is trainable Hamiltonian prediction with direct
measurement readout. We have not established quantum advantage or efficient
hardware costs. Generation remains incoherent, and the WikiText held-out test
remains unscored.

## Thirty-second version

We move from Quantum Context-Sensitive Embedding, or QCSE, to learning the
Hamiltonian that generates quantum evolution. Our model prepares a count-based
state, applies three learned evolutions on ten qubits, and obtains token
probabilities by measurement. The genuine training curve shows its loss
decreasing. Existing QCSE and Hamiltonian results motivate this direction, but
their different tasks do not establish a controlled ranking. A classical
network still leads on the Hamiltonian's task.

## Questions to prepare

**Does 44.10% versus 1.07% prove the Hamiltonian is better?**
No. One is most-likely-subword validation accuracy and the other exact-word
test accuracy. Vocabulary sizes, context, training and readout also differ.
They are descriptive evidence from the development process, not a matched
benchmark. Do not quote a ratio or percentage-point improvement between them.

**What would support a direct comparison?**
Train both model families on the same training-only tokenizer and statistics,
same grouped data split, causal context, target/readout metric and declared
update budget. Select checkpoints on validation, then lock both before test.
That comparison has not been performed for this poster.

**Does 44.10% describe one quantum measurement?**
No. It evaluates whether the most probable token is correct, using the full
simulated distribution. Expected one-draw success is a different quantity,
28.52% for this model, and is not the main displayed comparison.

**Why use a low-rank Hamiltonian?**
Only 128 learned orthonormal directions carry adjustable energies in a
1024-dimensional state space. This makes exact numerical evolution more
manageable than exponentiating an unconstrained full dense matrix.

**What do the equation's terms mean?**
C is context, l is stage, H is its Hermitian generator, Q the orthonormal
subspace matrix, e the learned energies, s the context shifts, theta the
trainable parameters, diag the diagonal-matrix constructor, and dagger the
complex conjugate transpose. The full key appears directly below the equation.

**Why does online training loss sometimes exceed validation loss?**
It averages sampled positions while weights change during the epoch.
Validation evaluates a fixed checkpoint on another fixed sample. Their
difference is not a clean overfitting diagnostic.

**Is the loss band a confidence interval?**
No. It is one sample standard deviation across three training seeds.

**Is this efficient on real quantum hardware?**
That remains open. The circuit illustrates register-level simulated evolution.
State preparation, native-gate compilation and measurement costs need assessment.

**Was the held-out WikiText test evaluated?**
No. Validation was reused for architecture decisions, and the test remains
unscored. The QCSE case study does report a separate test result.
