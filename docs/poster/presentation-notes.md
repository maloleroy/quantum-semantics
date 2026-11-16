# Hamiltonian poster presentation

Revised for the two-column poster, 17 November 2026. Read down the left column
for the model, then down the right column for the evidence.

## Three-minute English script

**0:00–0:30 — Purpose**

Can a learned Hamiltonian predict the next token of a text? Our contribution is
a language predictor that evolves a quantum state and reads token probabilities
directly by measurement. We study this in noiseless simulation. The left column
explains what we built, and the right column shows what it learns.

**0:30–0:55 — Tokenization**

We use WikiText articles and a byte-pair tokenizer fitted only on training
articles. A word can occupy several tokens: the real example here splits
“language” into four pieces. The vocabulary has 512 identifiers, so nine qubits
can represent the token register. We add one auxiliary qubit and condition the
model on up to 128 preceding tokens.

**0:55–1:40 — Hamiltonian and circuit**

The central equation defines each Hamiltonian. Q is a learned orthonormal basis
with 128 columns. The energies are learned, and a classical pooling of the
context shifts those energies. Context also sets local phase rotations.

We prepare a state from training bigram counts, apply three learned Hamiltonian
evolutions separated by phase rotations, and measure the token register. There
is no classical vocabulary output layer. The auxiliary measurement outcome is
ignored. The full model has about one million trainable real parameters.

The coherence bars come from an earlier word-level study. Keeping input
interference improves one-draw success from 5.71 to 6.76 percent compared with
a separately trained dephased control. This is a restricted coherence result,
not evidence for an entanglement advantage.

**1:40–2:15 — Learning**

The graph shows actual training epochs, averaged over three seeds. The solid
line evaluates fixed validation targets. The dashed line is the online training
loss. Validation NLL decreases from the count prior's 3.06 to 2.47. We choose
checkpoints by validation NLL and do not score the WikiText test.

**2:15–3:00 — Prediction and conclusion**

The benchmark shows that the classical multilayer perceptron still does better,
with NLL 2.22 under the same data and update exposure. Capacities and schedules
are not identical.

The Hamiltonian assigns the correct token an average probability of 28.52
percent, which is its expected success in one Born draw. But autoregressive
generation remains incoherent, as the uncorrected sample shows. We establish
trainability and a restricted coherence benefit. Better language generation and
an efficient hardware implementation remain open problems.

## Thirty-second version

We built a language predictor in which a learned Hamiltonian evolves a
count-prepared quantum state and measurement gives token probabilities. The
final model has nine token qubits, one auxiliary qubit and three evolution
stages. Its loss decreases with training, and input coherence helps against
a separately trained dephased control in an earlier study. But a classical
network still leads, and generated text remains incoherent. Our result is
trainable Hamiltonian prediction in simulation, not a quantum advantage.

## Questions

**Why is the Hamiltonian low rank?**
Only 128 learned orthonormal directions carry adjustable energies in a
1024-dimensional state space. This reduces the numerical evolution cost
compared with learning and exponentiating a full dense matrix.

**What does the auxiliary qubit do?**
Evolution acts jointly on the token and auxiliary registers. Measurement
marginalizes over its two outcomes, adding flexibility to the token
distribution. We do not claim that it proves an entanglement advantage.

**Why is training loss initially higher than validation loss?**
Training is an online average over sampled positions while the model changes.
Validation evaluates the current checkpoint on a different fixed set. Their
difference is not a clean overfitting diagnostic.

**What does the shaded band show?**
Plus or minus one sample standard deviation across training seeds. It is not
an article-level uncertainty interval or a statistical significance test.

**Is the tokenizer trained on validation or test articles?**
No. Byte-pair merges and count statistics use training articles only. Byte
coverage preserves any source word, including punctuation and whitespace.
The four token IDs shown are from the actual archived tokenizer.

**Is one-draw success the same as argmax accuracy?**
No. One-draw success averages the target probability. Argmax asks whether
the most likely token is correct and requires the full distribution. The
poster reports one-draw success because it corresponds to measurement.

**Does this demonstrate quantum advantage?**
No. The classical MLP has lower validation loss under the same input and
update exposure. Parameter counts and optimization schedules can differ.
The coherence result only compares two restricted architectures.

**Have you demonstrated efficient hardware circuits?**
No. The circuit drawing is a register-level description of simulated matrix
evolution. State preparation, native-gate compilation and measurement costs
must be established before a hardware efficiency claim.

**Was WikiText test performance measured?**
No. Validation was reused for architecture decisions. Its 8,192-target held-out
test remains unscored. The separate word-level coherence study does report
held-out test results.
