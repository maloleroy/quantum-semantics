# Poster presentation: one question, one design, two tests

Revision 7, 17 November 2026. Read across the top row, then across the shaded
results row: starting point (1), design (2), learning (3), control and answer (4).

## Three-minute script

**0:00–0:45 — 1. The starting problem**

Can ten qubits finish your sentence? We began with quantum word embeddings:
representing context as a quantum state and predicting the missing word's
binary code. The first row of the diagram shows Quantum Context-Sensitive
Embedding, or QCSE. The other rows distinguish predicting from both sides,
as in Bidirectional Encoder Representations from Transformers, or BERT, from
predicting the next token from past text, as in a Generative Pre-trained
Transformer, or GPT.

In our full QCSE study, exact-word test recovery reached only 1.07 percent.
Our next question was whether learning the quantum evolution directly could
give a useful predictor.

**0:45–1:35 — 2. Our design**

We made the Hamiltonian trainable. We also changed to a subword task, so this
is a new experiment, not a controlled accuracy improvement over QCSE.

The real tokenizer example splits “language” into four pieces. A vocabulary
of 512 token identifiers needs nine token qubits. We add one auxiliary qubit.
Each Hamiltonian learns a rank-128 subspace and its energies, with energy shifts
from the preceding context. The model-specific quantities are defined below the equation.

We prepare a state from training bigram counts, apply three learned evolutions
separated by phase rotations, and measure the token register. There is no
classical vocabulary output layer. By the Born rule, we square the output
amplitudes and sum over the two auxiliary outcomes to obtain each token's
probability. We predict the token with the highest probability and count an
exact match only when it equals the actual next subword in the text.

**1:35–2:15 — 3. Does it learn?**

The shaded row tests that design. The curve shows actual training epochs,
averaged over three seeds. Validation prediction loss decreases from 3.06 to
2.47 nats. We select checkpoints on fixed validation targets, and the most
probable subword is correct 44.10 percent of the time.

The band shows one standard deviation across training seeds. The dashed
curve is online training loss.

**2:15–3:00 — 4. What does that establish?**

We gave a classical multilayer perceptron the same data and update exposure.
Its most-likely-token validation accuracy is 50.35 percent, compared with
44.10 percent for the Hamiltonian. Both average three seeds, with checkpoints
selected by minimum validation loss. Capacities and optimization schedules differ.

So ten qubits do learn a token predictor, but sampled text remains incoherent,
as the saved excerpt shows. Our contribution is trainable Hamiltonian prediction
with direct measurement readout. A controlled comparison with word embeddings
remains to be done. These are noiseless simulations on reused validation;
the WikiText test remains unscored.

## Thirty-second version

We began with a quantum embedding model whose exact-word recovery was weak.
We then made the Hamiltonian itself trainable and predicted subword tokens
directly by measurement. The new model's validation loss decreases with training,
but the task differs from the original word task, so this is not a controlled
win over QCSE. A classical network has higher validation accuracy, and sampled text remains
incoherent. We establish a trainable quantum predictor and identify what still
needs to improve.

## Questions to prepare

**Does 44.10% versus 1.07% prove the Hamiltonian is better?**
No. The vocabulary, context, training recipe, readout and evaluation split differ.
The poster separates the earlier word-level problem from the new subword experiment.
Do not quote a ratio or a percentage-point improvement between those scores.

**What is the actual contribution?**
Learning a context-dependent Hamiltonian and obtaining token probabilities by
direct Born readout. The curve establishes learning in this recipe; it does not
establish superiority over all embedding or classical approaches.

**Why did you change to subwords?**
Byte-pair pieces represent text without requiring every word to be a vocabulary
entry, while retaining case, punctuation and spaces. The tokenizer and statistics
use training articles only. This task change is also why the QCSE scores are
not a direct benchmark against the final model.

**Do embeddings, bidirectional prediction and causal prediction exclude each other?**
No. All three models use representations. The first diagram explains a quantum
representation/readout, while the other two explain which context is visible.
Our Hamiltonian uses causal next-token prediction.

**Why is the classical comparison more directly informative?**
It uses the same WikiText subword task, inputs and update exposure. The models
still differ in capacity and optimization schedule. Both reported accuracies average
three seeds on the same 4,096 validation targets.

**How are the prediction scores defined?**
For context C, the Born probability of token t is
p(t | C) = sum over a in {0, 1} of |⟨t, a | ψ(C)⟩|².
The auxiliary qubit is ignored by summing its outcomes. Exact next-token accuracy
is the fraction of targets for which argmax_t p(t | C) equals the true next
subword identifier. Each context contains only preceding tokens; a partial word
or a semantically similar token does not count as an exact match.
Negative log-likelihood averages −ln p(y | C) for the true next token y;
it is reported in nats, and lower is better. Checkpoint selection uses this loss.

**Does 44.10% describe one measurement?**
No. It is accuracy of the most probable token, using the full simulated distribution.
Expected success of a single Born draw is 28.52%, a different quantity.

**What does the loss band show?**
One sample standard deviation across training seeds, not an article-level confidence
interval. Online training loss averages changing weights and sampled rows.

**What is the source of the text sample?**
The training-seed-22 model, prompt “The history of ”. The poster quotes the
uncorrected prefix “its first game ream , butvarBan7” of a 64-token continuation.

**What remains open?**
A matched embedding-versus-Hamiltonian benchmark, coherent generation, and
efficient state preparation and native-gate compilation. No hardware speedup
or entanglement-specific advantage is established. The held-out WikiText test
has not been scored.
