# Poster presentation notes

For the three-minute English presentation requested in the email. Rehearse
with a timer, pointing to A, B and C in that order. The text below is about
350 words; allow time to point and pause.

## Three-minute script

**0:00–0:30 — Question and motivation**

Can ten qubits finish your sentence? Our subject is quantum semantics:
representing meanings and their relations with quantum states and operators.
The square at the top distinguishes opposition, negation and inclusion.
We asked a narrower, measurable question: does a trainable quantum component
improve language prediction when classical controls receive the same information
and training exposure?

**0:30–1:10 — A: the metric**

We first reimplemented a published model, QCSE. Each output qubit predicts one
bit of a word identifier. Its published criterion accepts a prediction when at
least half the bits match. In our miniature audit of 57 test targets, QCSE
scored 56.1 percent. A constant prediction based only on training statistics
scored 96.5 percent. Under exact-word scoring, QCSE recovered no words, while
the constant prior reached 22.8 percent. The metric was rewarding label
statistics rather than useful word recovery.

**1:10–1:45 — B: the circuit control**

We then gave eight-qubit circuits learned inputs and a classical output layer.
The identity control keeps that classical machinery and removes the circuit.
The trained circuit and identity have nearly the same top-one accuracy, for
both left-only and bidirectional context. A small classical multilayer
perceptron leads in both cases. These comparisons use the same targets and
update budgets, although capacities and optimization schedules can differ.

**1:45–2:30 — C: direct Hamiltonian prediction**

Finally, we removed the classical vocabulary head. A learned Hamiltonian evolves
a prepared state, and the Born rule gives token probabilities. The final model
uses ten qubits. The graph shows changes to the recipe, not training epochs.
These models learn, and retaining input coherence helps against a separately
trained dephased control on an earlier word-level task.

But on WikiText, the final classical MLP has validation loss lower by 0.24 nats.
The quantum model's sampled continuations remain incoherent, as the example
shows.

**2:30–3:00 — Scope and conclusion**

All experiments use noiseless simulation. Most comparisons have only two or
three seeds, and the WikiText test split remains unscored after repeated
validation use. We demonstrate trainability and a restricted coherence benefit,
not a quantum advantage. The next scientific step is to test semantic relations
directly. The practical lesson is to check the metric and the classical control
before attributing a gain to quantum dynamics.

## Thirty-second version

We asked whether trainable quantum models improve language prediction over
classical controls given the same information. Three comparisons found no
quantum advantage: a constant guess beat QCSE's published metric, removing a
hybrid circuit changed little, and a classical network led the final
Hamiltonian comparison. Quantum learning and input interference do have
measurable effects in simulation, but generated text remains incoherent.
The lesson is to check the metric and the classical control.

## Questions to prepare

**Does this disprove quantum language modelling?**
No. The conclusion concerns the tested architectures and budgets in simulation.
It neither proves impossibility nor establishes a hardware speed comparison.

**What is the difference between top-1 and one-draw success?**
Top-1 asks whether the highest-probability token is correct. One-draw success
averages the probability assigned to the correct token, corresponding to a
single sample. NLL scores the full predictive distribution with a logarithm.

**Why can a constant prediction score 96.5%?**
Frequency-ranked word identifiers have unbalanced bits. The half-bit threshold
can reward predicting common bit values without identifying the word.
The result comes from the miniature audit, not the entire full-corpus sweep.

**Is 50% the random chance line?**
Not for the ten-bit audit. With independent uniform predicted bits, ties count,
so the half-bit chance level is 62.3046875%. For an odd number of bits it is 50%.

**How fair are the controls?**
They share the task, inputs, targets and update exposure. Parameter counts and
learning-rate schedules can differ, as the poster states. Identity is the
closest ablation of the hybrid circuit. The MLP is a strong classical reference.

**Did you evaluate an independent WikiText test?**
No. The plotted WikiText results use a reused validation set. The held-out
8,192-target WikiText test remains unscored. Other word-level studies do report
held-out tests, so avoid saying that every study lacks test evaluation.

**Does the coherence result prove an entanglement advantage?**
No. The control removes input coherence and trains separately. It establishes
a restricted interference benefit within those architectures. It does not
isolate entanglement or outperform the classical MLP.

**Is there a quantum speedup?**
None is demonstrated. We apply learned Hamiltonian evolutions numerically.
Efficient state preparation, compilation to hardware gates and measurement
costs remain unresolved.

**How does this connect to the semiotic square?**
The square motivates explicit semantic operators. Next-token prediction does
not directly test opposition or inclusion. A future relation-classification
task should compare the same semantic constraints in quantum and classical
models.
