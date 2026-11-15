# Quantum text prediction: what we have learned

As of 15 November 2026. Detailed inventories: [successful steps](experiments/successful.md) ·
[negative and inconclusive experiments](experiments/unsuccessful.md).

## The data and the task

We began with short Tatoeba sentences, then moved to **WikiText-2 Raw**, connected Wikipedia
articles. The current task preserves case, numbers, punctuation and whitespace. A tokenizer learned
only from training articles breaks text into **512 possible tokens**: individual UTF-8 bytes and
common byte combinations. Consequently, unfamiliar words remain representable; they do not become an
“unknown word.” Earlier word-based experiments discarded these details and had many unknown targets,
so their scores cannot be compared directly with today's token scores.

For each example, the model receives up to **128 preceding tokens** and predicts the next one.
Context stops at article boundaries and never contains the answer or future text. Training,
validation and test articles stay separate. Tokenization and frequency tables use training articles
only. The established training recipe draws 524,288 examples per logical epoch from a pool of
2,097,152 distinct positions, for 30 epochs: 30,720 batch updates. This is a sampled part of the
approximately 6.86 million available training transitions.

We train by **negative log-likelihood (NLL)**: lower is better, because the model assigns more
probability to the observed continuation. We select checkpoints on 4,096 fixed validation targets.
Separately, **sampling success** is the average probability that one draw produces the recorded
answer; **argmax accuracy** asks whether the most probable token is correct. These are different
metrics. Space/non-space scores expose models that mainly learn to emit spaces. Generation
repeatedly draws a token and feeds it back into the context, preserving malformed outputs rather
than filtering them.

The WikiText test subset of 8,192 targets remains **unscored**. Validation has guided many
experiments; repeating training seeds checks robustness, but provides no independent test of
generalization.

## The model, from its starting point to today's version

The basic Hamiltonian model loaded preceding words into a quantum superposition, applied a learned
**Hamiltonian**—the rule controlling how amplitudes mix and interfere—and measured the register to
obtain a word. Measurement produces probabilities through the **Born rule**, without a classical
output softmax. Training the Hamiltonian helped, and preserving coherence beat a separately trained
dephased control on Tatoeba.

Today's strongest replicated recipe, **`prior_content`, three stages/rank 128**, has about **1.07
million trainable parameters**. Nine qubits label the 512 tokens; one auxiliary qubit supplies extra
room for the evolution. It starts from a training-count estimate of what usually follows the last
token. Learned position weights and a small classical content selector summarize the recent 16
tokens and older context. These summaries control phase rotations and the Hamiltonians' energies
between successive evolution stages. Low-rank evolution makes simulation affordable. There is **no
learned classical vocabulary decoder**. This is nevertheless a hybrid: count-based preparation and
classical context processing contribute real information. Count preparation replaced the earlier
learned product-state preparation; the development history is not a stack of every successful
component.

| Current evidence | NLL ↓ | One-draw success ↑ | Argmax ↑ |
| --- | ---: | ---: | ---: |
| Replicated quantum recipe, three-seed mean | 2.4655 | 28.52% | 44.10% |
| Equally exposed strong classical MLP, three-seed mean | 2.2228 | 39.45% | 50.35% |
| Lowest observed quantum NLL: four-stage candidate, one seed | 2.4091 | 30.02% | 46.09% |
| Its matched classical MLP, one seed | 2.1662 | 42.40% | 51.81% |

The replicated model gains a small NLL benefit from 128 versus 16 tokens, although sampling and
argmax slightly worsen. Longer training and more capacity give promising single-seed improvements; a
fourth stage has not beaten comparable extra width on the declared NLL criterion. **Generated text
remains incoherent.** We have useful simulator improvements and evidence of learned quantum
dynamics, but no demonstrated advantage over the classical controls, entanglement-specific benefit,
or practical quantum speedup. [Replicated results](experiments/coverage-seeds.md) · [later capacity
candidates](experiments/depth-four.md).
