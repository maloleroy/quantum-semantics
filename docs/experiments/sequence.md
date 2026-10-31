# Sequential local preparation: declared manual comparison

31 October 2026. Implementation and scientific checks complete; training is
**unrun**. The [expanded product experiment](product-exposure.md#completed-results)
showed that more training data helps, but its matched MLP improves more and its
longer context contributes little. This screen changes context encoding while
retaining learned Hamiltonians and direct token measurement.

## Hypothesis and model

The product recipe compresses context into pooled local angles. Its decays retain
some positional information: it is not a bag of tokens or an order-invariant
encoder. The hypothesis here is that applying noncommuting local gates in token
order gives a more useful representation of byte-token sequences than summing
angles before preparation. It remains a fixed-size representation and can still
lose information; this is not a demonstrated solution for long context.

For ten qubits (nine token qubits and one auxiliary), start in all-zero state.
Apply a learned offset RY/RZ pair on every qubit. Then, from oldest token to
newest, apply an RY followed by an RZ on every qubit. For token ID j at age a,
these gate angles are `phase_scale * .97**a * table[axis, j, qubit]`. There are
two shared token tables, initialized Gaussian with standard deviation .1. The
angle weights are not normalized by context length; padding contributes identity.
All 128 context positions are eligible, including distant tokens. The 16-token
control keeps exactly the same recent-position weights.

The resulting state is still a product of ten single-qubit states. Apply two
learned rank-64 Hamiltonian evolutions with the existing learned local context-RZ
stage between them, then measure the nine token qubits and ignore the auxiliary.
Interleaving angles retain the last/recent/older three-stream pools and learned
offsets. There is no classical vocabulary decoder on this path. Local preparation
itself cannot entangle qubits; any later entanglement comes from the Hamiltonians.
This experiment does not isolate an entanglement-specific benefit.

The simulator evaluates the preparation exactly as ten local two-amplitude
factors, then forms their tensor product once. This avoids applying every local
gate to an exponentially large statevector. These particular preparation circuits
are efficiently classically simulable, and their processing cost must be counted.
The direct hardware sequence has at most 2,560 token-controlled local gates for
128 tokens and ten qubits, plus 20 preparation-offset gates, the interleaving
rotations and Hamiltonian implementation. Folding the local sequence into fewer
state-preparation gates moves work to classical processing. Neither route supplies
an end-to-end hardware speedup claim; Hamiltonian compilation, noise and shot-based
training remain unresolved.

`--amplitude-decay` is unused for sequence models, as it was for product models.
The .97 gate-angle decay is fixed in the encoding function, not controlled by
that option. `--phase-scale` scales token-dependent preparation/interleaving;
at zero it leaves context-independent learned offsets. Models require automatic
encoding, Born readout and at least two Hamiltonian layers.

## Retrained controls and budgets

The Hamiltonians retain 262,272 real parameters. Preparation tables add 10,240,
the interleaving tables add 15,360 and offsets add 30: **287,902** total, about
91.1% in the dynamics. This is 5,110 fewer parameters than the prior product
model. Parameter placement is not evidence of quantum usefulness.

Six independently trained fits:

| Label | Variant | Question |
|---|---|---|
| learned | `sequence`, 128 tokens | Does sequential preparation improve prediction? |
| short | `sequence`, 16 retained tokens | Does older context contribute? |
| pooled | `sequence_pooled`, 128 tokens | Does composing gates help beyond summing the same weighted angles? |
| fixed | `sequence_fixed`, 128 tokens | What happens when all controller tables/offsets remain fixed? |
| frozen | `sequence --freeze-core`, 128 tokens | Does training the Hamiltonians contribute? |
| mlp | `sequence_mlp`, 128 tokens | Can a matched classical map use the same prepared-state information better? |

The pooled variant sums weighted token RY angles and RZ angles separately, then
applies one RY/RZ pair after the common offset preparation. It has identical
allocated/trainable parameters to `sequence`. It still retains positional decay
and two noncommuting axes; it tests sequential composition versus angle aggregation,
not removal of every kind of order sensitivity or quantum noncommutation.

The fixed-controller model trains 262,272 dynamics parameters while retaining
25,630 fixed stored controller parameters. The frozen-core model trains only
25,630 controller parameters; its unchanged core remains stored and executed.
These are different trainable budgets and answer specific attribution questions.

The MLP independently learns the same sequential preparation and later angle
family. It receives three Bloch components per prepared qubit plus ten interleaving
angles (40 features), followed by a 473-unit tanh hidden layer and categorical
output. The Bloch components fully describe each local pure state and omit
unobservable factor/global phase. They are calculated classically from the local
factors, not estimated by taking multiple measurements of a single preparation.
Total MLP parameters: **287,711**, including the same 25,630 controller parameters.
This matches physical input information and approximate parameter count, not
compute or hardware cost. Compact Bloch features are a different classical
parameterization from the earlier product MLP's real-amplitude inputs; both
classical results will remain visible.

## Fixed protocol and evidence

Reuse `output/product-exposure/batch-MDwDbm/data/full.npz` and `short.npz` exactly:
131,072 training targets, 4,096 validation targets, fixed training-only byte BPE,
all-training counts, official article partitions and paired target byte lengths.
No new data preparation or tokenizer fitting. Held-out test scoring remains absent.

Each new fit uses seed 11, 30 epochs, batch 512, Adam LR .003, clipping 5,
float64/complex128 and one Torch thread. Training and checkpoint selection use
NLL with the existing sampling tie-break, including epoch zero. No hyperparameter
sweep or continuation from old optimizer states is included. Every new fit has
7,680 optimizer updates and 31 candidate validation checkpoints.

The expanded product and product-MLP predictions are reused as frozen references
(NLL 2.9673 and 2.7966). Their weights reproduced saved arrays before source edits,
recorded in commit `b4527ba`. An archived source snapshot from that commit matches
every recorded source hash for both runs. Preflight checks the archived source,
original datasets and selected weight locks, while the manifest pins reference
predictions and other artifacts. New fits use the normal current-source loader.
Old checkpoints are not re-scored under changed source or silently unlocked.

The summary re-evaluates all six new locked models and checks prediction/metric
agreement at 1e-12. It reports ten metric rows (six new models, two frozen model
references and two all-training count references) and 27 paired article-bootstrap
intervals, using `learned` minus nine controls for NLL, sampling and argmax.
Sampling success, argmax and NLL remain separate. Target bits per byte retains its
sampled-target denominator; it is not a full-corpus text-likelihood benchmark.

The same three prompts/seeds produce 64 direct draws each from six new models and
two count models, saved as text CSV and exact token-ID NPZ (24 samples). Historical
model samples stay in their original directories. No retries, filtering or
sampling-temperature changes. Improved NLL alone does not establish better
sampling or meaningful generation.

This remains a single-seed adaptive validation screen. Bootstrap intervals omit
seed/selection uncertainty. A useful result must be judged against the matched
classical/count controls, long-context comparison and generation diagnostics;
passing numerical tests or beating the previous quantum recipe alone does not
achieve the research objective.

## Checks and manual command

All **58 tests**, Ruff and strict Pyright pass. New checks compare sequential and
pooled preparation with explicit local gates, probabilities/gradients with dense
Hamiltonian circuits, and initial-state purity. They verify order sensitivity
under equal weights, padding identity, zero-scale context independence,
fixed/frozen budgets, and Bloch-state reconstruction/global-phase invariance.
Generic checkpoint/probability checks cover all four new model variants.

Generated runner syntax, imports, strict Pyright/Ruff, archived-reference checks
and no-fit preflight pass. The full new training/summary pipeline is **unrun**.
No new research training was launched or monitored by the assistant.

```bash
bash output/sequence-screen/launch.sh
```

The launcher uses the local
[codex-long-jobs skill](/home/malo/.codex/skills/codex-long-jobs/SKILL.md), with
no viewer and event-only delivery to preserve manual handoff. It prints the job
name, log/state paths and log-follow command. Return to the conversation after
completion for analysis. Allow roughly **45–90 minutes** for six concurrent
one-thread fits and summaries. This is an unbenchmarked planning estimate based
on the previous 28-minute fits plus sequential preparation overhead, not a measured
runtime or a promise of completion within that range.

Input checksums, a runner lock and a live-worker guard protect the run. If it
fails, resume only missing fits from initialization using its printed batch path:

```bash
bash output/sequence-screen/launch.sh output/sequence-screen/batch-XXXXXX
```

Existing completed fits are verified and preserved. The final
`SEQUENCE_SCREEN_COMPLETE` marker is printed only after all fits and summaries
succeed. Cancellation, if requested, uses the controller for the whole process
group. Competitive text generation, useful long context and quantum advantage
remain unproven.
