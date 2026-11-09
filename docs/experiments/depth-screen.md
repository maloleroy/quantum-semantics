# Three-stage quantum refinement versus matched width

9 November 2026. Declared before fitting. This is exploratory validation reuse.

## Question and prediction path

Content-controlled quantum prediction improves across seeds, but remains behind
the strong MLP and its own refitted short-context control. Locked interventions
show sensitivity to older context and its order. They do not identify the cause
of the generalization gap. Test a separate hypothesis: an additional contextual
Hamiltonian stage and RZ interleaving can improve the conditional Born distribution
beyond simply increasing two-stage rank.

Use the existing `prior_content` implementation with three Hamiltonian evolutions
and two RZ interleavings, instead of two evolutions and one interleaving. Initial
count amplitudes, query/key dimension 16, feature-specific position weighting,
shared contextual spectral-shift vector, one auxiliary qubit and direct token
measurement remain. Each evolution has its own learned basis and base spectrum.
No classical vocabulary output head is added. The extra interleaving includes its
own classical RZ-control features: this is a depth/interleaving recipe comparison,
not an exact isolation of quantum depth with every classical feature held fixed.

The candidate adds 278,902 parameters over the old model: 262,272 Hamiltonian
coordinates and 16,630 rotation-controller coordinates. The classical query/key
selector is unchanged. Context-dependent interleaving makes this different from
stacking input-independent unitaries. No efficient circuit compilation, native
hardware benefit, entanglement-specific gain or quantum advantage is assumed.

## Fixed comparison

| Label | Kind and input | Stages | Rank | Trainable parameters | Rate |
|---|---|---:|---:|---:|---|
| quantum | Full-context content quantum | 3 | 128 | 1,073,900 | Constant .003 |
| capacity | Full-context content quantum, wider | 2 | 176 | 1,071,526 | Constant .003 |
| short | Content quantum, independently refit with last 16 tokens | 3 | 128 | 1,073,900 | Constant .003 |
| fixed_basis | Content quantum with fixed bases/base energies | 3 | 128 | 287,084 | Constant .003 |
| mlp_cosine | Same-feature content MLP | 3-feature configuration | 128 | 1,073,506 | Cosine .003→.0003 |

The width control differs by 2,374 parameters (0.221%); the MLP differs by 394.
A wider spectrum also changes spectral-controller feature count, so total capacity
is approximately matched, not every parameter's location. Fixed bases still allow
context-dependent spectral shifts to learn; the entire Hamiltonian is not frozen.
The MLP replaces quantum evolution with a near-matched classical logit correction
using the same prior and context-feature family. Keep the archived seed-11 two-stage
content quantum (2.579836 NLL, 794,998 parameters) and strongest adaptive cosine
MLP (2.407749 NLL, 750,460 parameters) visible with their original locks.

All new fits start fresh at seed 11. Use 30 epochs, batch 512, Adam, clipping 5,
NLL loss, float64/complex128 and the same 524,288 training targets: **30,720 updates
per fit**. Select minimum validation NLL across initialization and all 30 epochs,
breaking ties by sampling success. No early stopping or score-dependent changes.
Window 128, vocabulary 512, recent window 16, auxiliary dimension two and phase
scale one remain unchanged. The short dataset masks the first 112 positions during
both training and validation without changing targets or statistics.

Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
Keep all training-only byte BPE/count statistics, article partitions, UNK policy,
target-byte counts, 4,096 validation targets and 8,192 test targets unchanged.
No held-out test scoring. All 60 validation articles were previously exposed;
this is an exploratory single-seed screen requiring later seed and independent
confirmation, not a fresh benchmark.

## Decision and reporting

Retain the deeper recipe only if selected full-context NLL beats both the archived
two-stage content model and the fresh rank-176 capacity control. A win only over
the smaller archived model cannot establish a depth benefit. Report every fit even
if this rule fails. Useful older context separately requires full minus independently
refitted-short NLL to have a descriptive article interval entirely below zero and
improved non-space NLL. Keep all sampling/argmax tradeoffs and classical deficits.

Report NLL, arithmetic sampling success S, argmax, sampled-target bits per byte,
space/non-space buckets, parameters, selected epochs and fit times. Retain bigram,
unigram and train-modal baselines, and paired article intervals conditioned on the
fits. These intervals do not correct repeated selection or seed uncertainty.
Reload every selected checkpoint on CUDA and reproduce its saved predictions.
Verify complete 31-row histories and selection locks before accepting results.

Generate three established prompts with seeds 20261030–20261032 and 64 direct
measurement-equivalent draws per full-context fit and count baseline. Keep all
invalid/control outcomes and raw IDs; omit short-model generation through the
unmasked generic interface. Samples are diagnostics, not proof of coherent text.

## Verification and resources

No model-source change is required. Existing independent dense-Hamiltonian tests
already exercise three-stage prior/content outputs and gradients beyond identity;
the three targeted cases pass locally. Generated tools pass Ruff/strict Pyright,
parameter counts and 16 reproduced validation rows per archived reference.

Use `prod10`, one `nvidia_a100_1g.10gb` MIG slice and four CPU support threads.
Reserve six hours; estimated total fitting is roughly three to five hours, pending
the full-size CUDA gate. Before any scientific fit, rerun all tests and measure
all five configurations with resident training data, checkpoint snapshot and Adam
state, three warmup updates and three synchronized blocks of eight updates.
Require actual CUDA tensors, finite gradients, optimizer changes, active query/key
gradients and exact initial count probabilities. Gate updates are discarded.
The MLP gate uses constant .003 only for resource measurement.

Record allocator peak memory and update timings. Stop before fitting if summed
30,720-update projections exceed five hours or any configuration reserves at least
80% of slice memory; review resources rather than changing batch/update budgets.
These short projections omit evaluation/I/O and do not guarantee every fit peak.
No full-A100 escalation is part of this protocol.

Tools, frozen source/reference snapshots and input hashes live under
`output/depth-screen/`; use isolated DGX checkout
`~/quantum-semantics-depth-screen-20261109`. Submit once, then run **`dgx start` on
the laptop** for the requested queue notification. Do not use codex-long-jobs.
Verify CUDA startup and stop model-side waiting while the Slurm job runs. A
queue-empty notification is not proof of success: inspect accounting, retrieve
artifacts and run the prepared independent audit after completion. Network failure
does not authorize resubmission or optimizer continuation.
