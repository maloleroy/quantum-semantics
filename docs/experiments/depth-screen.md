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


## Submission record

Slurm **11039** was submitted on 9 November 2026 with the declared prod10 MIG
allocation and six-hour ceiling. All 95 source/data/reference/tool inputs are
pinned. The laptop command `bash -ic 'dgx start'` succeeded; systemd reports
`dgx-watchdog.service` active/running (observed PID 27351).

The bounded startup inspection found **PENDING**, so actual GPU startup and the
new configuration resource gates are **not yet verified**. Scientific fitting is
ordered after those checks in the frozen runner. Goal work pauses while the job
waits/runs, without model polling. Queue-empty notification must be followed by
accounting and artifact inspection; the prepared `output/depth-screen/retrieve.sh`
retrieves and audits results without submitting training again.


## Completed results (audited 10 November 2026)

Slurm **11039 completed with exit 0:0 in 03:03:34** on 9 November, after starting
at 20:48:47. The notifier stopped after six SSH connection failures; training and
reporting continued independently. Retrieval on 10 November verified all 95 pinned
inputs and every output hash without restarting training. All **142 DGX tests**
and all five full-size CUDA learning/resource gates passed on the prod10 MIG slice.

GPU reloads reproduced each selected checkpoint's validation predictions. The local
independent audit reproduced five complete histories/selections, two references,
three baselines, ten metric rows, 20 target buckets, 26 paired intervals, both
decision rules and all 18 saved sample renderings. Resource timing arithmetic and
memory/time guards also verify. No held-out test was scored.

| Recipe | NLL | Sampling S | Argmax | Selected epoch |
|---|---:|---:|---:|---:|
| Three-stage full content | **2.540152** | **27.966%** | 43.774% | 27 |
| Two-stage rank-176 capacity | 2.574756 | 27.594% | 43.848% | 29 |
| Three-stage refitted 16-token | **2.522583** | **28.238%** | **43.872%** | 28 |
| Three-stage fixed bases | 2.975113 | 23.282% | 36.182% | 19 |
| Same-feature cosine MLP | 2.418089 | 37.433% | 47.119% | 12 |
| Archived two-stage content | 2.579836 | 27.296% | 43.115% | 27 |
| Strong archived cosine MLP | **2.407749** | **39.180%** | **47.925%** | 20 |
| Bigram | 3.059255 | 23.158% | 34.033% | — |

The candidate-retention rule **passes**. Three-stage minus archived content NLL
is **−.039685**, conditional article interval **[−.058310, −.019507]**; versus
matched-width capacity it is **−.034605**, interval **[−.053236, −.017082]**.
Sampling improves by .670 percentage points over archived content and .372 points
over width. Argmax is slightly lower than width by .073 points, with an interval
crossing zero. The NLL gain is not explained merely by total parameter count under
this budget. This compares depth/interleaving recipes, including the counted extra
classical RZ features, rather than isolating quantum depth alone. It remains a
single-seed screen on reused validation and requires replication.

The older-context criterion **fails**. Full minus refitted-short NLL is
**+.017568**, interval **[+.000884, +.032409]**; non-space NLL is worse for full,
**3.45961 versus 3.43505**. Sampling is .272 percentage points lower for full.
The short refit remains the stronger context-length control even after extra depth.
Full non-space sampling improves from **7.667% to 8.389%**, versus width **8.080%**,
short **8.680%**, and strongest MLP **19.303%**.

Three stages beat fixed bases by −.434961 NLL, supporting learned-basis utility
within this family without establishing entanglement or quantum advantage.
The same-feature MLP remains better by .122062 NLL, and the strongest archived
MLP by **.132403**, interval [.071541, .182098] for quantum minus MLP. The latter
also leads by **11.214 sampling percentage points**. The new, larger same-feature
MLP does not supersede the stronger archived reference.

The three full quantum samples remain incoherent and include one UTF-8 replacement
character. Width samples contain no invalid/control characters; fixed bases produce
six replacements and one other control character. Same-feature MLP samples also
remain weak and include repetitive punctuation. Every saved sample and token ID
is retained without filtering, including adverse outcomes.

Measured fitting takes **32.60 minutes** full, **30.53** width, **62.45** short,
**30.45** fixed bases and **25.98** MLP. Probe peak reserved memory stays below
**1.18 GiB**, and projected update-only time totals 3.01 hours. The completed
allocation confirms default prod10 feasibility with unchanged training budgets;
no full A100 is needed for this recipe.

Retain three-stage refinement as provisional. Replicate full/width/short at seeds
22 and 33 before treating it as a robust new quantum reference. Preserve the
short-context advantage, classical gap, bad samples and lack of independent
confirmation. Coherent long text and useful long-context generation remain unresolved.

Evidence: [all metrics](depth-screen-validation.csv),
[paired intervals](depth-screen-paired.csv),
[target buckets](depth-screen-target-buckets.csv),
[unfiltered samples](depth-screen-samples.csv),
[sample diagnostics](depth-screen-sample-diagnostics.csv),
[resource measurements](depth-screen-resources.csv), and
[timing blocks](depth-screen-timings.csv).
