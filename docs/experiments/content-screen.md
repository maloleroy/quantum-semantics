# Content-dependent controls: predictive comparison

5 November 2026. Declared before scientific fitting; exploratory validation reuse.

## Hypothesis and quantum prediction path

The replicated position model improves local prediction but has not demonstrated
useful older context. Its lag preferences are fixed across examples. Test whether
selecting preceding tokens according to their content and the last four query
tokens improves prediction. This is a representation hypothesis, not an established
explanation of the previous limitations.

`prior_content` adds the classical query/key selector documented in the
[implementation and resource probe](content-controller.md). It retains the
train-count amplitude prior, two contextual rank-128 Hamiltonians, local RZ
interleaving and direct Born token measurement, ignoring one auxiliary qubit.
No classical vocabulary head bypasses the quantum dynamics. Position softmax and
content selection remain classical preprocessing; no quantum-attention or hardware
speedup claim follows. All fixed count caches, arbitrary amplitude preparation
and general basis costs remain counted. Zero keys reproduce position pooling;
zero spectra reproduce the count prior. Probe updates are discarded completely.

## Fixed configurations and budget

| Label | Recipe | Rank | Trainable parameters | Learning rate |
|---|---|---:|---:|---|
| quantum | Full 128-token content quantum | 128 | 794,998 | Constant .003 |
| capacity | Position-only quantum with matched capacity | 135 | 794,365 | Constant .003 |
| short | Content quantum, independently refit with last 16 tokens | 128 | 794,998 | Constant .003 |
| fixed_basis | Content quantum with fixed bases and base energies | 128 | 270,454 | Constant .003 |
| mlp_cosine | Same-feature content MLP | 128 | 794,316 | Cosine .003 to .0003 |

The capacity control has 633 fewer parameters (0.080%). Fixed bases still permit
context-dependent spectra to train, so this is not a fully frozen Hamiltonian.
The short refit retains parameter shapes but masks the first 112 context positions
in both training and validation. The MLP receives identical features and prior,
then uses a classical logit correction in place of quantum evolution.

All five fits start fresh at seed 11. Use vocabulary 512, window 128, two layers,
auxiliary dimension two, recent window 16, query/key embedding dimension 16,
float64/complex128, Adam, clipping 5, batch 512 and 30 epochs. The unchanged
524,288 training targets imply **30,720 optimizer updates per fit**. Select by
minimum validation NLL, breaking ties by sampling success, from epoch zero and
all 30 completed epochs. No early stopping or changes based on emerging scores.
The cosine MLP schedule preserves the prior comparisons' stronger training recipe;
its timing probe used constant .003 only for resource measurement.

Retain the archived seed-11 position quantum (NLL 2.5951237, 754,038 parameters)
and strongest adaptive cosine MLP (NLL 2.4077489, 750,460 parameters). Pin each
reference's source, weights, selection metadata and validation arrays. Reproduce a
bounded subset before submission; no reference fit or sample is rerun for selection.
Historical full-GPU timings are not a controlled speed comparison with this MIG run.

Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The same 4,096 validation targets, 8,192 test targets, training-only byte BPE/counts,
article partitions, UNK policy and target-byte counts remain fixed. No test scoring.
All 60 validation articles have already been exposed: this is adaptive exploration,
not independent confirmation. Article bootstrap intervals condition on the trained
fits and do not correct repeated selection or seed uncertainty.

## Decisions and reporting

Retain content controls as a candidate only if full-context selected NLL beats
both the archived position model and the freshly fitted capacity control. Report
all outcomes even if this rule fails. A candidate gain requires later seed
replication; it does not inherit the position model's replicated evidence.

Claim useful older context only if full minus refitted-short NLL has a descriptive
paired article interval entirely below zero and non-space NLL also improves.
A gain over fixed bases supports learned-basis utility within this model family;
it does not isolate entanglement or establish quantum advantage. Keep both the
same-feature MLP and strongest archived MLP visible.

Report NLL, arithmetic sampling success S, argmax accuracy, sampled-target bits
per byte, space/non-space buckets, selected epochs, fit time, parameter counts,
and paired article intervals. Retain bigram, unigram and train-only modal baselines.
Reload all five selected checkpoints on CUDA and reproduce their saved validation
arrays before accepting reports. Verify complete 31-row histories and selection
locks. Generate the three established prompts with seeds 20261030–20261032 and
64 direct token draws for full-context fits and count baselines. Preserve every
invalid/control outcome and token array; omit short-model generation because the
generic generation path does not enforce its context mask. No filtering, retries
or rejection sampling. Predictions and samples are distinct evidence.

## Resources and execution

The completed bounded probe, Slurm 10910, passed all 142 tests and real CUDA
forward/gradient/update checks for all five configurations on the **10 GB MIG slice**.
It measured at most 1.018 GiB peak reserved allocator memory with full resident
training data. Update-only projections total 2.44 hours; full fitting also includes
validation, selected-training scoring and I/O. Keep `prod10` and
`--gres=gpu:nvidia_a100_1g.10gb:1`, four CPU support threads and a six-hour ceiling.
No budget reduction or full-A100 escalation is needed.

Generated tools, frozen source, reference locks and inputs are in
`output/content-screen/`; use isolated DGX checkout
`~/quantum-semantics-content-screen-20261105`. Local generated-tool Ruff and strict
Pyright checks, parameter budgets, data partitions/masks and 16 archived prediction
rows per reference pass. The allocation reruns all tests and full-size CUDA
learning gates, including active query/key gradients, before starting fresh fits.

Submit once with the codex-long-jobs controller. An observer connection failure
does not authorize training restart. After verified GPU startup, pause goal work
as requested while the experiment and completion delivery remain active. At this
declaration, no predictive results from the new content model are available.


## Verified submission

Slurm **10911** started on 5 November 2026 in `prod10` with one
`nvidia_a100_1g.10gb` slice under controller `qcse-content-screen-20261105-01`.
All **142 tests passed**. Every full-size configuration passed CUDA forward,
finite-gradient, optimizer-change and initial-prior checks; content configurations
also showed active query/key gradients. The first `quantum` fit entered training.
The experiment Python process was observed with **1,144 MiB GPU memory**; this
is device residency evidence, not a measured utilization percentage. All 93 pinned
source/data/tool/reference hashes still match. Startup evidence is retained in
`output/content-screen/startup/`.

Goal work pauses after verified startup, as requested, while the worker and queued
completion delivery remain active. No predictive results are available yet. Do not
edit the frozen checkout or restart training because an observer connection fails.


## Recovery in progress (9 November 2026)

The local controller exited 255 on 5 November at 22:19 after SSH reported
`No route to host`. On reconnecting, Slurm accounting confirms **10911 completed
with exit 0:0 in 02:28:55**. All five fits, locked-prediction checks and reporting
finished remotely, ending with `CONTENT_SCREEN_COMPLETE`. Both the remote input
manifest and generated-artifact checksums pass. No training was restarted.

Recovery worker `qcse-content-recover-10911` is retrieving the original artifacts
and will check their local hashes and locks. Independent report-audit code has
been prepared and passes Ruff/strict Pyright. Its first invocation encountered a
not-yet-transferred Slurm log, so the independent local audit remains pending;
remote completion is not yet a fully audited scientific result. Goal work pauses
while transfer and completion delivery continue.


## Completed results (audited 9 November 2026)

Recovery `qcse-content-recover-10911` completed successfully. All 93 pinned inputs
and generated-artifact hashes passed locally as well as remotely. Slurm 10911
completed all five fits in **02:28:55** on the prod10 MIG slice. GPU reloads
reproduced each locked checkpoint's validation predictions. The independent local
audit reproduced five complete histories and selections, two archived references,
three baselines, 10 metric rows, 20 target buckets, 26 paired intervals, both
decision rules and all 18 sample renderings. The earlier missing local log was
only an incomplete transfer; the completed audit passes. No test scoring occurred.

| Recipe | Validation NLL | Sampling S | Argmax | Selected epoch |
|---|---:|---:|---:|---:|
| Content quantum, full 128 | **2.579836** | **27.296%** | **43.115%** | 27 |
| Position quantum, matched capacity | 2.596239 | 26.582% | 42.505% | 29 |
| Content quantum, refitted 16 | **2.560480** | 27.510% | 43.359% | 29 |
| Content quantum, fixed bases | 2.987029 | 23.197% | 35.571% | 20 |
| Same-feature cosine MLP | 2.411385 | 38.581% | 47.852% | 14 |
| Archived position quantum | 2.595124 | 26.661% | 42.432% | 29 |
| Strong archived cosine MLP | **2.407749** | **39.180%** | **47.925%** | 20 |
| Bigram | 3.059255 | 23.158% | 34.033% | — |

The candidate-retention rule **passes**. Full content minus archived position NLL
is **−.015287**, conditional article interval **[−.030813, −.000776]**; versus
matched-capacity position it is **−.016403**, interval **[−.027372, −.005223]**.
Sampling improves by .635 percentage points over archived position and .714 points
over capacity. This is a small single-seed gain on repeatedly reused validation;
it requires replication and independent confirmation. An interval narrowly below
zero does not correct adaptive model selection or establish seed robustness.

The older-context criterion **fails**. Full minus independently refitted short NLL
is **+.019357**, interval **[+.003882, +.033149]**. Non-space NLL is also worse
for full: **3.51154** versus **3.48383**. Sampling is .214 percentage points lower
for full, interval [−.330, −.093]. Under this fit and target set, the short model
performs better; the query/key controller has not demonstrated useful older context.
Non-space sampling reaches **7.667%**, versus position **7.362%**, short **7.920%**,
and strongest MLP **19.303%**. Retain that gap alongside the aggregate improvement.

The full candidate beats fixed bases by −.407192 NLL, supporting learned-basis
utility within this family without isolating entanglement or quantum advantage.
It remains worse than the same-feature MLP by +.168451 and strongest archived MLP
by **+.172088 NLL**, the latter interval **[+.107592, +.224705]**. The strongest
MLP also leads by **11.884 sampling percentage points**. Classical preprocessing
is counted; the quantum candidate still has no classical vocabulary output head.

All three full-content quantum samples are valid UTF-8 without control characters,
but remain fragmented and incoherent. The capacity samples include one replacement
character; fixed-basis samples include five replacements and three other control
characters. The same-feature MLP samples are valid but include repetitive invented
strings. Every sample and raw token sequence is retained; no filtering or resampling.

Measured fit times are **26.31 minutes** full, **23.26** capacity, **50.73** short,
**24.95** fixed bases and **22.25** MLP. Their total is 2.46 hours, close to the
2.44-hour update-only probe projection. This validates using prod10 for these
configurations without changing batch/update budgets; it is not a hardware speedup
claim or a controlled comparison with archived full-A100 timings.

Retain content selection as a provisional quantum candidate. Next verify the
full/short/capacity comparison at seeds 22 and 33 under the identical recipe;
keep the replicated position baseline and strong MLP visible. Coherent long text,
useful older context, independent confirmation and quantum advantage remain open.

Evidence: [all metrics](content-screen-validation.csv),
[paired intervals](content-screen-paired.csv),
[target buckets](content-screen-target-buckets.csv),
[unfiltered samples](content-screen-samples.csv), and
[sample diagnostics](content-screen-sample-diagnostics.csv).
