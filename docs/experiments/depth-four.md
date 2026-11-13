# Fourth Hamiltonian stage at the longer training budget

12 November 2026. Declared before training; results pending.

The audited 90-epoch screen improves quantum NLL to 2.43218, but a strong MLP
reaches 2.17231. Full quantum selects epoch 68 and ends near that value (2.43346);
its selected training NLL is 2.40298 versus MLP 1.91087. This suggests testing
additional quantum capacity, without claiming that capacity uniquely explains the
gap. The older 2-to-3-stage study improved quantum NLL against a width control;
its results do not establish that a fourth stage will help on the expanded pool.

## Locked fits and controls

Three fresh seed-11 fits use the same full 128-token contexts, 2,097,152-position
training pool and 90-epoch / 92,160-update budget as the just-completed study:

| Label | Model | Stages / rank | Parameters | Rate schedule |
|---|---|---|---:|---|
| quantum | prior_content | 4 / 128 | 1,352,802 | Constant .003 |
| width | prior_content | 3 / 164 | 1,355,060 | Constant .003 |
| matched | prior_content_mlp | 4 / 128 features | 1,352,364 | Cosine .003 → .0003 |

Pin the completed 90-epoch seed-11 full quantum (3 stages/rank 128, NLL 2.43218),
short quantum (2.43510), and strong adaptive MLP (2.17231) as references. Preserve
their original source, configuration, selected weights, histories and saved
validation arrays. The strong MLP remains a required comparison even if the new
matched classical model performs worse.

The fourth stage adds 278,902 trainable coordinates: 262,272 Hamiltonian
basis/energy coordinates and 16,630 additional RZ-controller/position coordinates.
Thus about 94% of the added capacity is in the quantum dynamics. The width control
has 2,258 more parameters than the deep candidate (0.167%); the matched MLP has
438 fewer (0.032%). Stage-specific bases/base energies remain trainable and the
contextual spectral controller is shared across stages, as in the retained model.
The direct Born output has no classical vocabulary head, rejection, temperature
change or decoded shortcut. Count-prior preparation and content pooling remain
unchanged. These are supported configurations, requiring no source-model edits.

The depth recipe includes extra classical RZ features; the wider recipe also has
more spectral features. The matched MLP uses the candidate's same prior/context
feature construction with a classical correction to count logits. Its cosine
schedule follows the stronger classical training choice. This compares complete
recipes with counted capacity, not an isolated physical-depth intervention or
proof of a quantum advantage at equal information and training cost.

No new four-stage short-context fit is included in this initial capacity screen.
Comparisons against the archived three-stage short fit are descriptive only and
cannot establish a four-stage long-context benefit. A promising candidate needs
its own refitted short control, seed replication and independent confirmation.

## Data, optimization and decisions

Reuse the full expanded NPZ at `output/depth-coverage/data/full.npz`, SHA256
`37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`.
The masked dataset is needed only to verify the archived short reference, SHA256
`1730223fedbf3dc864398053eca442d67466bd3f042bfee4fbd5395e57f209bb`.
Preserve the training-only byte BPE vocabulary of 512, count statistics, causal
contexts, article partitions, 4,096 validation targets and UNK policy. No test
predictions or changes to dataset construction.

Each epoch samples 524,288 rows as a fresh permutation prefix. All three fits use
seed 11 and identical row schedules: 47,185,920 exposures and 92,160 batch-512
updates. Adam, gradient clipping 5, float64/complex128, auxiliary size 2,
recent window 16, phase scale 1 and zero context dropout remain unchanged. Every
fit starts fresh; no optimizer resume. Initial predictions must reproduce the
same training-count bigram distribution.

Train/select by NLL at epoch zero and 90 endpoints, breaking ties by greater S
then earlier epoch. Lock selected weights before validation reproduction and sample
generation. Declare a depth/interleaving screen pass only if four-stage full beats
both the smaller full reference and nearly parameter-matched width control with
paired article-bootstrap NLL intervals wholly below zero **and** lower non-space
NLL in both comparisons. Report separate comparisons with the matched and strongest
classical models. Sampling S, argmax and NLL remain distinct; a depth pass does not
imply classical superiority, useful longer context or coherent generation.

Use the existing 2,000-draw, seed-161803 article bootstrap. These are exploratory
single-seed reused-validation comparisons; intervals omit checkpoint-selection and
training uncertainty. Report all fits, adverse outcomes and selected/final epochs.
Keep the 30-epoch replicated model as the established reference and the 90-epoch
model as an exploratory reference rather than silently promoting either candidate.

Save nine metric rows, 18 target buckets, 29 paired intervals and 15 unfiltered
64-token continuations from all three new full-context fits plus bigram/unigram.
Use the existing three prompts and seeds 20261030–20261032. Store raw IDs in NPZ,
histories/tables in CSV and compact metadata in JSON. Preserve malformed text.

## GPU execution and recovery

Use isolated `~/quantum-semantics-depth-four-20261112`, prod10 with one
`nvidia_a100_1g.10gb` slice, four CPU threads and an eight-hour ceiling. Planning
estimate roughly five to six hours, extrapolated from the prior fits; the actual
CUDA gates measure memory and throughput before training. Require projected
optimizer time below six hours and peak memory below 80% of the slice. Do not
change scientific budgets to fit resources or run intensive CPU fallbacks.

Before fitting: verify all pinned inputs, run the full project suite and two
reporting tests, then gate all three configurations on real CUDA. Check full-pool
data residency, CUDA parameters/buffers/forward/gradients, finite gradients,
nonzero content-selector gradients, changed predictions, initial prior identity,
synchronized update timings and memory including a checkpoint copy. Report
simulator execution costs without claiming quantum hardware efficiency.

After fitting: verify all three locked configurations, exact 91-row histories and
sampling traces, no test scores, selected epochs, hashes and preserved references;
reload selected weights on CUDA and reproduce validation arrays. Independently
recompute reports, baseline scores, intervals, decision rules and all sample text.
Require Slurm COMPLETED 0:0 and artifact hashes before accepting results.

Tools/artifacts live in `output/depth-four/`. Submit once with
`bash output/depth-four/submit.sh`, run local `bash -ic 'dgx start'`, verify actual
CUDA training, then pause while waiting under the user's standing instruction.
No codex-long-jobs. After notification, `bash output/depth-four/retrieve.sh`
retrieves and audits the existing job without submitting another.

## Submission status — 12 November 2026

Slurm **11198** was submitted once with the declared prod10 slice and eight-hour
ceiling. All 115 pinned inputs verify locally and remotely. Local reference checks
reproduce 16 archived validation rows per reference; parameter budgets, two
reporting tests, Ruff and strict Pyright pass. The laptop `dgx start` service is
active (PID 21900 at verification).

The job is **PENDING**, with no allocation or executed startup gates yet. Slurm
reports `Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved_for_jobs_in_higher_priority_partitions`;
the node is `mixed-` and `sinfo -R` lists no explicit drain reason. The scheduler's
tentative start is 13 November 06:14 (DGX scheduler time), not a guarantee. No
training or new predictive result is claimed. Keep the existing job queued; do not
resubmit because it has not started. Work stops during this scheduler wait under
the user's standing instruction. On resume, inspect this same job and verify its
CUDA gates or completed artifacts before further research.

## Completed results — audited 13 November 2026

Slurm **11198 completed with exit 0:0 in 05:18:52**, after its earlier scheduler
wait. All 115 pinned inputs and downloaded artifact hashes verify. The independent
audit checks three locked fits, three archived references, all 91-row histories
and exact sampling traces, nine metric rows, 18 target buckets, 29 intervals,
decisions and 15 saved sample decodings. All 149 project tests, two reporting tests
and three actual CUDA learning/resource gates passed. Selected weights were
reloaded on CUDA and validation arrays reproduced. No restart, new test score or
post-result training extension occurred.

| Fit | Selected epoch | Selected NLL ↓ | Final NLL ↓ | Sampling S ↑ | Argmax ↑ |
|---|---:|---:|---:|---:|---:|
| Four-stage quantum, rank 128 | 83 | **2.409054** | 2.411165 | 30.016% | 46.094% |
| Three-stage quantum, rank 164 | 90 | 2.410507 | 2.410507 | 29.803% | 45.532% |
| Matched prior-feature MLP | 87 | **2.166224** | 2.167899 | **42.398%** | **51.807%** |
| Smaller three-stage quantum reference | 68 | 2.432182 | 2.433464 | 29.205% | 45.142% |
| Strong adaptive MLP reference | 86 | 2.172312 | 2.179377 | 42.437% | 51.953% |

The declared **depth/interleaving rule fails**. Four-stage minus wider three-stage
NLL is **−.001453**, conditional article interval **[−.012469, +.010486]**. Its
non-space NLL is also slightly worse: **3.29193 versus 3.28759**. The result does
not establish either superiority or equivalence in NLL. Sampling is higher by
**.213 percentage points**, interval [.088, .334]; argmax is higher by .562 points,
with an interval crossing zero. This metric-specific sampling gain does not turn
the prespecified NLL/non-space rule into a pass.

Both capacity increases improve on the smaller three-stage reference at this seed:
four-stage NLL difference **−.023128 [−.038866, −.007011]**, wider three-stage
**−.021675 [−.038849, −.006535]**. Both improve non-space NLL from 3.31743, and
sampling rises .811 and .598 percentage points respectively. The screen supports
further study of capacity, without attributing the gain specifically to a fourth
Hamiltonian stage. Seed replication remains necessary before replacing established
references. There is no new matched short refit, so no four-stage longer-context
claim follows from beating the archived smaller short model.

The matched classical model remains substantially better: quantum minus matched
MLP NLL is **+.242830 [.186311, .287980]**. It leads quantum sampling by **12.382
points** and argmax by **5.713 points**. Non-space sampling is **10.134% quantum
versus 22.870% matched MLP** (23.169% for the adaptive MLP reference). The matched
MLP's NLL difference from the adaptive reference is −.006089, interval
[−.042244, +.034445]; that does not establish a new classical winner. Keep both
classical comparisons visible. Quantum overall S still lies just below the
training-modal-space baseline, 30.016% versus 30.054%, with uncertainty crossing
zero; the modal predictor's infinite NLL remains visible in the tables.

Selected training NLL is 2.35370 four-stage, 2.36175 wider and 1.94090 matched MLP.
At selection, all three fits have visited all 2,097,152 pool positions. Each
completed run processes 47,185,920 training-row exposures. The width fit still selects its final epoch;
that is a limitation of the fixed-budget comparison, not authorization to extend
only that fit. The matched classical result shows that this feature construction
supports better prediction under a different trainable map. It does not identify
whether quantum parameterization, optimization or another constraint explains
the remaining gap.

All nine new-model samples decode as valid UTF-8 without non-whitespace control
characters, but remain incoherent. The third matched-MLP sample repeats Q-prefixed number fragments.
No sample was rejected, repaired or used for extra model selection. Neither the
capacity gains nor the cleaner decoding establish useful long-text generation.

Fit times are **116.31 minutes** four-stage, **110.04 minutes** wider and **89.57
minutes** matched MLP. CUDA gates use the complete resident training data
(2,164,260,864 bytes), and peak reserved memory is at most **2.641 GiB**. All nine
synchronized timing blocks, gradient/learning checks and memory/time guards verify.
These are simulator measurements, not quantum hardware performance evidence.

Retain both quantum capacity variants as exploratory candidates, with no preference
for additional stages justified by this NLL screen alone. The replicated 30-epoch
reference and exploratory 90-epoch reference retain their stated evidence levels.
No further job was launched during this completion audit.

Compact evidence: [metrics](depth-four-validation.csv),
[paired intervals](depth-four-paired.csv), [target buckets](depth-four-target-buckets.csv),
[raw samples](depth-four-samples.csv), [sample diagnostics](depth-four-sample-diagnostics.csv),
[resources](depth-four-resources.csv), and [timings](depth-four-timings.csv).
