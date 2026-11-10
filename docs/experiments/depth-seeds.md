# Seed replication of three-stage quantum refinement

10 November 2026. Declared before new fitting. The seed-11 depth/interleaving
recipe improves NLL over both the old two-stage content model and matched-width
capacity. Replicate that gain while preserving the stronger short-context and
classical controls. No model-source change is introduced.

## Fixed recipes and references

Run six fresh fits at seeds **22 and 33**: three-stage rank-128 full content,
two-stage rank-176 full content, and independently refitted three-stage rank-128
short context. Keep the [completed depth screen](depth-screen.md#completed-results-audited-10-november-2026)
recipes unchanged: vocabulary 512, context 128, auxiliary dimension two, recent
window 16, query/key dimension 16, phase scale one, float64/complex128, Adam .003
constant, clipping 5, 30 epochs and batch 512. Each receives **30,720 updates**.
Select minimum validation NLL across epoch zero and all 30 epochs, with sampling
success as tiebreaker. No early stopping or changes based on emerging results.

| Family | New fits | Archived fits | Trainable parameters |
|---|---|---|---:|
| depth | Seeds 22, 33 | Seed 11, job 11039 | 1,073,900 |
| capacity | Seeds 22, 33 | Seed 11, job 11039 | 1,071,526 |
| short | Seeds 22, 33 | Seed 11, job 11039 | 1,073,900 |
| content | None | Two-stage rank-128, seeds 11/22/33 | 794,998 |
| mlp | None | Strong adaptive cosine MLP, seeds 11/22/33 | 750,460 |

All nine references retain their original data/configuration/source/weight locks
and aligned target arrays. Reuse avoids unnecessary fits. The archived MLP keeps
its .003-to-.0003 cosine schedule. Fixed-basis and same-feature MLP results stay
visible in the first screen; this replication does not newly replicate those
attribution controls. Depth adds mostly quantum Hamiltonian parameters, plus the
explicitly counted extra RZ controller. The query/key selector stays unchanged and
there is no classical vocabulary head in the quantum candidate. This is a
recipe-depth comparison, not a pure quantum-depth isolation or quantum advantage.

Preserve 524,288 training, 4,096 validation and 8,192 test targets, training-only
byte BPE/count statistics, article partitions, target-byte counts and UNK policy.
Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The short dataset masks the first 112 positions without changing targets or
statistics. No test scoring, repartitioning, reference reselection or optimizer
continuation. All 60 validation articles were exposed previously: this is seed
robustness on reused validation, not independent confirmation.

## Decisions and reports

Require full three-stage NLL to beat both the archived rank-128 content model and
fresh rank-176 capacity control at **each new seed** and on the three-seed mean.
A good average cannot hide a failing new seed. Report all results and preserve
sampling/argmax tradeoffs, classical deficits and adverse samples.

Useful older context separately requires full minus refitted-short NLL below zero
at both new seeds, a three-seed mean paired article interval entirely below zero,
and improved mean non-space NLL. The first screen favored short context; extra
quantum depth or an intervention sensitivity result does not override this rule.

Report the 15 individual fits, bigram/unigram/train-modal baselines, space/non-space
buckets, means/sample standard deviations/ranges, and paired article intervals
per seed and for mean per-example differences. These are averages of model scores,
not ensemble probabilities. Intervals condition on the fitted models and do not
correct adaptive selection or all seed uncertainty.

Reload all selected checkpoints on GPU and reproduce saved validation arrays.
Check complete 31-row histories and selection locks before accepting reports.
Retain three 64-token direct-draw samples for each of the four new full-context
fits, using the established prompts and seeds 20261030–20261032. Keep invalid and
control outcomes and raw IDs. Do not generate short-model samples through the
unmasked generic interface, and do not resample archived references.

## Execution and validation

Measured prior fits took 32.60 minutes full, 30.53 width and 62.45 short. Two
repetitions project **4h11m** plus setup/reporting and timing variability. Keep
`prod10`, one `nvidia_a100_1g.10gb` slice, four CPU support threads and a six-hour
ceiling. The completed study stayed below 1.18 GiB peak reserved memory during its
resource probe. No larger GPU or batch/update reduction is needed.

Before fitting, rerun all tests and full-size CUDA/resource gates for all six
model/seed pairs: three warmup updates, finite CUDA gradients and optimizer changes,
active query/key gradients, exact initial count prior, then three timed eight-update
blocks with all training rows and a checkpoint snapshot resident. Stop before
training if projected aggregate optimizer time exceeds five hours or peak reserved
memory reaches 80% of the slice. Discard these gate updates and start all fits fresh.
The gate's timing projection omits validation and I/O; it is not a guaranteed runtime.

Local setup verifies parameter counts, input/reference locks, full/short masks and
16 archived prediction rows per reference. Generated tools pass Ruff and strict
Pyright. Three reporting tests guard against probability-mixture substitution,
hiding a failed seed in a mean, and unsupported older-context claims.

Tools and frozen manifests live in `output/depth-seeds/`; isolated DGX checkout
`~/quantum-semantics-depth-seeds-20261110`. Submit once and run **`dgx start` on
the laptop**. Stop model polling while awaiting the run. If the notifier loses its
SSH route, the Slurm job may still continue; inspect that job rather than resubmit.
A queue-empty notification must be followed by accounting, input/output hashes and
the prepared independent report audit. No codex-long-jobs controller is used.


## Verified startup

Slurm **11058** started on 10 November 2026 on prod10 with one 10 GB MIG slice.
All **142 DGX tests passed**, and all six full-size CUDA learning/resource gates
passed, including finite CUDA gradients, changed predictions and active query/key
gradients. The first `depth_s22` scientific fit entered training. Peak reserved
memory stays below 1.18 GiB; update-only projections total approximately 4h07m,
within the declared guard. All 232 pinned inputs still match.

The laptop `dgx start` command succeeded and the systemd notifier is verified
active/running (observed PID 125490). Startup evidence is retained in
`output/depth-seeds/startup/verification.txt`. Goal work pauses while the experiment
runs. No new predictive result is available yet, and no input/source is changed
beneath the running allocation. After notification, inspect accounting and use the
prepared retrieval/audit script; do not treat an SSH failure as a training failure.
