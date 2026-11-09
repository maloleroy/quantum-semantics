# Seed replication of content-dependent controls

9 November 2026. Declared before new fitting. The single-seed content screen
improved NLL over position models but failed the older-context criterion and
remained behind classical MLPs. Test whether the small gain survives new seeds
before changing the model again.

## Fixed recipes and archived evidence

Run six fresh fits: full 128-token `prior_content`, position-only `prior_position`
at matched-capacity rank 135, and independently refitted 16-token `prior_content`,
each at seeds **22 and 33**. Retain the exact recipes from the
[audited content screen](content-screen.md#completed-results-audited-9-november-2026):
rank 128 for content models, two layers, one auxiliary qubit, recent window 16,
query/key dimension 16, phase scale 1, float64/complex128, Adam .003 constant,
clipping 5, 30 epochs and batch 512. Each fit gets **30,720 updates** and
31 validation checks including initialization. Select minimum validation NLL,
breaking ties by sampling success. No early stopping or changes after seeing scores.

| Family | New fits | Reused fits | Trainable parameters |
|---|---|---|---:|
| content | Seeds 22, 33 | Seed 11, job 10911 | 794,998 |
| capacity | Seeds 22, 33 | Seed 11, job 10911 | 794,365 |
| short | Seeds 22, 33 | Seed 11, job 10911 | 794,998 |
| position | None | Seeds 11, 22, 33, jobs 10898/10906 | 754,038 |
| mlp | None | Strong archived cosine adaptive MLP, all three seeds | 750,460 |

All nine references retain their own source, weights, data/configuration and
selection locks. Reuse avoids redundant fits. The archived MLP keeps its
.003-to-.0003 cosine schedule. Fixed-basis and same-feature content MLP results
remain visible in the original report but are not replicated in this study.
No model source changes, new attribution claim, quantum advantage or coherent
long-text claim are introduced. The classical content selector remains explicitly
counted; candidate output still comes from quantum evolution and direct Born
measurement without a classical vocabulary head.

Keep the exact training-only byte BPE/count statistics and article partitions:
524,288 training targets, 4,096 validation targets, 8,192 held-out test targets,
vocabulary 512 and window 128. Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
Short contexts mask the first 112 positions without changing targets, groups,
byte counts, vocabulary or statistics. No test scoring, repartitioning or reselection
of archived checkpoints. All 60 validation articles were previously exposed;
seed robustness on this partition is not independent confirmation.

## Decisions and reporting

Require full-content NLL to beat both the archived position model and the fresh
matched-capacity position control at **each new seed** and on the three-seed mean.
A better average cannot hide a failing new seed. Report every result, including
sampling/argmax tradeoffs and the gap to the strong classical reference.

A useful older-context gain separately requires full minus short NLL below zero
at both new seeds, the three-seed mean paired article interval entirely below zero,
and improved three-seed mean non-space NLL. The previous seed-11 result favored
short context; changes in learned selection weights cannot substitute for this test.

Report 15 individual fits plus bigram/unigram/train-modal baselines, space/non-space
buckets, means/sample standard deviations/ranges, and paired article intervals
per seed and for the mean per-example difference. Average model scores are not
ensemble probabilities or a multi-shot prediction protocol. Intervals condition
on the three trained models and omit adaptive selection and broader seed uncertainty.

Reload selected weights on CUDA and reproduce saved predictions before accepting
reports. Check all 31 history rows and selection locks. For the four new full-context
fits, retain three 64-token direct-draw samples each using the established prompts
and seeds 20261030–20261032. Keep invalid/control outcomes and raw token IDs.
Do not generate short-model samples through the generic full-window interface;
archived samples remain archived. Use CSV for tables/history, NPZ for arrays/weights,
and small JSON for metadata.

## Resources and execution

Use `prod10` with one `nvidia_a100_1g.10gb` MIG slice and four CPU support threads.
Previous full fits measured 26.31 minutes content, 23.26 capacity and 50.73 short.
Two repetitions imply **3h20m** plus setup/reporting and timing variability; use a
six-hour allocation ceiling. All shapes, batches and update budgets are unchanged;
the completed MIG resource probe and full fits establish feasibility without a
full A100 request.

Generated tools, frozen source and reference/input manifests live in
`output/content-seeds/`. Use isolated DGX checkout
`~/quantum-semantics-content-seeds-20261109`. Local checks cover strict typing/linting,
parameter budgets, all reference locks, data/masks and 16 reproduced prediction
rows per reference. Three reporting checks reject probability-mixture substitution,
a failed seed hidden in an average, and unsupported older-context claims.

Before training, rerun all CUDA tests and check actual full-size CUDA forwards,
finite gradients, optimizer changes and count-prior initialization for all six
model/seed configurations. Also require active query/key gradients for content
models. Training starts fresh after these disposable checks. No CPU fallback.
Submit once with codex-long-jobs; observer failure does not authorize resubmission.
After verified startup, pause goal work while the experiment and completion
delivery continue. No new-seed results are available at this declaration.
