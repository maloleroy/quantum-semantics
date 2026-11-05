# Seed replication of learned-position refinement

5 November 2026. Declared before new fitting. The previous single-seed screen
improved NLL and sampling over the prior and matched-capacity pooling, but did not
establish useful older context or close the classical gap. Verify robustness of
that improvement before changing the architecture again.

## Fixed recipes and reused evidence

Run six new fits: `prior_position` with full 128-token context, fixed-pooling
`prior` at rank 131, and independently refitted 16-token `prior_position`, each at
seeds **22 and 33**. Keep all recipes unchanged from the
[completed screen](prior-position.md#completed-results): rank 128 for the two
position models, two layers, one auxiliary qubit, recent-window 16, phase scale 1,
30 epochs, batch 512, Adam .003 constant, clipping 5 and float64/complex128.
Each fit receives 30,720 updates and 31 validation checks including initialization.
Train and select on NLL, with sampling success as the selection tiebreaker.

| Family | New fits | Reused fits | Trainable parameters |
|---|---|---|---:|
| position | Seeds 22, 33 | Seed 11 from job 10898 | 754,038 |
| capacity | Seeds 22, 33 | Seed 11 from job 10898 | 753,414 |
| short | Seeds 22, 33 | Seed 11 from job 10898 | 754,038 |
| prior | None | Seeds 11, 22, 33 from prior screen/replication | 736,512 |
| mlp | None | Strong cosine adaptive MLP at seeds 11, 22, 33 | 750,460 |

All nine reused fits have pinned source/configuration/data/weight locks and aligned
target arrays. Reuse avoids redundant training; the strong MLP keeps its declared
.003→.0003 cosine schedule. The previous fixed-basis and same-feature MLP controls
remain in the original report but are not seed-replicated here. This study does not
newly establish quantum attribution, coherence, entanglement or hardware advantage.
No model-source changes are introduced.

Use exactly the same training-only byte BPE and count statistics, 524,288 gradient
targets, 4,096 validation targets and 8,192 held-out test targets. Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The short dataset masks the first 112 positions without changing any targets,
groups, tokenizer or statistics. All article partitions remain disjoint. Do not
score test, refit statistics, reselect archived checkpoints or sample a new partition.
All 60 official validation articles were already exposed: this is seed robustness
on reused validation, not independent article/corpus confirmation.

## Decisions and reporting

Require full-context position NLL to beat **both** prior and matched-capacity
pooling at **each new seed**, and on the three-seed mean against each control.
A better mean cannot hide a failing new seed. Report every seed even if the rule
fails, and preserve sampling/argmax tradeoffs and classical deficits.

For an older-context gain, additionally require full minus short NLL below zero
at both new seeds, the three-seed mean article-bootstrap interval entirely below
zero, and improved three-seed mean non-space NLL. The earlier seed-11 comparison
failed its older-context rule; changed learned lag weights alone cannot satisfy it.

Report all 15 individual fits plus bigram, unigram and train-modal references;
space/non-space buckets; per-family means, sample standard deviations and ranges;
and paired article intervals at each seed and for mean per-example differences.
The latter condition on three trained models and do not include all seed or adaptive
model-selection uncertainty. These are averages of model scores, not ensemble
probabilities or a multi-shot prediction scheme.

For the four new full-context fits, retain three 64-token direct draws each with
the same prompts and seeds 20261030–20261032. Keep all invalid/UNK/control outcomes.
Omit short-model generation through the generic full-window interface. Archived
samples remain archived and are not resampled. Reproduce selected predictions on
GPU before accepting report tables. Save histories/tables as CSV and arrays/weights
as NPZ, with only small JSON metadata.

## Execution

Generated scripts and frozen input/reference/source snapshots are under
`output/position-seeds/`. Use the isolated checkout
`~/quantum-semantics-position-seeds-20261105`, one full A100 and four CPU support
threads. Estimate **2–3 hours**, with a six-hour allocation ceiling; the previous
short fit's unusually high wall time makes this estimate uncertain. Before research
training, require all CUDA tests and full-size forward/gradient/optimizer checks
for all six model/seed configurations. No CPU training fallback.

Local checks cover all input/reference locks and full/short masks, strict generated
tool typing/linting, and reporting rules that distinguish averaged loss from a
probability mixture and reject hiding a failing seed in a better mean. Submit once;
network observer failure does not authorize resubmission. After verified startup,
pause goal work as requested while the experiment and completion delivery continue.
Results are not yet available.
