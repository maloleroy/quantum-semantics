# Longer optimization on the expanded training pool

11 November 2026. Declared before training; results pending.

The expanded-coverage replication retains full quantum at mean NLL 2.46546,
short quantum at 2.47748 and strong MLP at 2.22277. All six new fits selected
budget endpoint 30. At seed 22, full quantum NLL fell from 2.49268 at epoch 20
to 2.47087 at epoch 30; the short and MLP curves also improved. This motivates a
bounded longer-budget screen before adding model capacity. It does not establish
that optimization explains the classical gap. Earlier cosine decay on the older
adaptive model gave no clear quantum gain; this study keeps the current quantum
constant-rate recipe and tests additional updates on the larger pool.

## Locked comparison

Three fresh seed-11 fits, each with **90 logical epochs, batch 512 and 92,160
optimizer updates**, versus three pinned expanded-pool seed-11 30-epoch references.
No optimizer resume or loading selected weights as a training continuation.

| Label | Model | Layers / rank | Retained context | Parameters | Learning rate |
|---|---|---|---|---:|---|
| quantum | prior_content | 3 / 128 | 128 | 1,073,900 | Constant .003 |
| short | prior_content | 3 / 128 | Final 16, older positions masked | 1,073,900 | Constant .003 |
| mlp | adaptive_mlp | 2 / 128 | 128 | 750,460 | Cosine .003 to .0003 over 90 epochs |

Other architecture, Adam, clipping at 5, float64/complex128, direct Born readout
and no older-context dropout remain unchanged. The small content selector feeds
Hamiltonian dynamics; no classical vocabulary head is added to the quantum model.

Every epoch samples a fresh permutation prefix of 524,288 rows from the existing
2,097,152-position pool. Seed 11 controls initialization and sampling; all three
fits receive identical row schedules, totaling **47,185,920 row exposures**.
Full quantum and short quantum reproduce the old constant-rate trajectory through
30 epochs; verify their history prefix numerically to 1e-6 absolute/relative
tolerance. MLP cosine spans the new horizon, so its change is a longer training
recipe, not an isolated update-count effect or an identical 30-epoch prefix.

Reuse the exact expanded full/short NPZs in `output/depth-coverage/data/`:
full SHA256 `37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`,
short SHA256 `1730223fedbf3dc864398053eca442d67466bd3f042bfee4fbd5395e57f209bb`.
Preserve training-only 512-token byte BPE/counts, 128-column causal contexts,
article-disjoint partitions, 4,096 validation targets and the existing UNK policy.
No new data preparation, vocabulary fitting or held-out-test scoring. Test arrays
are checked for dataset identity/partition separation only, never predicted.

## Selection and decisions

Train and select by validation NLL at epoch zero and each of 90 endpoints; ties
use greater sampling success and then the earlier checkpoint. Lock selected
weights before reproducing predictions and generating samples. More checkpoints
increase selection opportunity. Report selected and final epochs, full histories,
selected-training NLL, exposure counts, timing and memory; do not interpret a lower
selected validation NLL alone as independent generalization evidence.

Declare two distinct exploratory decisions: the longer-budget full quantum must
beat its 30-epoch full reference with a paired article-bootstrap NLL interval
wholly below zero and lower non-space NLL; the older-context decision applies the
same conditions against the new 90-epoch short refit. Report new versus old short
and MLP, and full quantum versus the strongest classical reference even if it
widens the gap. Sampling success, argmax and NLL remain separate outcomes.

Use existing 2,000-draw, seed-161803 article bootstrap. These intervals condition
on one training seed and reused validation, and omit checkpoint-selection and
training uncertainty. A passing screen needs seed replication and independent
confirmation; it is not a quantum advantage or coherent-generation claim.

Save nine metric rows, 18 space/non-space buckets, 29 paired intervals and 12
unfiltered 64-token continuations from full quantum, MLP, bigram and unigram.
Use the established three prompts and seeds 20261030–20261032, save raw token IDs,
retain malformed UTF-8/control characters and do not reject or repair samples.
Omit short generation because the generic generator supplies full context.

## Execution and audit

Use isolated `~/quantum-semantics-budget-screen-20261111`, prod10 with one
`nvidia_a100_1g.10gb` MIG slice and four CPU support threads. Planning estimate
about 5h40m plus setup/reporting, based on tripling the measured 30-epoch fits;
eight-hour Slurm ceiling. Before fitting, run all project tests, two reporting
tests and actual CUDA gates for all three recipes: resident full training data,
model/buffers, forward passes, finite gradients, changed predictions, nonzero
selector gradients, synchronized update throughput and peak memory. Require less
than 80% slice memory and projected optimizer time below six hours. This does not
justify a full-GPU allocation or changing the scientific budget.

Pin all source, tools, dependencies, datasets and archived references. Verify
source/data/weight locks, 91-row histories, exact sampling exposure traces and
constant-rate prefixes. Reload all new selected weights on CUDA and reproduce
validation arrays. Independently audit metric tables, count baselines, intervals,
decisions and saved sample decoding; check Slurm COMPLETED with exit 0:0 and all
artifact hashes before accepting results.

Generated tools and artifacts live under `output/budget-screen/`; submit once with
`bash output/budget-screen/submit.sh`. After Slurm starts, run local
`bash -ic 'dgx start'`, verify live CUDA training, then pause goal work at the user's
standing request while the experiment runs. Do not use codex-long-jobs. On completion,
`bash output/budget-screen/retrieve.sh` retrieves and audits without resubmitting.

## Completed results — audited 12 November 2026

Slurm **11137 completed with exit 0:0 in 05:35:25**. The user pause interrupted
only the local startup observation; the remote job completed normally. Recovery
verified all 115 pinned inputs and downloaded artifact hashes, three selection
locks, all 91-row histories and sampling exposure traces, three references,
nine metric rows, 18 target buckets, 29 paired intervals, decision rules and 12
saved sample renderings. Both quantum 30-epoch prefixes reproduce. All 149 project
tests, two reporting tests and three real CUDA/resource gates passed. No restart
or held-out test scoring occurred.

| Fit | Selected epoch | Selected NLL ↓ | Final NLL ↓ | Sampling ↑ | Argmax ↑ |
|---|---:|---:|---:|---:|---:|
| Full quantum, 90 epochs | 68 | **2.432182** | 2.433464 | 29.205% | 45.142% |
| Short quantum, 90 epochs | 90 | 2.435101 | 2.435101 | 29.441% | 45.459% |
| Strong MLP, 90 epochs | 86 | **2.172312** | 2.179377 | **42.437%** | **51.953%** |
| Full quantum, 30 epochs | 30 | 2.462532 | 2.462532 | 28.486% | 44.214% |
| Short quantum, 30 epochs | 30 | 2.476735 | 2.476735 | 28.552% | 44.360% |
| Strong MLP, 30 epochs | 30 | 2.222069 | 2.222069 | 39.420% | 50.244% |

The **longer-budget rule passes**: full quantum gains −.030350 NLL, conditional
article interval **[−.041088, −.019161]**, and improves non-space NLL from 3.35333
to 3.31743. Sampling improves .719 percentage points, interval [.607, .820], and
argmax improves .928 points, interval [.456, 1.407]. This is a single-seed screen
with more validation selection opportunities. Retain the 30-epoch three-seed
reference as established evidence and the longer recipe as a promising candidate.

The **older-context rule fails at the longer budget**: full minus short NLL is
−.002919, interval **[−.013819, +.008246]**. Non-space NLL is 3.31743 versus 3.32392,
but full sampling is .236 points lower, interval [−.347, −.129], and argmax is
.317 points lower with an interval crossing zero. This does not overturn the
30-epoch seed replication; it limits transfer of that context claim to this recipe.

The classical gap widens again: full quantum minus MLP NLL is **+.259869**,
interval [.205429, .306494], versus +.240462 for the seed-11 30-epoch recipes.
MLP's own gain is −.049757 [−.072031, −.025357]; its longer cosine schedule also
changes the optimization trajectory. It leads sampling by 13.232 points and
argmax by 6.812 points. Non-space sampling is **9.414% quantum versus 23.169% MLP**.
All selected fits have visited the complete 2,097,152-position pool; selected
training NLL is 2.40298 full, 2.41248 short and 1.91087 MLP. These training/validation
numbers do not identify a unique cause of the gap.

Quantum and MLP samples remain incoherent. The third quantum continuation contains
one UTF-8 replacement character; none of the three MLP samples has replacement or
other control characters. All raw samples are retained without rejection or repair.
Neither lower NLL nor increased sampling establishes useful long-form generation.

Fit times are 97.61 minutes full, 185.38 short and 49.79 MLP. All three use CUDA with
the complete training contexts/targets resident (2,164,260,864 bytes); peak reserved
memory is at most 2.578 GiB. Nine synchronized timing blocks pass the six-hour
projection and memory guards. These are simulator costs, not quantum hardware
speedup. No architecture changes were needed for this experiment.

Compact evidence: [metrics](budget-screen-validation.csv),
[paired intervals](budget-screen-paired.csv),
[target buckets](budget-screen-target-buckets.csv),
[raw samples](budget-screen-samples.csv),
[sample diagnostics](budget-screen-sample-diagnostics.csv),
[resources](budget-screen-resources.csv), and [timings](budget-screen-timings.csv).
