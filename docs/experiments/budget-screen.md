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
