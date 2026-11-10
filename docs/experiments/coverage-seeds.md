# Replicate the expanded-coverage and older-context NLL results

## Declared protocol — 10 November 2026

The expanded-pool seed-11 experiment improved full quantum NLL from **2.54015 to
2.46253** at the same update budget and beat the independently refitted short
model (**2.47673**). Both declared interval/non-space criteria passed. However,
full sampling was slightly lower than short, all models selected the final epoch,
and the MLP improved more (**2.22207**), widening the classical gap. Generation
remained incoherent. Test seed robustness before strengthening the context claim.

Run six fresh fits: expanded full quantum, expanded short quantum and expanded
strong MLP, each at **seeds 22 and 33**. Preserve all recipes, datasets and selection
rules. Do not add epochs, change rate schedules, tune dropout or modify the model.
The retained quantum prediction path remains count-prior state preparation,
three contextual Hamiltonian stages and Born measurement without a classical
vocabulary decoder. This is replication of a data/training recipe, not an additional
architecture or quantum-advantage experiment.

## Fixed inputs and exposure

Use the same nested 2,097,152-position pool (30.57% of available training transitions)
and its short-context counterpart under `output/depth-coverage/data/`. Tokenizer,
train-only statistics, article grouping, targets and held-out arrays remain fixed.
Vocabulary is 512, context 128, validation 4,096 and excluded test 8,192 targets.
Short inputs retain only the latest 16 tokens. No test scoring occurs.

Full SHA256:
`37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`.
Short SHA256:
`1730223fedbf3dc864398053eca442d67466bd3f042bfee4fbd5395e57f209bb`.
The 524,288-position old-pool full reference retains SHA256
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
No new positions are sampled into the pool and no vocabulary or count table is refit.

Each fit uses **30 logical epochs**, drawing the first 524,288 rows of a fresh
permutation of the full pool per epoch, without replacement within each epoch.
Initialize the NumPy sampler once with the fit seed. Full, short and MLP share
exactly the same row schedule **within each seed**. Seeds 22 and 33 change both
initialization and training permutations; this is whole-recipe seed robustness.
Keep batch 512, **30,720 updates**, **15,728,640 row exposures**, float64/complex128,
Adam, clipping norm 5, NLL loss and 31 validation-selection opportunities. Select
lowest NLL, then highest sampling S, then the earliest epoch on an exact tie.
Dropout remains zero. Histories retain per-epoch counts and cumulative unique rows;
report selected-checkpoint exposure separately from final exposure.

| Family | Layers / rank | Parameters | LR schedule |
|---|---|---:|---|
| Full quantum (`prior_content`) | 3 / 128 | 1,073,900 | Constant .003 |
| Short quantum (`prior_content`) | 3 / 128 | 1,073,900 | Constant .003 |
| Strong MLP (`adaptive_mlp`) | 2 / 128 | 750,460 | Cosine .003 to .0003 |

Auxiliary size is 2, recent window 16, phase scale 1. The MLP retains amplitude
decay .9 and quantum the recorded .7; count-prior preparation does not use that
decay. The MLP is the strongest retained equally exposed competitor, not an exact
parameter/schedule match. Selected training-pool NLL is on the larger pool and
must not be compared to old-pool training NLL as if examples were identical.

## References and declared replication criteria

Pin **12 archived fits**: expanded full/short/MLP at seed 11, and old-pool
full/short/MLP at seeds 11/22/33. Verify original selection metadata, configuration,
training settings, dataset/source/weight hashes and saved predictions. Preserve
historical metadata fields rather than editing old locks to match newer defaults.
Copy source snapshots and a reference-origin table. Keep the original seed-11
sample IDs/text, including the invalid quantum sample, alongside the references.

For each of **coverage** (expanded full minus old-pool full) and **older context**
(expanded full minus expanded short), require all of the following:

- Lower NLL at **each new seed**, 22 and 33.
- Lower non-space NLL at **each new seed**.
- A three-seed mean NLL article interval entirely below zero.
- Lower three-seed mean non-space NLL.

Report the two decisions separately. A positive mean cannot hide a failed new
seed or non-space regression. Do not require each individual seed's interval to
exclude zero; show all intervals so this distinction stays visible. Seed 11 is
retained in the mean and already passed its single-seed rules. These decisions
apply to NLL, not sampling or argmax; preserve those scores and their uncertainty.

For each seed and the three-seed mean, report paired NLL/S/argmax differences for
expanded full versus old full, expanded full versus expanded short, expanded full
versus expanded MLP, expanded short versus old short, and expanded MLP versus old
MLP (**60 interval rows**). Use 2,000 article bootstrap replicates, seed 161803.
For the mean comparison, average the paired per-target metric differences across
seeds before resampling article groups. These are mean model losses/probabilities,
**not predictions from an ensemble**. Report mean, sample SD, min and max across
seeds for all six families, with all 18 individual fits and three count/modal
baselines visible (21 metric rows, 42 target buckets).

The same validation articles have already guided development. Three-seed evidence
is not fresh-partition confirmation, and article intervals do not quantify all
initialization or adaptive-selection uncertainty. Neither passing criterion proves
quantum advantage, entanglement-specific benefit, hardware speedup or meaningful
long-form generation. Keep the strongest new MLP comparison explicit even if
quantum improves over its own references.

Generate three unfiltered 64-token continuations for each new full quantum and
MLP fit (**12 new samples**), using the existing prompts and draw seeds 20261030–32.
No draws are repaired/rejected or temperature-adjusted. Save IDs and decode them
independently during audit. Short generation remains omitted through the generic
full-window generator. Archived adverse outcomes remain visible.

## Execution and completion evidence

The previous full/short/MLP fits took 33.30/62.64/16.88 minutes. Two replications
project about **3h46m of fitting**, plus setup/reporting; plan **3h45m–4h15m** on
one **prod10 `nvidia_a100_1g.10gb`** slice with four CPU support threads and a
six-hour allocation. Previously measured peak reserved memory was at most 2.58 GiB.
Keep the batch and update budget unchanged; no full-GPU escalation is assumed.

Before any fresh fit, require all project CUDA tests, the reporting tests and
six actual GPU learning/resource gates. Verify model/data tensors, forward passes,
finite gradients, optimizer changes and quantum query/key gradients; hold all
2,097,152 training rows and a best-checkpoint snapshot on CUDA during measurement.
Require peak reserved memory below 80% of the slice and projected total optimizer
time below five hours. Inspect a failed gate instead of automatically restarting
or changing the protocol.

Use one isolated checkout and submission. Pin all inputs and source snapshots
under `output/coverage-seeds/`. Run laptop `dgx start`, verify actual GPU startup,
then stop goal work while waiting as requested. Do not use codex-long-jobs or treat
an SSH/observer failure as permission to restart training. After notification,
verify accounting, all hashes, six complete histories and sampled-exposure traces,
selected validation arrays, tables, intervals, seed summaries, decisions and saved
sample renderings before claiming completion.

Local preflight passes all six parameter counts and all reference locks, reproducing
16 saved rows for each of eight quantum references and all 4,096 rows for each of
four MLP references. Three reporting tests verify mean-loss semantics and rejection
of hidden failed-seed/non-space outcomes. Ruff and strict Pyright pass for the
generated tools. Model/source training code is unchanged from the audited 149-test
DGX run; the complete CUDA suite will run again in the new allocation.
No new predictive result is available at declaration.

## Verified GPU startup

Slurm **11093** is RUNNING on prod10 with one `nvidia_a100_1g.10gb` slice in
`~/quantum-semantics-coverage-seeds-20261110`. The earlier interrupted transfer had
not submitted a Slurm job: the local submission handle was gone, accounting showed
no seed-study job and 98 pinned files were still missing remotely. Transfer resumed
into that checkout, all 310 pinned inputs verified, and the study was submitted
once. No training fit was restarted and the frozen scientific inputs were unchanged.

All **149 project tests**, **three reporting tests** and **six actual CUDA
learning/resource gates** passed. Peak reserved memory is at most **2.578 GiB**
with all 2,097,152 training rows resident. Measured update times project about
**3h40m of optimizer work**, below the five-hour guard; setup, validation and
reporting add time. The first `quantum_s22` fit has started, with run metadata
confirming CUDA, the declared configuration and unchanged dataset/source hashes.
All 310 inputs remain intact.

The laptop `dgx start` notifier is active/running (observed PID 79521). Startup
accounting, logs and resource measurements are retained under
`output/coverage-seeds/startup/`. Goal work pauses during the experiment wait as
requested; no predictive result is yet available. Inspect accounting and use the
prepared retrieval/audit script after notification rather than restarting on an
observer or connection failure.
