# Training with randomly hidden older context

## Declared protocol — 10 November 2026

The three-stage content model improves on matched width across three seeds, yet
full context has worse validation NLL than independently refitted 16-token models
at every seed. Its selected training NLL is better. This motivates a bounded
training regularization test, without establishing a diagnosis of overfitting.

Train the unchanged three-stage rank-128 `prior_content` model at seed 11 with
older-context dropout probabilities **0.25 and 0.50**. At each optimizer step,
independently for each example, replace all positions before the latest 16 tokens
with padding (-1) at the declared probability. Keep sequence positions, targets,
latest tokens, train-only BPE vocabulary and count statistics unchanged. There is
no inverted-dropout scaling. Use a dedicated device generator seeded 11, separate
from model initialization and the NumPy permutation generator. Resident data are
never modified. Histories record how many contexts actually changed each epoch;
rows whose older branch was already empty are excluded from that count.

This trains a single set of quantum weights on a mixture of full and short inputs.
It tests whether occasionally removing older information improves intact-context
prediction. It does not isolate a general regularization mechanism or prove that
long context is useful merely by improving over the unregularized full model.
The short reference needs no analogous older-branch dropout: all those inputs are
already padding, so this intervention would have no effect. A positive screen
still requires seed replication and independent-partition confirmation.

Both models retain **1,073,900 parameters**, count-prior state preparation, three
Hamiltonian stages, context-controlled spectral shifts/interleaved RZ gates and
direct Born token readout. No larger classical controller or vocabulary decoder
is added. Default dropout is zero and inference models are unchanged. Historical
reference source trees and metadata remain frozen; the new trainer's additional
zero-default field is handled explicitly when checking old training records.

## Fixed data, optimization and comparisons

Use the existing WikiText byte-BPE dataset: vocabulary 512, context 128,
524,288 training rows, 4,096 validation targets and 8,192 excluded test targets.
Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
Article partitions and all statistics remain unchanged. No test scoring occurs.
Validation articles have already informed prior development; this is an adaptive
screen, not an independent confirmation study.

Keep 30 epochs, batch 512, **30,720 Adam updates**, constant learning rate .003,
gradient norm clipping 5, float64/complex128 and NLL loss. Select among epoch 0
and all 30 completed epochs by intact full-context validation NLL, with sampling
success as tiebreaker and the earliest epoch on an exact tie. Online training loss
is measured on the mixed inputs; selected training NLL is measured on intact data.
Do not compare those as if they were the same quantity.

Pin the archived seed-11 three-stage full model (NLL **2.5401515**), independently
refitted short model (**2.5225834**) and strongest cosine MLP (**2.4077489**).
The MLP retains its historical schedule; it is a strong performance reference,
not a freshly matched dropout control. Report train-only bigram, unigram and modal
baselines too. Verify all model/data/weight hashes and reproduce bounded archived
predictions before launch. Quantum model/inference source files must remain identical to the archived versions;
only training and CLI code change. The older MLP predates the additional prior-model
families in shared modules; reproduce all 4,096 saved MLP validation scores before
launch, retaining its original source hashes rather than rewriting its metadata.

Report each dropout rate separately, including NLL, expected one-draw correct-token
probability S, argmax, non-space/dominant-target buckets, selected training NLL,
parameter counts, timings and fixed-prompt unfiltered generation. Generate 64 direct
draws for the three established prompts and seeds 20261030–32 for both candidates,
bigram and unigram. Preserve invalid UTF-8, control characters and whitespace.
Validation and generation never apply dropout; no probability temperature or
rejection procedure is introduced.

Select the rate with lower validation NLL, then higher S, then `drop25` on an
exact tie. An improvement over the full reference requires lower NLL; separately
report whether its paired article interval lies below zero. A preliminary
older-context result requires **all three**: lower NLL than the full reference,
a full-minus-refitted-short NLL article interval entirely below zero, and lower
non-space NLL than short. Use 2,000 article bootstrap replicates, seed 161803.
Report all rate/control intervals, not just the selected winner. Intervals are
conditional on this reused validation sample, without adjustment for selecting
among two rates or earlier experiments. They cannot establish independent success,
coherent generation or quantum advantage. The MLP gap remains explicit.

## Execution and completion evidence

Expected fit time is roughly **65–90 minutes** total from the two prior full-model
fits; mixed short/full batches may change throughput. This exceeds the laptop
budget. Use one **prod10 `nvidia_a100_1g.10gb`** slice, four CPU support threads and
a three-hour allocation. Keep scientific budgets unchanged. Before either fit,
run the full test suite and a GPU gate for each rate: actual CUDA inputs, forward
passes, finite gradients, query/key gradients, weight/prediction changes and
correct masking. Retain full resident data and best-checkpoint storage during
measurement. Require peak reserved memory below 80% of the slice and the projected
combined optimizer time below two hours. A failed gate stops before fitting and
must be inspected; it does not authorize automatic escalation or restart.

Tools and source snapshots live under `output/context-dropout/`; inputs are pinned
in `inputs.sha256`. Submit once with `submit.sh`, then run the laptop `dgx start`
notifier. Verify actual GPU learning and first-fit startup, then stop goal work
while waiting as requested. Do not use codex-long-jobs. After notification,
`retrieve.sh` checks accounting, input and artifact hashes, full histories,
selection locks, stored arrays and independently recomputed reports/sample decoding.
A connection or observer failure alone does not authorize restarting training.

No new experiment result is available at declaration.

Local preparation passed: 103 CPU tests (43 CUDA checks deferred to DGX), two
reporting-decision tests, Ruff formatting/linting and strict Pyright for project
code and generated tools. Endpoint masking equals an independently masked training
dataset on the tiny fixture, stochastic fits repeat exactly, resident data remain
unchanged and selected validation is checked on intact inputs. Preflight verified
both parameter counts, 16 saved predictions from each quantum reference and all
4,096 saved MLP predictions. These are implementation/input checks, not predictive
results for the new recipes.
