# Broader training coverage at a fixed update budget

## Declared protocol — 10 November 2026

The replicated three-stage quantum model improves on two-stage and matched-width
models, but loses to its independently refitted short-context version and the
strong MLP. Older-branch masking produced only a weak single-seed signal. Earlier
coverage increases improved a less capable quantum model, but helped its MLP more.
The current gradient pool still contains only **524,288 of 6,859,088 eligible
training positions (7.64%)**. Test broader gradient coverage for the stronger
model before attributing the remaining deficit solely to architecture.

This is a data-coverage recipe comparison at fixed optimizer exposure. It is not
an isolated quantum architectural change, and it does not assume that more data
will close the classical gap. No larger classical encoder/decoder is introduced.
The quantum path remains count-prior amplitudes, three learned rank-128 Hamiltonian
stages with contextual spectrum/RZ control, and direct Born token measurement.

## Data and fixed logical epochs

Expand the training pool to **2,097,152 unique positions (30.57%)**. Retain every
previous position and uniformly sample the remaining 1,572,864 without replacement
from eligible training positions, using seed **20261110**. Keep sorted source
positions in NPZ. Re-encode the existing source training articles with the exact
stored byte-BPE model, verify byte round trips, reconstruct all old training rows,
and reproduce the all-training transitions/unigram statistics. No tokenizer/count
refit, new corpus, split change or held-out scoring occurs.

The vocabulary remains 512, context width 128, validation 4,096 targets and excluded
test 8,192 targets. Every validation/test array remains identical to the old full
dataset. The short dataset masks only the first 112 context positions in every
partition; targets, groups, byte lengths and last 16 tokens remain identical.
Official article groups remain disjoint. Validation has already informed prior
experiments; these results cannot be independent confirmation.

Full dataset SHA256:
`37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`.
Short dataset SHA256:
`1730223fedbf3dc864398053eca442d67466bd3f042bfee4fbd5395e57f209bb`.
The old full dataset retains SHA256
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.

Use **30 logical epochs**, each comprising the first **524,288 rows** of a fresh
NumPy permutation of the entire 2,097,152-row pool. Seed the sampler at 11 once and
advance it between epochs. Sampling is without replacement within an epoch, with
revisits permitted between epochs. All three fits share the exact sampled row
schedule. Batch size stays 512: **30,720 updates and 15,728,640 row exposures** per
fit, exactly the archived references' budgets. Epoch zero and each logical-epoch
endpoint give the same 31 validation-selection opportunities at the same update
counts. The new `--epoch-examples` option defaults to all rows, preserving historical
permutation order and full-epoch behavior when unset.

At the end of the declared 30-epoch schedule, **2,096,779 distinct pool rows** have
been visited. The selected checkpoint can occur earlier: report its cumulative
unique exposure separately, as well as the final schedule total. Histories record
actual example count and cumulative distinct row count per logical epoch; the audit
reconstructs these independently from NumPy permutations. Retain final partial
batches in the general API, although this experiment's batch size divides exactly.

This changes coverage/repetition and row ordering together. It does not isolate
one abstract regularization mechanism. The archived controls repeatedly visit the
old pool; they already have exactly matching update budgets and selection cadence,
so no redundant new repeated-pool fit is required. Their source, selected weights
and saved predictions remain pinned. Additional training-pool rows are candidates
for gradients, not additional independent validation observations.

## Fits, references and decisions

| New fit | Recipe | Parameters | Learning-rate schedule |
|---|---|---:|---|
| quantum | `prior_content`, three layers, rank 128, full context | 1,073,900 | Constant .003 |
| short | Same quantum model, only latest 16 tokens | 1,073,900 | Constant .003 |
| mlp | Strong `adaptive_mlp`, two layers, rank 128 | 750,460 | Cosine .003 to .0003 |

All use seed 11, one auxiliary qubit, recent window 16, phase scale 1, Adam,
clipping norm 5, float64/complex128, NLL training and NLL validation selection.
Sampling S breaks NLL ties, then the earliest checkpoint wins. Dropout stays zero.
The MLP retains amplitude decay .9; quantum retains its recorded .7 (count-prior
preparation does not use amplitude decay). These reproduce each model's retained
recipe. The MLP is a strong equally exposed competitor, not an exactly matched
parameter-count or optimizer-schedule control.

Pinned seed-11 references from the old pool have NLL **2.5401515** full quantum,
**2.5225834** short quantum and **2.4077489** MLP. Before launch, verify their source,
data and weight hashes and reproduce bounded quantum predictions and all 4,096
MLP validation predictions. Inference code is unchanged from the quantum reference;
only training/CLI support for logical epochs changes. The older MLP's original
source is preserved despite added unrelated prior families in current modules.

Report each fit, its corresponding old-pool reference, and train-only bigram,
unigram and modal baselines: NLL, expected one-shot correct-token probability S,
argmax, non-space/dominant-target buckets, parameter counts, training-pool NLL,
selected unique exposure and fit time. Training-pool NLL now covers 2,097,152 rows,
including rows not yet seen by the selected checkpoint. It is **not** directly
comparable to the old 524,288-row training NLL as an overfitting diagnostic.
Validation targets stay identical and remain directly paired.

A coverage candidate passes only if expanded-minus-old full quantum NLL has an
article-bootstrap interval wholly below zero and non-space NLL also improves.
Useful older context requires the same interval/non-space conditions against the
new equally exposed short refit. Keep the new MLP gap explicit; merely beating an
older less-exposed MLP would not establish quantum advantage. Separately report
MLP versus old MLP and short versus old short to show their coverage response.
Use 2,000 paired article bootstrap replicates, seed 161803. The 29 intervals are
conditional single-seed descriptions on reused validation, without accounting for
prior adaptive model selection. Any gain still needs replication and independent
confirmation.

Generate 64 direct draws for each of the three established prompts with seeds
20261030–32, for full quantum, new MLP, bigram and unigram (12 samples). Save token
IDs and raw renderings; preserve invalid bytes, controls and whitespace. Do not
apply dropout, temperature or rejection. Short-model generation remains omitted
through the generic full-window generation path. Valid bytes alone are insufficient
for meaningful long text; inspect coherence and retain adverse outcomes.

## Execution and audit

Data preparation and local tests fit the ten-minute laptop limit. Full fits exceed
it: prior full/short/MLP measurements suggest roughly **1.5–2.5 hours** combined,
with larger dataset loading and training-pool evaluation adding overhead. Use
one **prod10 `nvidia_a100_1g.10gb`** slice, four CPU support threads, and a four-hour
allocation. Keep batch/update budgets unchanged. No full-GPU escalation is assumed.

Before fitting, require the full CUDA test suite and a gate for all three recipes:
actual CUDA resident training data, model tensors, predictions, finite gradients
and optimizer changes; query/key gradients for both quantum fits; correct count
prior initialization for quantum; and representative update timings. Hold the
entire larger pool and best-checkpoint snapshot during memory measurement. Stop
before fitting if peak reserved memory reaches 80% of the slice or projected total
optimizer time exceeds three hours. Inspect any failure without automatically
restarting or changing the scientific budget.

All generated tools/data and a source snapshot live under `output/depth-coverage/`.
Pin inputs before one isolated submission. Run laptop `dgx start`, verify actual
GPU startup, then stop goal work while waiting as requested. Do not use
codex-long-jobs. On completion, check Slurm accounting, input/output hashes, source
and weight locks, complete histories and sampled exposure, GPU-reproduced selected
validation arrays, all tables/intervals/decisions and saved sample decoding.

Local setup passes **105 CPU tests** (44 CUDA checks deferred to DGX), two reporting
tests, Ruff formatting/linting and strict Pyright for source/tests/generated tools.
Tests preserve full-epoch RNG behavior, refresh sampled rows, reject oversized
budgets and independently reproduce a logical epoch including its final partial
batch. Preflight verifies nested data, unchanged held-out arrays/statistics, model
parameter counts, 16 archived rows per quantum reference and all 4,096 MLP rows.
No new predictive result is available at declaration.
