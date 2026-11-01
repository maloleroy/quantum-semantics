# Direct one-shot objective: declared manual diagnostic

1 November 2026. Prepared, **not trained**. The [readout screen](product-readout.md#completed-results)
adds no useful prediction benefit. Return to unrotated rank-128 product and test
the explicitly declared one-shot objective rather than introducing another model.

## Question and limits

So far these byte-token fits trained and selected by NLL while reporting expected
one-shot success S = mean p(correct token). This screen trains by `--loss success`
and selects by `--selection success` (existing NLL tie-break). It asks how much
of the sampling deficit can be changed by that combined objective/selection recipe.
It does not isolate loss from checkpoint selection as separate factors.

S is not a proper distribution-learning objective: at a fixed context, expected
success is linear in the predicted probabilities and is maximized by putting all
mass on a most likely target. Consequently better S can coexist with much worse
NLL, reduced diversity and repetitive text. This diagnostic must not redefine the
research goal as score improvement alone. A candidate that collapses to dominant
outcomes is not a successful language generator, even if it beats an NLL-trained
model on S. NLL, target-frequency buckets, argmax and unfiltered samples remain
visible; no generation model is promoted solely for higher S.

A useful context-free reference is missing from the earlier sampling tables:
always predict the most common **training** target. In this fixed byte task it is
ID 33, the space byte. It occurs in 1,231/4,096 validation targets, giving **30.05%**
one-shot success/argmax with no context, above the current quantum 23.20%. This
is distinct from sampling from the unigram distribution, whose success is 9.30%.
The modal baseline is chosen from all-training transition counts, with lowest-ID
argmax tie-breaking; it is not selected on validation. It is also a degenerate
text generator, illustrating why S alone cannot establish the broader goal.

The baseline assigns exactly zero probability to non-modal targets. Its true NLL
and target bits/byte are therefore infinite; report `inf` in CSV rather than the
finite numerical floor used for ordinary model scores. Do not bootstrap infinite
NLL differences. Sampling/argmax paired differences remain well-defined. Model NLL
still uses the existing probability floor of 1e-30; if a fitted distribution develops
exact zeros, that clipped metric understates its mathematical cross-entropy.

## Three fits and fixed references

| Label | Model | Trainable parameters | Objective/selection |
|---|---|---:|---|
| success | Product, rank 128, Born readout | 555,284 | success / success |
| mlp | Matched product-feature MLP | 554,138 | success / success |
| frozen | Product with frozen Hamiltonians | 30,740 | success / success |

The frozen core remains stored and executed. No rotated readout or new classical
vocabulary head is added. The frozen-Hamiltonian comparison asks whether training
the dynamics contributes under the new objective. Different trainable budgets
limit its interpretation; nonzero gradients or parameter placement are not proof
of a quantum advantage.

Reuse the completed NLL-trained, NLL-selected rank-128 product and MLP predictions
from `output/product-capacity/batch-Z0fDku`, with their matching `a5a4523` archived
source, selected weights, datasets and predictions pinned. Their scores are
2.9281 NLL / 23.20% sampling and 2.8303 / 31.04%, respectively. Archived source
hashes are verified rather than bypassing current-source loading checks.

All new fits start fresh from seed 11 and use the exact expanded full dataset:
131,072 training targets, 4,096 validation targets, 128-token contexts, original
training-only byte tokenizer, article groups, UNK policy and count statistics.
Two rank-128 Hamiltonian blocks, one auxiliary qubit, recent window 16 and phase
scale 1 are unchanged. Adam LR .003, clipping 5, batch 512 and 30 epochs remain
fixed: 7,680 updates and 31 candidate checkpoints including epoch zero. Use
float64/complex128 and two Torch threads per worker (six requested threads).
No LR/seed sweep, optimizer continuation, tokenization refit or test scoring.

This is a single-seed adaptive validation diagnostic. Changing the objective now
is exploratory, not retroactive re-selection of earlier checkpoints. It does not
establish reproducibility or held-out advantage.

## Reports and checks

Re-evaluate all three new locked validation predictions and compare with saved
arrays/metrics at 1e-12. Include eight rows in `validation.csv`: three new fits,
two frozen model references, bigram, unigram and the deterministic modal baseline.
The baseline's token ID is explicit and its arrays are saved in NPZ.

`target-buckets.csv` separates modal-target and all-other-target results for every
model, using the training-defined modal ID. This exposes improvements that mostly
reflect more confident space prediction. It is descriptive, not a new selection
criterion. Counts, S, argmax, NLL and target bits/byte stay separate.

`paired.csv` has 23 rows: success-trained quantum minus seven controls for
NLL/S/argmax, excluding the modal baseline's undefined NLL difference, plus the
success-trained MLP versus its NLL-trained counterpart. Verify paired targets,
article groups and target byte lengths before computing descriptive article-bootstrap
intervals. These omit seed/checkpoint/model-selection uncertainty.

Three new models and two count references generate the same three fixed prompts
and seeds, 64 direct draws each (15 samples). Save every token ID in NPZ and its
rendering in CSV. No retries, filtering, temperature transformation or replacement
of repetitive samples. The deterministic baseline's output is trivially repeated
spaces and needs no random generation; historical samples stay in their old runs.

No model source changed. Generated-tool Ruff/strict Pyright, Bash syntax, imports,
archived-reference preflight and input checksums pass. The modal-token identity and
validation frequency were checked without fitting. The training/summary pipeline
is **unrun**; no research training was launched or monitored by the assistant.

## Manual command

```bash
bash output/product-success/launch.sh
```

The launcher uses the local
[codex-long-jobs skill](/home/malo/.codex/skills/codex-long-jobs/SKILL.md), with
no viewer and event-only delivery to preserve manual handoff. It prints job and
log/state paths plus log-follow instructions. Return after completion for analysis.
Allow roughly **20–60 minutes**, an unmeasured estimate based on the 20.7-minute
unrotated two-thread product fit. This is neither a measured duration nor an upper
bound; prior estimates have been imperfect.

Input checksums, a runner lock and a live-worker guard protect execution. Resume
missing fits from initialization with the printed batch path:

```bash
bash output/product-success/launch.sh output/product-success/batch-XXXXXX
```

Completed fits are verified and retained. `SUCCESS_SCREEN_COMPLETE` is emitted only
after all fits, lock checks and summaries succeed. Requested cancellation uses the
controller for the complete process group. Meaningful generation, useful long
context and a quantum advantage remain unproven.
