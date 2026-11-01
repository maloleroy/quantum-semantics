# Direct one-shot objective: declared manual diagnostic

1 November 2026. Completed and independently audited. See the
[completed results](#completed-results); the original diagnostic protocol follows. The [readout screen](product-readout.md#completed-results)
adds no useful prediction benefit. Return to unrotated rank-128 product and test
the explicitly declared one-shot objective rather than introducing another model.

## Completed results

All three fits in `output/product-success/batch-LfpKOE` completed and the job
printed its final marker. Independent evaluation reproduced every new locked
prediction/metric to 1e-12; the two archived NLL references, modal-baseline arrays,
16 target buckets, 23 paired interval rows and all 15 saved sample renderings were
checked. Infinite modal-baseline NLL was verified explicitly and not bootstrapped.
No new draws or held-out test scoring occurred.

| Model | Sampling success | Argmax | NLL | Fit minutes |
|---|---:|---:|---:|---:|
| Success-trained product | 34.36% | 34.79% | 8.0966 | 20.7 |
| Success-trained MLP | 30.05% | 30.05% | 21.1360 | 6.0 |
| Success-trained frozen Hamiltonians | 20.30% | 30.98% | 5.2461 | 15.2 |
| NLL-trained product reference | 23.20% | 37.11% | 2.9281 | 20.7 |
| NLL-trained MLP reference | 31.04% | 40.58% | 2.8303 | 7.3 |
| Training-modal space predictor | 30.05% | 30.05% | infinite | — |

Direct success training increases quantum sampling by 11.162 percentage points
versus its NLL-trained reference, interval [10.372, 11.960], while **worsening NLL
by 5.1685** and reducing argmax by 2.319 points. Quantum sampling exceeds the modal
baseline by 4.306 points, interval [3.706, 4.940], so it is not simply a constant
space predictor. The same success-training recipe drives the MLP into exactly
that failure mode: its three saved continuations are 64 spaces each, with one
unique token and almost-zero probability on other targets.

The quantum gains mostly favor common outcomes. On the 1,231 space targets it
assigns 98.96% correct probability, versus 66.34% for its NLL reference. On the
2,865 other targets it improves sampling from 4.66% to 6.60%, but reduces argmax
from 10.96% to 7.16% and worsens NLL from 3.9616 to 11.5560. The NLL-trained MLP
still gets 10.40% sampling and 17.63% argmax on these non-space targets.

Quantum continuations repeat forms of `the`, commas and periods. Each saved
64-token continuation has just five distinct token IDs, with spaces making up
exactly half of the draws. One continuation contains a control character. This
is **not meaningful generation** and must not replace the NLL-trained reference.
The direct-success model's advantage over the collapsed success-trained MLP is
recipe-specific. The already available NLL-trained MLP can return argmax at
40.58% accuracy, exceeding quantum one-shot success 34.36%; the result does not
establish a practical quantum advantage.

All fits selected epoch 30. The longest took 20.7 minutes, within the declared
20–60 minute planning range. Intervals remain conditional article-bootstrap
summaries of a single seed and reused validation, excluding selection uncertainty.
Source, byte coverage, groups, statistics and test discipline were preserved.

Evidence: [metrics](product-success-validation.csv),
[paired intervals](product-success-paired.csv),
[target buckets](product-success-target-buckets.csv),
[sample diagnostics](product-success-sample-diagnostics.csv).
For further generation experiments, return to the declared NLL training/selection
recipe and retain S, argmax, dominant-token and sample diagnostics. The broader
objective remains conditional language modeling, not optimization of S alone.
The next representation hypothesis should let quantum evolution process ordered
parts of the context, rather than only changing rank, readout or a score surrogate.

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
validation frequency were checked without fitting. At handoff the pipeline was unrun; the user subsequently completed it and the
independent audit above passed; no research training was launched or monitored by the assistant.

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
