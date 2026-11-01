# Broader product-model training coverage

1 November 2026. Prepared for manual execution. No new data or fits have yet been
produced. This follows the negative [streaming result](stream.md#completed-results)
and retains the unrotated product model. The earlier increase from 32,768 to
131,072 training targets improved NLL beyond equal-update repetition, although
it improved the MLP more. Test a further coverage increase without assuming a
quantum advantage or meaningful long-context generation.

## Data and matched exposure

The locked WikiText training partition contains 6,859,088 eligible transitions.
Current gradient training uses 131,072 (1.91%). Add 393,216 distinct positions,
sampled from the remainder with seed 3141, for **524,288 unique training positions
(7.64%)**. Read the exact saved original positions from
`output/product-exposure/batch-MDwDbm/data/positions.npz`; every old position is
retained. Do not refit the training-only 512-outcome byte BPE or count statistics.

Preparation re-encodes the source training articles, checks byte-for-byte decoding,
reconstructs every old context/target/group/byte length, and reproduces the stored
all-training counts and smoothed unigram. All validation and test arrays remain
unchanged. Contexts are causal, padded at article boundaries and 128 tokens wide.
Normalized article groups retain the original split policy. No test is scored.

For the repeated-set control, concatenate four exact copies of the existing
131,072-row training partition. Both new NPZ datasets thus have 524,288 rows,
but only the expanded dataset has additional unique source positions. Repetition
is explicit optimizer exposure, not additional independent data. Tokenizer and
count statistics remain identical in both datasets; counts are never inflated
from duplicated rows.

Each fit runs 30 epochs of 1,024 batches (batch size 512): **30,720 optimizer
updates and 15,728,640 row exposures**. Both datasets are shuffled each epoch by
the existing seed-11 training routine. Checkpoint selection evaluates the same
validation set at epoch zero and after each of the 30 epochs, giving equal
selection opportunities at equal update counts. This differs from the earlier
120-small-epoch control, which had more checkpoint opportunities. The repeated
rows may occur in the same minibatch; this is a declared repeated-example recipe,
not an exact replay of four independent small-epoch shuffles.

## Four new fits and references

| Fit | Model | Unique source positions | Trainable parameters |
|---|---|---:|---:|
| quantum | Unrotated product | 524,288 | 555,284 |
| repeat | Unrotated product | 131,072, repeated fourfold | 555,284 |
| mlp | Matched product MLP | 524,288 | 554,138 |
| repeat_mlp | Matched product MLP | 131,072, repeated fourfold | 554,138 |

All use two rank-128 joint Hamiltonian stages (or the matched MLP budget), auxiliary
size 2, recent window 16, phase scale 1, Adam learning rate .003, seed 11, NLL loss
and NLL validation selection. Amplitude decay .9 is recorded but unused by these
product models. The quantum path remains local product-state preparation,
Hamiltonian evolution with context-dependent RZ interleaving, then direct token
measurement. There is no classical word decoder or sampling-temperature change.
Models and optimizer states start fresh; selected weights are not training resumes.

Retain the four audited NLL references: product ranks 64/128 and their matched
MLPs. Their data, weights, archived source and saved prediction arrays are pinned;
the arrays were independently reproduced earlier. The rank-64 MLP's 2.7966 NLL
remains a stronger reference than the rank-128 MLP's 2.8303. Compare the new quantum
candidate with both, as well as its newly retrained matched MLP. A gap to either
classical option remains a gap; no claim of superiority follows from beating only
one control. Frozen-Hamiltonian attribution was examined in prior screens and is
not repeated here; this screen isolates coverage within retained recipes.

Report 11 metric rows, 22 space/non-space target buckets and 32 paired interval
rows. Comparisons are expanded quantum versus the ten other models/baselines,
plus expanded MLP versus repeated MLP, using NLL, sampling success and argmax.
Skip infinite-NLL paired inference against the train-modal predictor. Count controls
retain all-training statistics. Group-bootstrap intervals describe this single-seed,
reused-validation screen and exclude training/selection uncertainty.

Save 18 fixed-prompt 64-token continuations: all four new fits plus bigram/unigram,
using prompts and seeds from the preceding screens. Save raw tokens in NPZ and text
in CSV; include every draw, invalid byte sequence and UNK. No reference resampling
or test scoring occurs. NLL, non-space prediction and generated text all matter.
Coverage improvement alone cannot establish useful long-context modeling or a
hardware quantum advantage. Any promising result will still require confirmation.

## Manual execution and checks

From the repository root:

```bash
bash output/product-coverage/launch.sh
```

Planning estimate: **1–3 hours**, unmeasured for this larger dataset. Four fits run
concurrently with two Torch threads each on the eight-physical-core Ryzen 7 7840S.
The launcher uses the codex-long-jobs controller with event-only delivery and no
viewer. It first prepares/verifies the new datasets, then trains, verifies locked
predictions, reports comparisons and generates samples. Training is not launched
by the agent. Reported fit times will replace this estimate after completion.

Source, dependencies, scripts, raw source files, existing datasets and archived
references are checksummed. Newly prepared data receive a separate checksum file
before fitting, checked again afterward. Completed fits are preserved on retry:

```bash
bash output/product-coverage/launch.sh output/product-coverage/batch-XXXXXX
```

The final marker is `COVERAGE_SCREEN_COMPLETE`. Generated tools, datasets and fits
live under ignored `output/product-coverage/`. Shell syntax, tool imports, Ruff,
strict Pyright, reference/source checks and small independent tests of nested
position selection, causal boundaries, byte lengths and fourfold repetition pass.
The full new dataset will be checked during manual preparation; it does not yet
exist. No model or training implementation changed for this protocol.
