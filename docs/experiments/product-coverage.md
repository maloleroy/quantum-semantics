# Broader product-model training coverage

1 November 2026. Completed and independently audited. The original protocol follows. This follows the negative [streaming result](stream.md#completed-results)
and retains the unrotated product model. The earlier increase from 32,768 to
131,072 training targets improved NLL beyond equal-update repetition, although
it improved the MLP more. Test a further coverage increase without assuming a
quantum advantage or meaningful long-context generation.

## Completed results

The controller reports success for `qcse-coverage-20261101T133331-163228`; all four
fits and the final completion marker are present in `batch-RofZOb`. Independent
evaluation reproduced every new locked prediction and metric to 1e-12. Dataset
hashes, nested positions, fourfold repeated rows, preserved tokenizer/counts and
validation/test partitions, four archived references, 11 metric rows, 32 paired
intervals, 22 target buckets and 18 saved sample renderings were checked.
No resampling or held-out test scoring occurred.

| Model | NLL | Sampling success | Argmax | Fit minutes |
|---|---:|---:|---:|---:|
| Expanded product | 2.7870 | 24.42% | 38.94% | 100.9 |
| Repeated-set product | 2.9077 | 23.79% | 38.26% | 100.4 |
| Expanded matched MLP | 2.4941 | 36.66% | 46.24% | 38.4 |
| Repeated-set matched MLP | 2.8344 | 31.60% | 41.38% | 38.2 |
| Previous rank-128 product | 2.9281 | 23.20% | 37.11% | 20.7 |
| Previous rank-64 MLP | 2.7966 | 30.84% | 41.06% | 8.6 |
| All-training bigram | 3.0593 | 23.16% | 34.03% | — |

Expanded coverage improves quantum NLL by .12079 over equal-update repetition,
interval [−.14300, −.09884], and sampling by .630 percentage points, interval
[.505, .747]. Repetition alone improves less over the previous 30-small-epoch
reference. Quantum NLL also beats bigram by .27230, interval [−.29744, −.25009],
and sampling by 1.264 points, interval [.975, 1.557].

The matched MLP benefits more: its NLL improves .34036 over repetition, interval
[−.39392, −.28732], and sampling rises 5.060 points. The new quantum model trails
this equally exposed MLP by **.29290 NLL**, interval [.22639, .34966], and **12.241
sampling points**, interval [11.471, 13.010] in deficit magnitude. A nominal .00961
NLL lead over the older, less-exposed rank-64 MLP has an interval spanning zero;
it does not establish superiority over a competitive classical fit.

On non-space targets, quantum sampling is 5.24%, accuracy 13.37% and NLL 3.7808,
versus MLP 16.83%, 25.31% and 3.4388. The quantum selected training NLL is 2.7315
versus validation 2.7870; its small gap suggests limited benefit from treating
sample overfitting as the only remaining problem, but does not prove a capacity
or optimization diagnosis. MLP selected training NLL is 2.2359.

Quantum samples remain fragmented, with one UTF-8 replacement character and four
other controls across three saved continuations, despite zero UNKs. MLP samples
have no invalid/control outcomes in this small sample, but remain semantically
fragmented. Neither produces meaningful long text. No fresh long-context benefit
was isolated by this screen.

Selections are quantum epoch 29, repeated quantum 26, MLP 27 and repeated MLP 9.
The longest fit took 100.9 minutes, within the 1–3-hour estimate. Intervals remain
single-seed, reused-validation descriptions, not confirmation of a quantum advantage.
Retain the expanded product/MLP as the new predictive references. Next test
context-dependent Hamiltonian eigenvalues against matched static capacity and
classical controls, rather than relying on additional data alone.

Evidence: [metrics](product-coverage-validation.csv),
[paired intervals](product-coverage-paired.csv),
[target buckets](product-coverage-target-buckets.csv),
[sample diagnostics](product-coverage-sample-diagnostics.csv).

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
