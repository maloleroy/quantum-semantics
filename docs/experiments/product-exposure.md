# Product-state training exposure: declared manual comparison

30 October 2026. Completed and independently audited; see the
[results](#completed-results). The original declared protocol is retained below. This follows the audited
[product-state screen](product.md#completed-results), whose 128-token predictor
reached 3.1975 validation NLL and 21.79% sampling success. It has a small
single-seed NLL lead over the matched MLP, but trails its sampling score and
trails the all-training bigram on NLL/sampling. Samples remain fragmented.

## Completed results

All six fits completed in `output/product-exposure/batch-MDwDbm`. The main job
printed its completion marker after summary/verification; a second launch was
rejected by the live runner lock and did not duplicate fits. Quantum fits took
21.6–28.0 minutes; MLP fits took 8.6–8.9 minutes.

Independent evaluation reproduced all six locked validation prediction arrays,
selection metrics and table metrics to 1e-12. The nested original training sample,
unchanged validation/test arrays, paired full/short contexts, original source/data
hashes, old product reference, all 30 paired intervals and all 24 saved sample
renderings were checked. CSV sample reading explicitly preserves carriage returns;
otherwise universal newline conversion changes generated control characters.
No resampling, filtering or held-out test evaluation occurred.

| Model | Targets × epochs | NLL | Sampling | Argmax | Target bits/byte |
|---|---:|---:|---:|---:|---:|
| Product, 128 tokens | 131,072 × 30 | 2.9673 | 23.17% | 36.77% | 2.6710 |
| Product, 16 tokens | 131,072 × 30 | 2.9667 | 23.11% | 36.50% | 2.6704 |
| Matched MLP | 131,072 × 30 | 2.7966 | 30.84% | 41.06% | 2.5172 |
| Product, repeated small set | 32,768 × 120 | 3.1793 | 21.91% | 35.72% | 2.8618 |
| MLP, repeated small set | 32,768 × 120 | 3.2306 | 26.48% | 35.82% | 2.9080 |
| Frozen Hamiltonians | 131,072 × 30 | 4.2567 | 18.00% | 31.20% | 3.8316 |
| Original product reference | 32,768 × 30 | 3.1975 | 21.79% | 35.28% | 2.8782 |
| All-training bigram | counts | 3.0593 | 23.16% | 34.03% | 2.7537 |

Broader training exposure improves product NLL by .2120 versus the equal-update
repeated-set control, interval [−.2359, −.1853], and sampling by 1.263 percentage
points, interval [1.048, 1.464]. Merely repeating the original set selected epoch
54 rather than 120 and improved little over its original 30-epoch score. The
expanded product selected epoch 29, with train NLL 2.8082 versus validation 2.9673.

The MLP benefits more from expanded exposure: NLL improvement .4341 and sampling
+4.361 points over its repeated-set control. The earlier small-data product NLL
lead does **not** carry over to the expanded matched comparison. Product is now
worse by .1708 NLL, interval [.1167, .2133], 7.672 sampling points, interval
[7.116, 8.217] in deficit magnitude, and 4.297 argmax points. The frozen-core
comparison still supports a contribution from trained Hamiltonians within this
recipe; it does not establish quantum superiority or entanglement-specific benefit.

Product beats all-training bigram NLL by .0919, interval [−.1155, −.0672], and
argmax by 2.734 points. Sampling differs by only +.014 points, interval
[−.265, +.286], so there is no demonstrated sampling advantage over bigram.
The 128-token versus 16-token NLL difference is +.00061, interval
[−.00981, +.01090]. Its sampling gain is only .060 points, interval [.008, .116].
That small single-seed gain does not establish useful long-context generation.
All intervals are conditional article bootstrap intervals on reused validation;
seed and selection uncertainty remain excluded.

Fixed-prompt output still consists of fragments. The learned model's three draws
contain zero UNKs but three UTF-8 replacement characters and two other control
characters. Expanded MLP draws contain two replacement characters. The learned
model's clean three-sample result in the preceding screen was not a general
valid-text guarantee. Frozen-core outputs contain five UNKs, six replacement
characters and 37 other controls. Every outcome remains included.

Evidence: [metric table](product-exposure-validation.csv),
[paired intervals](product-exposure-paired.csv),
[sample diagnostics](product-exposure-sample-diagnostics.csv).
The next hypothesis concerns token-order representation: the current preparation
compresses each context into pooled local angles. Weighted pooling already retains
some positional information, so this is not a claim that it is order-invariant.
A sequential noncommuting local-rotation preparation can test a different encoding,
with a retrained pooled counterpart and matched classical control. More data alone
has not yielded meaningful prose or a quantum advantage.

## Question and data

Does broader gradient-training exposure improve the product model more than
repeating the current small training set for the same optimizer budget?
The existing gradient set contains 32,768 of 6,859,088 eligible training
transitions. Count baselines already use all of them. This experiment increases
the gradient set to 131,072 targets (about 1.91% of eligible transitions), without
changing the model or refitting its tokenizer.

The preparation script loads the original 512-outcome byte vocabulary, merge
rules, count statistics and 128-token dataset. It re-encodes training articles
with those exact merge rules and checks their byte-for-byte round trip. It
reconstructs the original seed-1618 training-position selection and verifies
every original context, target, group and byte length. It adds 98,304 distinct
training positions, sampled without replacement from the remainder with seed
2718. The expanded set therefore contains the complete original training set.
Source-file fingerprints, all-training counts and smoothed unigram statistics
must match the original preparation before training can begin.

Validation and test arrays are copied unchanged. Official article partitions,
causal boundary padding, normalization-based group identities, target byte
lengths, the reserved UNK outcome and all-training statistics are retained.
A paired short dataset masks every context column except the final 16; its
weights are independently trained. No test evaluation is included.

## Fits and selection

All fits use seed 11, batch size 512, Adam LR .003, clipping 5, two rank-64
Hamiltonian blocks, one auxiliary qubit, recent window 16, phase scale 1,
float64/complex128 and one Torch thread. NLL is both the training objective and
validation checkpoint criterion, with sampling success as its existing tie-break.
Initial epoch-zero checkpoints remain eligible. All fits start from initialization;
this is not continuation from selected weights or optimizer resume.

| Label | Model | Training targets | Epochs | Optimizer updates |
|---|---|---:|---:|---:|
| learned | Product, 128-token context | 131,072 | 30 | 7,680 |
| short | Product, 16-token context | 131,072 | 30 | 7,680 |
| frozen | Product, frozen Hamiltonians | 131,072 | 30 | 7,680 |
| mlp | Matched product-feature MLP | 131,072 | 30 | 7,680 |
| repeat | Product, original 128-token set | 32,768 | 120 | 7,680 |
| repeat_mlp | Matched MLP, original set | 32,768 | 120 | 7,680 |

The main exposure comparison is `learned` versus `repeat`. The expanded matched
MLP tests whether any gain requires quantum dynamics rather than a similarly
parameterized classical replacement. `repeat_mlp` supplies the corresponding
classical exposure comparison; paired intervals also compare the two MLP
exposures and the two original-set equal-update architectures. The short and frozen-core controls
check older-context benefit and dependence on trained Hamiltonians at the new
exposure. The existing fixed-controller result is not extrapolated to the larger
set. Parameter budgets remain 293,012 for learned product, 292,695 for matched
MLP and 30,740 trainable parameters for the frozen-core control.

Equal updates means equal gradient-example presentations (3,932,160), not equal
wall time, data diversity or selection opportunities. Current training evaluates
validation once per epoch, so the repeated-set fits have 121 candidate checkpoints
versus 31 for expanded-set fits, including epoch zero. The original 30-epoch product
fit is reused as a locked reference (`small30`), not an equal-budget control.
Every reference file is pinned and current-source/selected-weight locks are checked.

This remains a single-seed adaptive validation screen, not independent confirmation
of the small MLP NLL difference or a quantum advantage. No new validation partition
is claimed. More data is a testable hypothesis, not a known remedy for deficient
context representation or generation.

## Reports and verification

The runner recomputes all six selected validation predictions and compares them
with saved arrays/selection metrics to 1e-12. It checks paired target/group/byte
arrays, includes unchanged all-training count baselines, and produces:

- `validation.csv`: six new fits, the original product reference, bigram and unigram.
- `paired.csv`: learned-minus-eight-controls differences and article-group intervals
  for NLL, sampling success and argmax, plus expanded-MLP versus repeat-MLP and
  repeat-product versus repeat-MLP (30 rows total). These omit model/seed-selection
  uncertainty.
- `samples.csv` and per-model NPZ token IDs: the same three prompts and seeds as the
  product screen, each with 64 direct draws, no rejection, retries or temperature.
- Per-fit weights, selection locks, histories and saved validation predictions.

Sampled-target bits per byte retains its original ratio-of-total-NLL-to-total-target-
bytes definition; it is not full-corpus perplexity or raw-string marginal likelihood.
Samples are qualitative diagnostics, not a text-quality benchmark. A small NLL gain
with worse sampling or fragmented generation does not satisfy the overall goal.

Generated tools live under `output/product-exposure/`. Bash syntax, module imports,
Ruff, strict Pyright on the generated Python tools and independent small checks
of nested sampling and causal article-boundary construction pass. Source code is
unchanged. At handoff, full preparation and training were unrun. The user subsequently
completed both, and the independent audit above passed their integrity checks.

## Manual command

From the project directory:

```bash
bash output/product-exposure/launch.sh
```

The launcher uses the local
[codex-long-jobs skill](/home/malo/.codex/skills/codex-long-jobs/SKILL.md)
with an event-only completion record and no viewer, preserving the user's manual
handoff. It prints the job name, log path, state path and log-follow command.
It detaches execution; return to the conversation after completion to audit results.
No training has been launched by the assistant. The launcher invokes the controller
through Bash because its installed wrapper does not have executable permission.

Allow approximately **25–35 minutes** for six concurrent one-thread fits plus
preparation/audit. This estimate extrapolates the prior 6.1-minute quantum fits
at four times the optimizer updates; it is not a measured runtime for this screen.
The six fits use separate cores on the current 16-logical-CPU laptop. Larger dataset
arrays add memory costs; no near-20-qubit feasibility or hardware speedup is claimed.

Input checksums, a runner lock and live-worker guard prevent accidental source
changes or duplicate screen execution. Failed runs retain completed selections.
To resume missing fits from initialization, use the printed batch path:

```bash
bash output/product-exposure/launch.sh output/product-exposure/batch-XXXXXX
```

Completed fits are verified and preserved. A completed prepare step is checked and
reused; an interrupted preparation is rebuilt before any fit starts. The final
`EXPOSURE_SCREEN_COMPLETE` marker is printed only after every fit, lock check and
summary succeeds. Cancellation, if requested, must use the job controller so the
whole process group is stopped.
