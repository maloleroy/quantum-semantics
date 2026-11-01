# Learned local token measurement basis: declared manual comparison

1 November 2026. Completed and independently audited. See the
[completed results](#completed-results); the original declared protocol follows.
The [capacity comparison](product-capacity.md#completed-results) improved product
NLL modestly at rank 128 but left sampling near 23.2%, far below the matched MLP.
This screen tests a small change in the quantum measurement basis instead of
increasing rank again.

## Completed results

Batch `output/product-readout/batch-IXmgFt` completed all four fits and printed its
final success marker. Independent evaluation reproduced all four locked validation
arrays and metrics to 1e-12, verified the four archived model references and input
checksums, and checked all 33 paired interval rows and 18 saved sample renderings.
No new sampling or held-out test scoring occurred.

| Model | NLL | Sampling success | Argmax | Target bits/byte | Fit minutes |
|---|---:|---:|---:|---:|---:|
| Rotated rank 128 | 2.92786 | 23.110% | 37.43% | 2.63542 | 96.2 |
| Unrotated rank 128 | 2.92811 | 23.198% | 37.11% | 2.63564 | 20.7 |
| Rotated rank 64 | 2.96683 | 23.180% | 36.65% | 2.67050 | 89.8 |
| Unrotated rank 64 | 2.96734 | 23.172% | 36.77% | 2.67096 | 27.7 |
| Fixed controller, rank 128 | 3.93490 | 11.938% | 30.05% | 3.54188 | 94.7 |
| Frozen Hamiltonians, rank 128 | 4.22220 | 14.482% | 33.23% | 3.80048 | 90.1 |
| Matched MLP, rank-128 budget | 2.83027 | 31.035% | 40.58% | 2.54758 | 7.3 |
| Best-NLL MLP reference | 2.79656 | 30.844% | 41.06% | 2.51724 | 8.6 |

At rank 128, rotated-minus-unrotated NLL is −.00025, interval
[−.00944, +.00978]. Sampling is slightly worse by .0877 percentage points,
interval [−.1482, −.0350]. Argmax improves by .3174 points with interval
[0, .6459], which does not establish an overall gain. Rank-64 NLL changes by
−.00051, interval [−.01024, +.01094], and sampling by +.0085 points, interval
[−.0431, +.0592]. The local measurement basis adds no clear prediction benefit.

The rotated rank-128 model still benefits from trained Hamiltonians compared with
the frozen-Hamiltonian control: −1.2943 NLL and +8.628 sampling points. But it
trails the matched MLP by .09758 NLL and 7.925 sampling points. These comparisons
remain single-seed adaptive validation evidence, with article-bootstrap intervals
conditional on selected weights rather than accounting for seed/model selection.
No quantum advantage or new long-context benefit is established.

All new fits selected epoch 30. Rank-128 training NLL is 2.71818, almost unchanged
from unrotated 2.71981. The learned rank-128 samples still contain fragments,
two UTF-8 replacement characters and one other control character, with zero UNKs
in these three draws. Frozen-Hamiltonian samples contain four UNKs, 25 replacement
characters and 29 other controls. Every outcome remains in the saved samples.

The longest fit took **96.2 minutes**, slightly above the 30–90 minute estimate.
It was much slower than the previous 20.7-minute unrotated fit, but these runs are
not a controlled runtime profile. Given the negligible prediction benefit, keep
unrotated product as the working reference rather than optimizing or scaling the
rotated variant on this evidence.

Evidence: [metrics](product-readout-validation.csv),
[paired intervals](product-readout-paired.csv),
[sample diagnostics](product-readout-sample-diagnostics.csv).
The next diagnostic separates NLL training from direct expected-success training,
with classical/frozen controls and a dominant-token baseline. Better one-shot
success alone must not be mistaken for better conditional language modeling.

## Physical change and hypothesis

`product --decoder rotated` adds one learned RX followed by RY on each of the nine
token qubits after the final Hamiltonian, immediately before direct measurement.
The auxiliary is still ignored; no auxiliary readout angles are allocated because
a local unitary on that discarded register leaves token probabilities unchanged.
There are **18 new real parameters** and 18 local physical gates, initialized to
zero so initial predictions match the unrotated model with the same seed exactly.

The preparation, interleaved context RZ gates, Hamiltonians and training objective
are unchanged. There is no classical vocabulary decoder, probability calibration,
postselection, temperature transform or extra sampling attempt. The final distribution
is the joint Born distribution in a learned local basis, with the auxiliary traced
out by summing its outcomes. This is not the separate classical marginal decoder.

A restricted spectral block acts as identity on its orthogonal complement. A local
measurement-basis rotation acts across the register, so it may provide useful
expressivity cheaply compared with another rank increase. This is a hypothesis,
not a proof of an independent bottleneck or a guaranteed gain. For unrestricted
unitary families a final unitary can be absorbed into the whole evolution; the
relevance here is the particular low-rank, interleaved parameterization. This
experiment does not demonstrate hardware speedup or isolate an entanglement benefit.

The implementation rotates the word dimension identically in every auxiliary
sector before computing its marginal. Default `--decoder born` allocates no readout
parameter and retains its prior state-dictionary structure. `product_fixed` also
supports rotated readout; sequence and product-MLP variants retain their existing
readout restrictions.

## Four independently trained fits

| Label | Model | Rank | Trainable parameters |
|---|---|---:|---:|
| rotated128 | Product with rotated readout | 128 | 555,302 |
| rotated64 | Product with rotated readout | 64 | 293,030 |
| fixed | Fixed context-controller tables/offsets, learned dynamics/readout | 128 | 524,562 |
| frozen | Frozen Hamiltonians, learned context controller/readout | 128 | 30,758 |

The rank-128 unrotated reference has 555,284 parameters, only 18 fewer.
Both rank-128 controls store all 555,302 parameters including frozen values.
`--freeze-core` here freezes **Hamiltonians**, while readout angles remain trainable.
Consequently that control tests the contribution of training Hamiltonians given a
learned preparation/readout; it does not freeze every quantum operation. The fixed
controller leaves preparation/interleaving tables and offsets at initialization
while training Hamiltonians and readout. These controls have different trainable
budgets and answer separate attribution questions.

Reused unrotated references: expanded product rank 64 (2.9673 NLL / 23.17% sampling),
product rank 128 (2.9281 / 23.20%) and both corresponding matched MLPs (2.7966 /
30.84%, and 2.8303 / 31.04%). There is no new MLP refit solely to add 18 parameters;
its already approximately matched budget differs by less than one percent. A small
quantum recipe improvement would not by itself establish superiority over those
controls or any broader classical family.

## Declared data, optimization and reports

Use the exact expanded `output/product-exposure/batch-MDwDbm/data/full.npz`:
131,072 training targets, 4,096 validation targets, 128-token contexts, the stored
training-only 512-outcome byte BPE, official article partitions, original count
statistics and target byte lengths. No tokenizer refit, changed UNK policy or test
scoring. No optimizer continuation from selected checkpoints.

All new fits start with seed 11 and use two Hamiltonian blocks, one auxiliary
qubit, recent window 16, phase scale 1, 30 epochs, batch 512, Adam LR .003, clipping
5, float64/complex128 and two Torch threads each. Four workers request eight threads.
NLL trains and selects weights, with the existing sampling tie-break; epoch zero
is eligible. Each fit has 7,680 optimizer updates and 31 validation checkpoints.
This is a single-seed adaptive validation comparison, not fresh confirmation.

Rank-128 references were audited before source edits in commit `a5a4523`, with
matching source archived from that commit. Rank-64 references retain their source
snapshot `b4527ba` and earlier verified prediction arrays. Preflight verifies each
snapshot against its recorded source hashes and checks original datasets/weight
locks. The manifest pins all reference arrays and metadata. New fits use the normal
current-source loader. Old checkpoints are not unlocked or re-scored under changed
source.

The summary independently reproduces all four new locked prediction arrays and
metrics to 1e-12. Ten metric rows cover four new models, four frozen model references
and the all-training bigram/unigram. Paired target/group/byte-length checks precede
33 article-bootstrap interval rows: rotated128 versus the other nine models for
NLL/sampling/argmax, plus rotated64 versus its unrotated and MLP references.
Intervals omit seed/selection uncertainty. Sampling success and argmax remain
separate; target bits per byte retains its sampled-target denominator.

The same three fixed prompts/seeds produce 64 direct draws each for four new
models and two count references (18 samples), stored as exact token IDs in NPZ
and rendered text in CSV. No retries, filtering or temperature changes. Historical
samples remain in their original batches. Text quality and invalid outcomes remain
part of assessment even if NLL improves; longer-context benefit is not established
by this screen.

## Verification and manual launch

All **60 tests**, Ruff and strict Pyright pass. New tests match probabilities and
all parameter gradients against explicit word-qubit rotations and dense Hamiltonian
circuits, confirm initial identity against the same-seed unrotated model, demonstrate
auxiliary-rotation invariance, check fixed/frozen parameter budgets and restore a
nonzero-readout checkpoint. Existing scientific tests continue to pass.

Generated-tool syntax, imports, Ruff/strict Pyright, archive/reference preflight and
checksums pass. At handoff the full pipeline was unrun; the user subsequently completed it and
the independent audit above passed. No new
training was launched or monitored by the assistant.

```bash
bash output/product-readout/launch.sh
```

The launcher uses the local
[codex-long-jobs skill](/home/malo/.codex/skills/codex-long-jobs/SKILL.md), with
no viewer and event-only delivery to preserve manual handoff. It prints the job
name, log/state paths and log-follow command. Return after completion for analysis.
Allow roughly **30–90 minutes**, an unmeasured planning range based on the recent
20.7-minute two-thread rank-128 product fit plus readout overhead. It is not a
measured timing or upper bound. Prior timing estimates have both under- and
overshot; this new range should not be treated as a guarantee.

Source/data/reference checksums are checked before and after training. A runner
lock and live-worker guard prevent duplicate fits. Failed runs preserve completed
selections; resume missing fits from initialization using the printed batch path:

```bash
bash output/product-readout/launch.sh output/product-readout/batch-XXXXXX
```

The `READOUT_SCREEN_COMPLETE` marker is printed only after all fits, lock checks
and summaries succeed. Requested cancellation uses the controller for the whole
process group. Meaningful generation and a useful quantum advantage remain unproven.
