# Larger segmented-context study: all twelve fits verified

29 October 2026. All twelve planned fits in
`output/segmented-large/batch-IAnOGm` are complete. The user ran the targeted
recovery for sector-mixture seeds 22 and 33; these finished in 32.4 and 33.2
minutes. The original ten fits are preserved, as are the two unfinished run
directories. No study workers remain. The earlier interruption's cause is unknown.

## Verification and protocol

The original source/data/runner and recovery checksum manifests pass. For all
twelve fits, the selection loader verified dataset, source and selected-weight
hashes. The audit checked model/seed identities, 131,072 training targets, 4,096
validation targets, architecture and optimizer/selection settings. Reloaded
selected weights reproduce every stored validation score within atol/rtol 1e-12;
recorded scalar metrics agree within 1e-12. Paired targets and groups match.
Test remains unscored.

All models use WikiText-2, the same training targets and validation articles,
512-word vocabulary with train-only statistics, and a 32-word context. Quantum
models have two rank-64 Hamiltonian blocks and 262,272 parameters. The same-input
MLP has 261,734 parameters. All use NLL loss and validation NLL checkpoint
selection, 30 epochs, batch 512, quantum LR .001 and MLP LR .003. See the
[original protocol](segmented.md). UNK is 41.80% of validation targets; known-word
probabilities are not renormalized.

The generated full audit is `output/segmented-large/audit_full.py`. Versioned
[fit records](segmented-large-fits.csv), [model summaries](segmented-large-summary.csv)
and [paired differences](segmented-large-paired.csv) contain all three seeds.
The earlier ten-fit snapshots remain available as
[partial fits](segmented-large-partial-fits.csv),
[partial summaries](segmented-large-partial-summary.csv) and
[partial pairs](segmented-large-partial-paired.csv); they are historical records.

## Completed three-seed results

Means over seeds 11, 22 and 33. Sampling is expected one-shot correct-token
probability, distinct from argmax accuracy and NLL.

| Model | Validation NLL | NLL seed SD | Sampling success | Known-target success | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier auxiliary preparation | 3.5547 | .00530 | 14.04% | 2.43% | 35.94% |
| Coherent segmented input | 3.5655 | .00551 | 12.37% | 2.32% | 36.65% |
| Sector mixture | 3.4884 | .00277 | 12.26% | 2.36% | 37.74% |
| Same-input MLP | **3.1641** | .00287 | **22.09%** | **5.72%** | **42.55%** |

The two recovered mixture NLLs are 3.491132 (seed 22, epoch 30) and 3.485593
(seed 33, epoch 27), consistent with 3.488350 (seed 11, epoch 29). The mixture
is the best quantum recipe here by the declared NLL criterion. The earlier
auxiliary model has higher sampling success. MLP leads both overall and on
known-target metrics. The previously evaluated all-training bigram has NLL
3.0300 and sampling 22.68%, but sees more unique training transitions.

Coherent minus sector-mixture NLL is **+.07710**, article-group descriptive 95%
interval [+.06409, +.08985]. Known-target NLL is worse by .12192, interval
[+.10206, +.14267]. Coherent aggregate sampling is higher by .114 percentage
points, interval [+.052, +.169], while known-target sampling is lower by .043
points, interval [−.088, −.002]. Aggregate sampling alone therefore does not
establish a useful cross-sector interference benefit. The sector mixture retains
coherence within each context stream; this is not full word-basis dephasing.

Coherent minus earlier-auxiliary NLL is +.01076, interval [−.01887, +.04176],
with sampling lower by 1.666 percentage points. The earlier small-subset coherent
NLL edge does not establish an overall improvement at this larger budget.
Against MLP, coherent NLL is worse by .40136, interval [+.35107, +.45663], and
sampling is lower by 9.716 points, interval [−10.441, −9.114]. Even the better
mixture remains .32426 NLL behind MLP and 9.830 points behind on sampling.
There is no demonstrated quantum advantage.

Paired intervals average differences across the three fitted seeds, then
bootstrap validation article groups. They condition on those seeds and reused
validation examples; adaptive architecture/checkpoint selection and broader
seed uncertainty are not included. This is validation research, not fresh
held-out confirmation. More data improved the recorded recipes relative to
smaller-subset fits, but the comparison changes examples and update count.

## Recovery and implications

The recovery runner successfully started only the two unfinished seeds from
initialization, retained completed fits, checked manifests, and produced strict
twelve-fit `validation.csv` and `paired.csv` summaries. The original mixture fit
ran for 48.4 minutes under six-worker contention; the two recovery fits took
32–33 minutes under two workers. These are different contention conditions,
not evidence of a model speedup. No additional recovery is needed.

The three-seed evidence does not justify scaling coherent segmentation unchanged.
A completed [training-only allocation diagnostic](sector-calibration.md) fitted
a single recent-sector probability while freezing selected quantum weights.
It stayed near .515 and improved validation NLL by only .00082, with a small
aggregate sampling decrease. This does not support another long run merely to
tune the global allocation. It cannot replace a jointly retrained comparison. Competitive
known-word generation, useful long-context quantum computation and hardware
advantage remain unresolved.
