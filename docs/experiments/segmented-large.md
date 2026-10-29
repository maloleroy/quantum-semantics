# Larger segmented-context study: ten completed fits, two missing

29 October 2026. Audited `output/segmented-large/batch-IAnOGm` after the user
reported that the run appeared complete. Ten of twelve planned fits have locked
selections, weights, histories and validation arrays. Sector-mixture seeds 22
and 33 have only run metadata/datasets and empty logs, with no fit records or
checkpoints. A process inventory found no surviving study workers. The cause
of termination is not established; empty logs do not diagnose it. No new fits
were started during this analysis.

## Verification

The original source/data/runner checksum manifest passes. For every completed
fit, the selection loader verified dataset, source and selected-weight hashes.
The audit checked the declared 131,072-example training budget, 4,096 validation
examples, seed/model identities, model dimensions and optimizer/selection settings.
It then reloaded each selected model and recomputed validation probabilities;
all stored per-example scores matched within atol/rtol 1e-12, and recorded scalar
metrics matched within 1e-12. Target/group arrays agree across paired models.
This recomputation used validation only. Test remains unscored.

The generated audit tool is `output/segmented-large/audit_partial.py`. Compact
[fit records](segmented-large-partial-fits.csv),
[model summaries](segmented-large-partial-summary.csv) and
[paired differences](segmented-large-partial-paired.csv) are versioned. Missing
seeds are explicitly excluded; no values are imputed. This is partial coverage,
not the completed three-seed mixture comparison.

## Observed results

Means across completed seeds; the mixture row contains only seed 11.

| Model | Completed seeds | Validation NLL | One-shot success | Known-target success | Argmax |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier auxiliary preparation | 11,22,33 | 3.5547 | 14.04% | 2.43% | 35.94% |
| Coherent segmented input | 11,22,33 | 3.5655 | 12.37% | 2.32% | 36.65% |
| Sector mixture | **11 only** | 3.4884 | 12.30% | 2.37% | 37.72% |
| Same-input MLP | 11,22,33 | **3.1641** | **22.09%** | **5.72%** | **42.55%** |

NLL seed standard deviations for the complete families are .00530, .00551 and
.00287 respectively. Known probabilities are not renormalized; validation UNK
rate is 41.80%. All rows use identical validation examples and, within this
study, identical training targets. Quantum LR is .001, MLP LR .003; parameters
are approximately matched as specified in the [protocol](segmented.md).
The all-training bigram's earlier NLL 3.0300/success 22.68% remains a stronger
NLL reference but sees more unique training transitions than the gradient models.

Three-seed coherent-segmentation minus earlier-auxiliary NLL is +.01076,
article-group descriptive 95% interval [−.01887, +.04176]. Sampling decreases
by 1.666 percentage points, interval [−1.966, −1.367]. Known-target NLL improves
slightly (5.0961 → 5.0476), but known-target sampling falls (2.43% → 2.32%).
Thus the earlier small-subset coherent NLL edge does not establish an overall
improvement at this larger training budget.

Against the same-input MLP, coherent NLL is worse by .40136, interval
[+.35107, +.45663], and sampling is lower by 9.716 percentage points, interval
[−10.441, −9.114]. Known-target metrics also favor the MLP. This is not evidence
of a quantum advantage.

For the **single matched seed 11**, coherent minus sector-mixture NLL is +.07902,
interval [+.06330, +.09567]; known-target NLL is worse by .12768. Coherent aggregate
sampling is slightly higher by .103 percentage points, while known-word sampling
is lower. The remaining mixture seeds are needed for the planned three-seed
comparison; do not present the one completed seed as its mean or variability.

Paired intervals average differences across the listed seeds, then bootstrap
validation article groups. They condition on those fitted seeds and reused
validation examples; adaptive architecture/checkpoint selection and broader
seed uncertainty are not included. The experiment is still validation research,
not fresh held-out confirmation. More data improved the recorded recipes relative
to their smaller-subset fits, but that comparison changes both examples and update
count and cannot isolate the cause.

## Recovery of the two unfinished fits

Completed fits are preserved, including the first mixture seed. The missing two
have no optimizer/checkpoint state to resume. `output/segmented-large/recover.sh`
starts only those seeds from their original initialization/settings in new run
subdirectories under the existing batch. Old empty logs/directories remain
available; new logs have unique recovery filenames. A lock and live-process
check prevent overlapping recovery; checksum and completed-fit checks run first.
Re-execution skips any newly completed recovery seed. A nonzero worker exit
stops final aggregation without discarding other results.

When both complete, the original strict twelve-fit summarizer writes
`validation.csv` and `paired.csv` inside `batch-IAnOGm`. The original pinned
runner and model sources were not changed. Recovery syntax, tool availability,
checksum guards and the completed-fit preflight were checked; long recovery
training has **not** been launched. The shell recovery process itself has not
been tested through a full long run.

Observed durations were about 32–34 minutes for completed coherent quantum fits,
48.4 minutes for the mixture fit and 12.8–13.8 minutes for MLP fits. Estimate
30–50 minutes for the two recovery workers running concurrently, depending on
contention/thermal behavior. This exceeds the user's ten-minute boundary:

```bash
bash output/segmented-large/recover.sh
```

After recovery, report the two missing mixture results before selecting a further
architecture or larger run. The current evidence favors the classical baseline;
competitive known-word generation, useful long-context quantum computation and
hardware advantage remain unresolved.
