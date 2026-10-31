# Product Hamiltonian capacity: declared manual comparison

31 October 2026. Completed and independently audited. See the
[completed results](#completed-results); the original declared protocol follows. Return to the faster product-state
model after the [negative sequential screen](sequence.md#completed-results).
[Fixed-checkpoint diagnostics](product-diagnostics.md) show a substantial sampling
deficit beyond the dominant target, and about one quarter of incoming state mass
in each learned rank-64 subspace. This motivates a capacity hypothesis, not a
claim that rank is a proven bottleneck or that increasing it must improve results.

## Completed results

All four new fits in `output/product-capacity/batch-Z0fDku` completed, and the job
printed its final success marker. Independent evaluation reproduced all four
selected prediction arrays and metrics to 1e-12, with source/data/weight locks
intact. Both archived rank-64 references, all 30 paired intervals and all 18 saved
sample renderings were checked. No new samples or held-out test scoring occurred.

| Model | NLL | Sampling | Argmax | Target bits/byte | Fit minutes |
|---|---:|---:|---:|---:|---:|
| Product rank 32 | 3.0327 | 22.95% | 36.13% | 2.7298 | 11.6 |
| Product rank 64, reference | 2.9673 | 23.17% | 36.77% | 2.6710 | 27.7 |
| Product rank 128 | 2.9281 | 23.20% | 37.11% | 2.6356 | 20.7 |
| MLP matched to rank 32 | 2.8205 | 30.45% | 40.50% | 2.5388 | 4.0 |
| MLP matched to rank 64, reference | 2.7966 | 30.84% | 41.06% | 2.5172 | 8.6 |
| MLP matched to rank 128 | 2.8303 | 31.04% | 40.58% | 2.5476 | 7.3 |
| All-training bigram | 3.0593 | 23.16% | 34.03% | 2.7537 | — |

Increasing quantum rank improves NLL across this single-seed screen. Rank 128
minus rank 64 gives −.03923, interval [−.05410, −.02515], but sampling changes
only +.026 percentage points, interval [−.068, +.124]. Argmax +.342 points also
has an interval crossing zero. Rank 128 beats rank 32 NLL by .10458 and sampling
by .243 points. Both new quantum fits selected their final epoch 30; rank-128
train NLL is 2.7198, versus validation 2.9281.

The rank-128 product model still trails its matched MLP by .09783 NLL, interval
[.03900, .14859], and **7.838 sampling percentage points**, interval
[7.212, 8.496] in deficit magnitude. Even the smaller 161,200-parameter MLP beats
its NLL and sampling. Larger MLP capacity does not monotonically improve NLL;
the previous rank-64-matched MLP remains the best NLL reference. More quantum rank
therefore improves one metric modestly but does not explain or close the main
quantum/classical gap. No rank-256 escalation is justified by this screen alone.

Rank-128 product beats bigram NLL by .13115 and argmax by 3.076 points. Sampling
is only +.040 points, interval [−.224, +.287], so a sampling advantage remains
unestablished. All intervals are conditional article bootstrap intervals on reused
validation; seed/selection uncertainty is omitted. Different parameter shapes and
thread counts further limit causal interpretations of the historical comparison.

Rank-128 samples remain fragmented and contain two replacement characters and
one other control character across the three draws, despite no sampled UNKs.
The rank-32 samples contain one control character. The large MLP's three samples
have none of these outcomes but still fail to produce reliable meaningful prose.
Every saved outcome is retained, without retries or filtering.

Actual maximum new fit duration was **20.7 minutes**, below the broad 1–3-hour
planning range. The new runs used two threads each; historical rank-64 fits used
one with a different concurrent workload. This does not prove that rank 128 is
intrinsically faster, but the measured two-thread product run is a better timing
reference for similar future experiments than the slow sequence encoder.

Evidence: [metrics](product-capacity-validation.csv),
[paired intervals](product-capacity-paired.csv),
[sample diagnostics](product-capacity-sample-diagnostics.csv).
The next hypothesis tests a learned local measurement basis with only 18 extra
quantum rotation parameters, instead of another large rank increase. It still
requires independently trained controls; improvement is not established.

## Controlled question

Test rank 32 and rank 128 against the completed rank-64 product reference while
holding the two-block architecture, token-angle controller, data, optimizer and
selection rule fixed. Each rank has an independently trained approximately
parameter-matched product-feature MLP. A gain shared by the MLP may simply reflect
more capacity; a quantum-only recipe gain still requires replication and stronger
attribution before claiming advantage.

| Rank | Product parameters | Matched MLP parameters | Status |
|---:|---:|---:|---|
| 32 | 161,876 | 161,200 | Two new fits |
| 64 | 293,012 | 292,695 | Reuse locked expanded-data fits |
| 128 | 555,284 | 554,138 | Two new fits |

Controller parameters remain 30,740 at every rank. The MLP receives the same real
prepared amplitudes and later RZ angles as the existing product-feature control;
its hidden width follows the existing Hamiltonian-budget matching formula.
No sequence encoder, classical vocabulary decoder on the quantum path, new
calibration, postselection or supplied count prior is introduced.

All four new fits use the exact expanded dataset at
`output/product-exposure/batch-MDwDbm/data/full.npz`: 131,072 training targets,
4,096 validation targets, the original training-only byte BPE, 128-token contexts,
official article partitions and fixed count statistics. No tokenizer refit or new
test scoring. Rank 64 references retain their previously audited predictions.

New fits use seed 11, 30 epochs, batch 512, Adam LR .003, clipping 5, float64/
complex128, two Hamiltonian layers, one auxiliary qubit, recent window 16 and
phase scale 1. NLL trains and selects checkpoints, using the existing sampling
tie-break and including epoch zero. Each fit gets 7,680 optimizer updates and 31
validation checkpoints. No new LR or seed sweep. All weights start fresh.

Four concurrent processes each request **two Torch threads**, for eight requested
threads on the current eight-physical-core laptop. The old rank-64 runs used one
thread per fit with a different concurrent workload. Timing differences therefore
are not a controlled rank-only speed benchmark. Different parameter dimensions
also change initialization draws; a single-seed rank response cannot isolate a
universal causal effect of rank or establish reproducibility across seeds.

This is a focused capacity screen, not a claim of matched hardware execution or
quantum speedup. A larger spectral basis costs more classical simulation, storage
and potential hardware compilation. Input preparation/readout semantics are
unchanged. No useful long-context or text-generation claim follows from extra
parameters alone. New frozen-core/context controls and fresh confirmation remain
necessary if a larger-rank candidate merits further investigation.

## Checks and reports

Both archived rank-64 references match source snapshot `b4527ba`, and their weights
reproduced the saved validation scores during the completed diagnostics. Preflight
checks that source, datasets and selected weights. The checksum manifest pins
reference predictions and scripts as well as current source and data. New runs
use the ordinary current-source selection loader; old weights are not silently
loaded under mismatching source.

The summary checks every new selected validation prediction/metric to 1e-12,
verifies paired targets/groups/byte lengths and writes eight metric rows: four
new fits, two rank-64 references and unchanged bigram/unigram controls. It writes
30 descriptive paired article-bootstrap intervals: rank-128 product minus the
other seven models, plus rank-128 MLP versus rank-64 MLP and the two rank-32
models versus their respective rank-64 references, for NLL, sampling and argmax.
Intervals exclude seed/selection uncertainty on adaptively reused validation.
Target bits per byte retains its existing sampled-target denominator.

The same three prompts/seeds generate 64 direct draws each for four new models
and two count controls: 18 samples, exact IDs in NPZ and text in CSV. No filtering,
retries or temperature adjustment. Sampling and NLL remain distinct, and fragmented
text is still a failure of the broader generation objective even if one metric
improves. Old reference samples stay in their original batch.

No model source changed for this screen. Parameter totals were checked by model
construction without training. Generated-tool Ruff/strict Pyright, module imports,
Bash syntax, archived-reference preflight and input checksums pass. At handoff the four-fit pipeline was unrun; the user subsequently completed it
and the independent audit above passed.

## Manual launch

```bash
bash output/product-capacity/launch.sh
```

The launcher uses the local
[codex-long-jobs skill](/home/malo/.codex/skills/codex-long-jobs/SKILL.md) with
no viewer and event-only delivery, preserving manual execution and return to the
conversation. It prints the job name, log/state paths and log-follow command.
No training was launched or monitored by the assistant.

Allow a broad **1–3 hours**, an unbenchmarked planning range based on the previous
28-minute rank-64 product fit and higher-rank/threading overhead. It is not a
measured runtime or upper bound. The sequential screen's 45–90 minute estimate
underestimated its 3.44-hour learned fit; that estimate is not reused here.

A lock and live-worker guard reject duplicate execution. Input checksums are
checked before and after training. The completion marker
`CAPACITY_SCREEN_COMPLETE` is printed only after all required fits and summaries
succeed. Resume missing fits from initialization with the printed batch path:

```bash
bash output/product-capacity/launch.sh output/product-capacity/batch-XXXXXX
```

Completed selections are verified and preserved. Requested cancellation uses the
job controller to stop the entire process group.
