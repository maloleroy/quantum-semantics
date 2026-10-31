# Product Hamiltonian capacity: declared manual comparison

31 October 2026. Prepared, **not trained**. Return to the faster product-state
model after the [negative sequential screen](sequence.md#completed-results).
[Fixed-checkpoint diagnostics](product-diagnostics.md) show a substantial sampling
deficit beyond the dominant target, and about one quarter of incoming state mass
in each learned rank-64 subspace. This motivates a capacity hypothesis, not a
claim that rank is a proven bottleneck or that increasing it must improve results.

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
Bash syntax, archived-reference preflight and input checksums pass. The four-fit
training/summary pipeline is **unrun**.

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
