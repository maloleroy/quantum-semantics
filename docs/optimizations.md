# Optimizations and compute

## CPU scaling pilot and next dataset (2026-10-23)

**Scope correction:** these timings describe the latent hybrid POC, with a
learned classical encoder/controller and decoder. Four qubits mean 16 latent
basis states, independent of output vocabulary. A vocabulary-space model with
8,192 word basis states requires 13 qubits, or 14 with source/target roles.
Neither its sparse backend nor its runtime has been profiled here. The hybrid's
learning gains have not been attributed to its Hamiltonian, so these profiles
are not a justification for scaling the intended vocabulary-space experiment.

Planning constraint supplied by the user: **24 physical CPU cores, approximately
10 GB RAM and a few hours wall time**. The available profiling host has eight
physical Ryzen 7 7840S cores / 16 logical CPUs. Its short measurements are not
a benchmark of the future 24-core host.

Eight synthetic step profiles used the actual `ContextResidual` implementation,
batch 128, AdamW, clipping and backpropagation through the matrix exponential.
Each had two warmups and five timed updates. The table shows medians; it excludes
corpus preparation/loading, validation, final training metrics, checkpoints,
concurrent-fit contention and sustained thermal behavior.

| Qubits | Trainable tensors | Vocabulary / context | CPU threads | Targets/s | 2M targets ×10 passes, training-only |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 | 465,104 | 4,096 / 16 | 4 | 13,775 | 24 min |
| 5 | 2,376,096 | 8,192 / 32 | 4 | 3,241 | 1h43m |
| 5 | 2,376,096 | 8,192 / 32 | 8 | 5,206 | 1h04m |
| 6 | 2,646,464 | 8,192 / 32 | 4 | 821 | 6h46m |
| 4 | 84,044,816 | 8,192 / 32 | 4 | 230 | 24h10m |

The small model uses embedding/hidden widths 64/128. The 5/6-qubit models use
128/256, and the 84M example uses 2,048/1,024. These are execution probes, with
no corpus learning or quality comparison. They do not isolate every factor
when changing model width or vocabulary. The 5-to-6-qubit pair holds those
classical widths and vocabulary fixed.

Provisional planning ranges for approximately ten full training passes:

- **4 qubits / 0.5M parameters:** a 5–10M-token corpus is a plausible larger target.
- **5 qubits / 2.4M parameters:** start with 1–3M training tokens. Around 5M is a
  stretch requiring fewer passes, fewer repeated fits or a larger time allowance.
- **6 qubits / 2.6M parameters:** about 0.3–0.8M tokens is more consistent with
  the same time allowance; 2M ×10 already projects to nearly seven hours at four
  threads before overhead.

These are forecast ranges, not hard dataset-size limits. Total processed
targets equal corpus tokens ×passes ×fits. At 5 qubits, the measured eight-thread
rate suggests about 1.1 hours of training for 2M ×10 in one fit. Three eight-thread
workers on 24 physical cores could cover six comparable fits in two waves:
about 2.1 hours training-only, with a **3–6-hour planning allowance** for a small
paired campaign after streaming changes. This assumes comparable per-core
performance and effective thread scaling on the actual host; neither is measured.
It does not include another full hyperparameter grid. RAM must bound worker count.

### Memory and backend changes required

The present implementation builds a dense Hamiltonian and full exponential per
context. At q qubits, one complex64 matrix occupies `8 * 4**q` bytes. At **19
qubits that is 2 TiB for one matrix**, before batching or gradients. The current
generator count also equals `2**q`, giving a dense generator bank of
`8 * 8**q` bytes. Moving toward the user's <20-qubit ceiling requires a new
structured/matrix-free evolution backend and compact readout; it is not a
practical setting for the current dense POC.

The current runner also needs streaming before a million-token corpus:

- `score()` concatenates every vocabulary distribution. One million targets
  ×8,192 classes ×float32 is **30.52 GiB**. Accumulate metrics by batch instead.
- Generate context windows from a token stream instead of materializing all
  overlapping windows, and keep official document/split boundaries.
- A dense float32 bigram table at vocabulary 8,192 is **256 MiB** per copy. Share
  immutable data, cap workers and avoid the unused dense co-occurrence graph
  when preparing the latent model. Token vocabulary should stay around 4K–8K.
- Train-only subword tokenization avoids a huge word vocabulary and a large UNK
  target population. It changes token counts and the meaning of perplexity.

Weights/gradients/Adam moments for 100M float32 parameters are roughly 1.6 GB,
before activations and temporary arrays. Such a model may fit individually in
10 GB but is a poor first target for a paired multi-seed CPU campaign. The
84M profile illustrates the throughput cost. Parameter count alone is not a
runtime or memory bound for this Hamiltonian model.

### Dataset choice

**WikiText-2 is the recommended first standard corpus.** Its published training
split has 2,088,628 word-level tokens and standard validation/test splits. Use
the raw version with a training-only 4K–8K subword tokenizer; record resulting
token counts and do not compare that perplexity directly to the published
word-level metric. Preserve article boundaries and official partitions.
[Original dataset paper](https://arxiv.org/html/1609.07843),
[dataset files/card](https://huggingface.co/datasets/Salesforce/wikitext).

A fixed **1–3M-token subset of TinyStories** is an alternative with simpler
language. The full dataset is large despite its name; select complete stories
to meet the token budget and retain independent evaluation stories. Its authors
demonstrate coherent text with models below 10M parameters; that is not evidence
that this Hamiltonian model will match those results.
[TinyStories paper](https://arxiv.org/abs/2305.07759),
[dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

No dataset was downloaded or large training launched in this planning step.
Local pilot details are recorded in [validation](validation.md).

Keep model quality, execution cost and forecast scope separate. A simulator
optimization is not a quantum-hardware speedup; larger optimizer batches change
training, whereas a smaller simulation microbatch is intended to bound memory.
The latter can still alter floating reductions/dropout trajectories.

## What the code implements

| Change | Code / effect | Verification boundary |
| --- | --- | --- |
| Reuse unique contexts | `state_cache.py`: compute one prediction per distinct context and scatter to examples. | Preserves example weighting/order; cache/simulator tests cover repeated contexts and eviction. |
| Device statevectors / SPSA lanes | `simulation.py`: tensor batches, both perturbations together, direct RX flip arithmetic and combined diagonal phases. | Compare against Qiskit and sequential SPSA; CPU is not CUDA certification. |
| Batched fixed encoder | `context_matrices`, `encoding_angles_batch`, `QCSEModel.encode_many`, `TensorSimulator.encode`. | Avoids one Qiskit circuit construction per context; preserves all five scalar recipes and bounds encoding calls by simulation batch. Newly published local code, not measured DGX acceleration. |
| Bounded streaming cache | Small context sets packed on device; large sets retain independent CPU state allocations. | Evicted views must not retain a whole temporary tensor. Cache budget bounds retained states, not all host data, serialization or peak temporary memory. |
| Analytic identity hybrid | `benchmark_models.py`: closed-form XYZ product-state features. | Exact control without statevectors. Matched CRZ/id cost ratio is valid; old-source versus new-source timing is not an isolated optimization measurement. |
| Sampled monitoring | Independent fixed 2048-target samples; full final validation once. | Removes repeated full-corpus inference/checkpoint embeddings; changes monitoring schedule, not training eligibility. Older capped final evaluation remains a separate protocol. |
| Timing/evidence ledgers | Phase/epoch/setup/train/eval/save/fit/wrapper records plus `sacct`. | A wrapper span is not total allocation accounting. `.batch`/`.extern` entries are not extra jobs. |

The unpublished encoder review found phase/hash row/column indexing drift,
unbounded equal-length encoding groups and cached tensor views retaining batch
storage. The publication fixes those issues and adds parity/storage regressions.
It does not replace any source/hash in frozen 483bea6 studies.

## Measured DGX costs

| Cohort | Work | Elapsed scope | Interpretation |
| --- | ---: | --- | --- |
| Timed 90 |9.9525 allocation-hours|2h30m56s array|Short early-stopped protocol. Pilot separate. |
| Screen 156 |366.492 array allocation-hours|94h19m33s array|Includes short reused task; calibrated all-fit cost 370.506h and serial pilot+CV≈98.35h have different scope. |
| Follow-up258+30 |348.443 allocation-hours|97h23m41s workflow|Includes recorded stages/pilots; peak four, about 89.4% four-slot utilization. |
| Target controls 32 |33.616 wrapper MIG-hours|10h46m50s wrappers|Profile job 10334, queueing and packaging excluded. |
| Causal completion 8 |15.3573 wrapper MIG-hours|3h52m33s wrappers|Forecast 16.391h /4.152h; actual work/span 6.31%/6.65% below forecast. Pilot 10399 excluded. |
| Replicated gradients |450 recorded stage seconds|Preflight+two objective probes|Not training cost; full Slurm allocation/queueing absent. |

Exact historical reports and their recovery commands are listed in
[findings](findings.md#evidence-and-recovery). Do not pool costs from these
different recipes into one fit rate.

## Microbatch 64 opportunity

One 10 GB MIG pilot covers both tasks/four anchors, batch 64 and final partial 32,
microbatches 8/16/32/64. Five measured steps follow one warmup. All recorded
forward/loss/gradient/one-Adam-parameter/moment checks pass. Eval-mode parity
disables dropout; normal-training timing keeps it.

| Model, optimizer batch 64 | L4 micro 8→64 ms | Speedup | L2R2 micro 8→64 ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| MLP |8.835→1.816|4.87×|8.950→1.659|5.40×|
| BERT |50.056→7.853|6.37×|50.215→7.750|6.48×|
| CRZ |230.543→30.197|7.63×|230.164→30.087|7.65×|
| identity |11.244→2.165|5.20×|11.390→2.153|5.29×|

Max recorded PyTorch reserved memory 113.25 MB excludes non-Torch allocations.
Largest discrepancies: logits 2.15e−6, gradients 1.34e−7, Adam parameters 1.45e−5.
These are saved CUDA summary checks, not replayed CUDA tensors on the Mac.
All scientific control fits retained microbatch 8.

The proposed 16 full-trajectory checks have **about 1.46 MIG-hours training-only**
work from 31300×measured median step times. This excludes evaluation, gradient
diagnostics, transfer, setup, saves, scheduling tails and queueing. Evaluation
was not profiled at micro 64. **There is no trustworthy whole-batch ETA yet.**
Do not divide an old wall-clock estimate by 7.7 or treat five repeats as a
confidence interval. Fresh profiles and full-budget paired checks precede adoption.

## Other optimization candidates

The product-input depth 2 nearest-neighbor hybrid has Schmidt rank≤4 across
each cut. An exact matrix product state (MPS) evaluator could avoid exponential-in-q statevectors at
fixed depth; at q8 its overhead may outweigh savings. State/XYZ/input-gradient/
parameter-gradient/Adam/resume parity and fresh timing are required first.
No MPS backend is implemented. This bound does not automatically cover the
paper context encoder/rings or growing depth.

Profile the actual bottleneck before tuning worker counts or adding qubits.
Keep Torch/BLAS threads explicit; local checks use one. Old 32-thread launcher
targets are historical, not measurements of an optimal four-CPU allocation.
Avoid compilation, mixed precision, graph caching or new kernels until there
is an isolated measured need and a parity check. New model/readout/sampler/graph
modules need their own profiles; q8 timings are not q14 or graph-evolution ETAs.
