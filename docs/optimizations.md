# Optimizations and compute

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

See the original [run reports](runs.md) for timestamps, accounting filters and
raw summaries. Do not pool costs from these different recipes into one fit rate.

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
