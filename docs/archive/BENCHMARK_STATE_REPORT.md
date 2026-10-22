# QCSE/BERT: state, evidence and compute plan

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


Audit date: **2026-10-08**. Branch: `codex/qcse-bert-robust-benchmark`.

Follow-up **2026-10-09**: [TRAINABILITY.md](TRAINABILITY.md) records the barren-plateau
literature, circuit-gradient diagnostics, optional qubit/depth/init controls,
new tests and locally prepared 1,530-fit plan. The evidence below remains the
historical standard-matrix audit; current code requires freshly prepared studies.

## Current conclusion

The controlled benchmark is implemented, locally tested and prepared at full
corpus scale. **No new DGX job has been submitted and no production comparison
has completed.** There is currently no evidence that QCSE outperforms BERT,
generalizes semantically, or offers a quantum computational advantage.

The earlier checks covered execution and important circuit properties, but were
insufficient to validate every prepared row, independently reconstruct saved
results, or support a DGX duration estimate. This audit fills those local gaps;
hardware performance and comparative research results remain open.

## What was built and corrected

The first four commits added the research audit, task-matched model adapters,
shared experimental protocol and a full-corpus prepared plan. The implementation
uses one runner for QCSE, learned quantum hybrid, mean embeddings, ordered MLP,
small BERT MLM and the existing causal attention controls. It includes seven
ansatz choices, frozen circuits, depth/context/ID-permutation/objective controls,
five grouped development folds, three seeds, validation-only selection, fresh
development refits and paired seed/sentence bootstrap.

Every fit keeps continuation and inference checkpoints, exact prediction IDs,
word NLL/top-1/top-5, a training-only frequency baseline, decoded examples,
capacity/update/draw counts, timing and device provenance. Prepared studies save
input, artifact and effective-source hashes plus source/lockfile snapshots.

The second audit corrected four concrete issues:

1. Selection and reporting previously trusted aggregate result fields after
   metadata checks. They now reconstruct aggregates, retrieval correctness and
   frequency scores from saved predictions, verify them against prepared splits,
   and validate training budgets. Corrupted results are rejected.
2. Resume previously replaced runtime provenance. Runtime and CLI status records
   now retain individual attempts. Fit seconds include setup; failed partial
   epochs and preflight costs still require attempt/Slurm accounting.
3. The 1,000/230-task array split could halve utilization when its small array
   finished early. It is now 615/615, five simultaneous jobs each. Resource-option
   abbreviations cannot override array/export/dependency controls.
4. CUDA reserved memory is now retained alongside allocated memory. Independent
   data-audit, batch-profiling and compute-projection tools were added.

Existing uncommitted focused-experiment changes were preserved. Historical
results from different objectives, splits and budgets remain separate evidence.

## What was checked

| Level | Evidence | Limit |
| --- | --- | --- |
| Full corpus | Rebuilt vocabulary/frequencies and deduplicated sentence pool from both source files; checked source/artifact/snapshot hashes | Shared full-corpus vocabulary is transductive |
| Every example | Independently reconstructed all **3,368,084** context/target rows from sentence positions | Does not establish language quality |
| Every split | All five folds partition development, validation groups cover it exactly once, train/validation/test are disjoint; both tracks share outer groups | Exact duplicates grouped; near-duplicate/template leakage unmeasured |
| Circuit values | All seven ansatzes compared with Qiskit; original encoding/Z marginals checked; 14-qubit/64-layer backend parity previously passed at approximately 3.3e-15 maximum error | CPU evidence |
| Derivatives | Six trainable ansatzes passed independent finite-difference gradcheck; dead final diagonal gradients verified | Small-register numerical gradient checks |
| Training | QCSE/hybrid/MLP/BERT weighted microbatch gradients match full batches with dropout disabled; interrupted BERT restores Adam, dropout/sampler RNG and exact CPU state | Microbatch-dependent dropout draws; no cross-device bit equality |
| Saved evidence | All **24** real-corpus smoke fits validated; all selected checkpoints reloaded; **192** replayed predictions match ranks/metrics exactly | Tiny smoke budget; no quality inference |
| Statistics | Pair alignment, whole-sentence resampling, token weighting, identity comparisons and nonfinite-input rejection tested | Conditional exploratory intervals, three seeds, no multiplicity correction |
| Operations | Balanced array offsets/concurrency, submission receipts and override rejection tested without contacting Slurm | Actual scheduler execution unverified |
| Whole repository | **115 passed, 46 skipped**, 129.11 seconds; Ruff, Pyright and shell/diff checks pass | All skipped cases require unavailable CUDA/MPS |

Final focused rerun: **26 passed**, 25.25 seconds. A clean archive of source
commit `2949842` also passed all 43 CPU adapter preflights, independently of the
uncommitted focused-experiment edits.

Raw evidence is retained under `outputs/benchmark-audit/`: `prepared-audit.json`,
`smoke-replay.json`, `bit-metric-diagnostic.json`, profiles and `compute.json`.
The original real-corpus smoke is
`outputs/robust-smoke/qcse-bert-20261007T225825Z-fe_jdj9s/`.
Historical artifact replay deliberately reads its saved manifest directly;
current launches still enforce current-source hashes.

The corrected full-data plan is
`outputs/robust-dgx-audited-plan/qcse-bert-20261007T233022Z-83f4xxt4/`.
Its manifest fingerprint is
`ad263266423e0f91fcb577078b77aa4b4bbf0c7ed7b85fedbb5cf3251f2b6798`,
with source commit `2949842e19ff5057c949a66b65d66113d13bd6fa` and an explicitly
recorded dirty workspace. Its data artifacts are byte-identical to the original
plan; source/model provenance was checked separately. The complete independent
data audit passed again on this current plan. Its dry run produces exactly two
`0-614%5` arrays, with second-array offset 615. The previous manifest predates the
integrity fixes and must not be launched with changed source.

## What the literature and algebra establish

[QCSE v2](https://arxiv.org/html/2509.05729v2) uses numeric context features,
encoding/ansatz gates and Z marginals. Its reported accuracy thresholds bit
agreement, rather than requiring the correct vocabulary word. Our independent
diagnostic illustrates the difference: on the full causal development targets,
an all-zero 14-bit prediction scores **98.68%** at agreement ≥0.5, but only
**7.86%** exact-word top-1. Unbiased random bits pass that threshold with
probability **60.47%** at 14 bits. These calculations use our corpus/bit width;
they do not reproduce the paper's datasets or dispute its reported arithmetic.

Direct algebra gives CRZ's generator `(Z_target - Z_control Z_target)/4`;
RZZ and RXX instead use `ZZ/2` and `XX/2`. They are distinct interactions, each
Hermitian for real weights. The generally nonsymmetric context matrix is not
itself a Hamiltonian. Final RZ/CRZ/RZZ operations commute with Z readout and
therefore have no effect on its marginals. More nominal parameters do not
necessarily mean more effective capacity. The word likelihood used here is a
normalized product of marginal bit probabilities, not the joint Born law.

[BERT](https://aclanthology.org/N19-1423/) supplies a concrete Transformer
architecture and original pretraining protocol. Our actual BERT MLM modules
are trained from scratch at width 16/32, depth 2/4, using shared word IDs and
one absent target per context. There is no WordPiece, external pretraining or
next-sentence objective. They are controlled BERT-style architectural baselines,
not a reproduction of BERT-base or a comparison against pretrained BERT.

[QNLP](https://arxiv.org/abs/2005.04147) and
[lambeq](https://arxiv.org/abs/2110.04236) motivate grammar-derived circuit
composition, which this fixed numeric context encoder does not implement.
[Expressibility/entanglement descriptors](https://arxiv.org/abs/1905.10876)
do not by themselves establish NLP utility; the
[barren-plateau literature](https://arxiv.org/abs/1803.11173) motivates gradient
and depth controls. [NLP significance-testing guidance](https://aclanthology.org/P18-1128/)
supports choosing the statistical unit carefully: overlapping word windows are
dependent. Here resampling preserves complete sentences.

Remaining comparison limits are material: search sizes and parameter/FLOP costs
are unequal; the full vocabulary includes test text; identity/local ansatz arms
retain entangling encoding gates; CV training folds overlap; bootstrap intervals
condition on selection and a fixed split. This is an exploratory prediction
study, not a completed test of general semantic representations.

## Compute duration

**Updated estimate:** [BENCHMARK_COMPUTE.md](BENCHMARK_COMPUTE.md) records fresh
2026-10-09 timings and the extended 1,530-fit matrix. The historical measurements
below are retained as evidence and should not be used as the current ETA.

The matrix contains **1,230 CV fits**, followed by **33 selected refits**. Corpus:
**202,172 sentences**, **10,864 words**. Causal/CBOW pools contain 1,582,956 and
1,785,128 examples. Final test uses 2,000 complete sentences: 15,944 causal and
17,944 CBOW examples. Per trainable epoch: 5,000 replacement draws, optimizer
batch 64, microbatch 8, **79 updates**, plus up to 2,048 monitoring examples.
Fifty epochs mean 250,000 draws, roughly 22–25% of a fold's eligible training
examples by count, with repeats; they are not 50 full-corpus passes.

Measured all **43** adapters at the real vocabulary on CPU, four Torch threads,
one warmup and three measured training/evaluation batches. Representative median
seconds per optimizer batch of 64, including microbatch accumulation and Adam:

| Model | Parameters | Train / eval seconds per 64 |
| --- | ---: | ---: |
| QCSE, 14 qubits, 2 layers | 82 | 3.29 / 1.37 |
| QCSE, 14 qubits, 8 layers | 328 | 11.10 / 4.85 |
| Learned 4-qubit hybrid | 185,454 | 0.095 / 0.035 |
| Ordered MLP | 187,312 | 0.017 / 0.009 |
| BERT, width 16, 2 layers | 191,730 | 0.057 / 0.026 |
| BERT, width 32, 2 layers | 385,394 | 0.066 / 0.029 |

These are local batch measurements, not trained-model scores. Raw repeat timings
are retained. Some measurements overlapped CPU tests; variation is substantial.
CBOW spot checks and a follow-up without concurrent tests are retained separately.
Follow-up causal QCSE train batches measured 3.71 seconds at two layers and
12.32 seconds at eight; CBOW measured 5.27 and 8.87 seconds respectively.
These differences reinforce the need for an allocated-device pilot rather than
a precise extrapolated ETA.
The following workload projection uses the complete causal profile for both
tracks, which have the same tensor dimensions. It includes final inference and
conservatively prices refits using each family's most expensive potential winner.

| Completed epochs per trainable fit | CPU-equivalent hours, CV + refits | Ten slots, assumed 2× CPU speed | Ten slots, assumed 5× | Ten slots, assumed 10× |
| --- | ---: | ---: | ---: | ---: |
| 10 | 619 | 1.29 days | 0.52 days | 0.26 days |
| 25 | 1,431 | 2.98 days | 1.19 days | 0.60 days |
| 50 | 2,784 | 5.80 days | 2.32 days | 1.16 days |

**The speed factors are assumptions, not measured DGX acceleration or a confidence
interval.** A 25-epoch/5× planning case is about **286 MIG slot-hours / 29 hours
at ten continuously available slots**. At equal CPU speed it is about six days.
MIG may deliver less acceleration, including less than CPU speed for launch-bound
arms. No single reliable DGX ETA exists yet.

Startup, source/data hashing, both allocation preflights, checkpoint I/O,
failed/replayed work, reporting, queue waits and imperfect slot utilization are
excluded. Raw CPU variance and CBOW differences are additional uncertainty; do
not interpret decimals as prediction precision. Old cluster timing reflects
legacy SPSA, different validation scope and three fits per configuration, so it
does not calibrate this autograd pipeline.

The slowest 50-epoch CPU projection is **14.75 hours per fit**, exceeding the
12-hour job limit. A GPU pilot must verify duration and full-vocabulary/deep-arm
peak memory on an allocated **A100 10 GB MIG** device before the bulk launch.
CPU peak RSS in the initial all-arm profiler was about 511 MiB, a cumulative
process measurement that does not prove CUDA memory feasibility.

## Remaining compute and research

1. **DGX calibration:** prepare from the frozen cluster checkout; run both CUDA
   preflights, full-vocabulary batch profiling for all arms and CBOW spot checks.
   Run short end-to-end QCSE/deep/BERT pilot fits, including save/resume and final
   inference. Retain device UUID, driver/runtime versions, logs, time and memory.
   Replace assumed speed factors with measured per-arm costs and decide whether
   the 12-hour limit/microbatch size is suitable.
2. **Production computation:** run all 1,230 CV fits; validate completeness and
   evidence; freeze family winners/epochs; run all 33 fresh refits; produce the
   2,000-draw paired-bootstrap report and back up every study/log artifact.
3. **Analysis:** inspect optimization/gradient behavior, depth and generator
   effects, context/ID-permutation sensitivity, frequency controls, resource
   efficiency and seed variability. Report uncertainty and failed runs; no
   favorable-fold selection or test-driven retuning.
4. **Research beyond this matrix:** establish an inductive vocabulary and
   near-duplicate-aware split; add stronger classical capacity/compute controls
   and a separately specified pretrained-BERT comparison. Add downstream semantic
   similarity/inference tasks and, if pursued, a precisely defined Hermitian
   graph Hamiltonian or grammatical composition model. Hardware noise and any
   quantum-advantage claim require their own experiment.

See [BENCHMARK.md](BENCHMARK.md) for executable handoff commands and
[BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md) for detailed generator/comparison
definitions.
