# Progress — 2026-09-14

Branch: `corpus-training-sweep`, based on `perf/batched-tensor-simulation`.
Reference `main`: `87fa9a7cef03522957d442485a9bcee35c749a71`.

## Implemented

- Included the supplied `tatoeba.csv` and `cleaned_sentences.csv` alongside
  `phrases.csv` (the user-confirmed original third dataset).
- Header-aware loading, Unicode normalization, basic/dedupe/strict cleaning,
  seeded uniform/source-balanced sentence sampling, and all seven dataset subsets.
- A fixed vocabulary from all three full inputs before filtering/sampling/splitting:
  **11,428 words, 14 qubits**. Default dedupe retains 305,602 usable sentences before
  sampling. The supplied sources overlap; they are not independent benchmarks.
- Every new train/prepare invocation gets a unique folder under `outputs/` (or
  the `--output` parent). Archives/JSON use atomic replacement. Resume keeps the
  existing run unless `--output` requests a new folder. Checkpoints carry provenance.
- Backend preflight and regression tests; no change to the simulator or optimizer.
- 150 ten-epoch experiments: two objectives × five layer/batch settings × fifteen
  data profiles. The cluster caps each experiment at 128 sentences/512 examples.
- Slurm layout: **10 chains × 3 jobs × 5 sequential experiments**. Three arrays of
  ten tasks use `aftercorr`; at most ten jobs run at once. A failed group stops its
  chain. Submit using `bash scripts/submit_sweep.sh` after `uv sync --locked`.

## Validation

- Compared the actual Python model/training source from the reference `main`
  commit with 36 ten-epoch configurations: CPU/MPS, causal/CBOW, layers 2/8/64,
  Adam batches 2/8/32 paired with those depths, simulation batches 1/7/256.
- Largest CPU differences: weights `4.44e-16`, embeddings `2.78e-15`, BCE `1.31e-14`.
- Largest MPS differences: weights `2.48e-8`, embeddings `1.46e-6`, BCE `3.45e-6`.
  These checks support numerical equivalence within float32 rounding, not
  bit-for-bit GPU equivalence or identical trajectories for different Adam batches.
- MPS preflight at 14 qubits/64 layers: maximum Qiskit error `5.15e-7`;
  parallel-versus-sequential error `1.50e-7`.
- Final CPU/MPS/scheduling suite: **63 passed, 14 CUDA cases skipped**. Ruff,
  Pyright, shell syntax checks, and the skill validator passed.
- **16 real-data MPS runs completed ten epochs each**, with 64 examples and the
  same full vocabulary. Covered both objectives, all seven dataset subsets,
  layers 2/8/64, batches 16/64/256, all three cleaning modes, and both sampling modes.
  Verified finite archives, model/archive weight equality, exported embeddings,
  eleven history rows, standalone CPU inference from MPS checkpoints, and no
  sentence-text overlap between training and test partitions.
- Local evidence: `outputs/validation/main-parity.json`,
  `outputs/validation/pipeline-smoke.json`, and run folders in
  `outputs/pipeline-smoke/`. Generated outputs are intentionally not committed.
- The Slurm submitter is tested with a fake scheduler; actual submission has not
  occurred. CUDA preflight correctly fails on this Mac (no CUDA build/device).

## Cluster handoff

The user supplied an A100-SXM4-80GB with a 9,728 MiB MIG allocation and driver
580.173.02. Keep the 10 GB allocation for the bounded pipeline checks; the cache
and one SPSA buffer each use at most 64 MiB, excluding other intermediates/library
overhead. No CUDA memory peak or cluster execution has been measured locally.

Use the checkout's locked environment, retain Slurm's `CUDA_VISIBLE_DEVICES`,
and use the confirmed `gpu:nvidia_a100_1g.10gb:1` GRES for `prod10` (the batch script now
requests it by default). Each
scheduled job checks real CUDA inference, training, and checkpoint resume before
its five experiments. [README.md](README.md) contains the commands and matrix.

Keep subsequent edits minimal and driven by observed failures. The batching
implementation already passes the reference comparisons. Do not rewrite it to
address an unconfirmed CUDA problem; inspect the cluster preflight failure first.
