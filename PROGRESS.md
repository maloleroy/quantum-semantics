# Progress — 2026-09-14

Branch: `corpus-training-sweep`, based on `perf/batched-tensor-simulation`.
Reference `main`: `87fa9a7cef03522957d442485a9bcee35c749a71`.

## Implemented

- Active sources are `phrases.csv` and `cleaned_sentences.csv`. Tatoeba is excluded
  from both training and vocabulary because cleaned sentences are its curated version.
- Header-aware loading, Unicode normalization, basic/dedupe/strict cleaning,
  and seeded uniform/source-balanced sentence sampling.
- A fixed vocabulary from both full active inputs before filtering/sampling/splitting:
  **10,864 words, 14 qubits**. Default dedupe retains **202,172 sentences** before
  sampling. Old checkpoints retain their own vocabulary/IDs.
- Every new train/prepare invocation gets a unique folder under `outputs/` (or
  the `--output` parent). Archives/JSON use atomic replacement. Resume keeps the
  existing run unless `--output` requests a new folder. Checkpoints carry provenance.
- Backend preflight and regression tests; no change to the simulator or optimizer.
- The sweep has **75 causal configurations, 150 epochs per fit**: 45 alpha/LR/window
  combinations (0.1/1/3 × 0.0001/0.0003/0.001 × 2/4/6/8/12) at 8 layers/batch 64;
  24 matched depth/batch comparisons; six uniform/balanced sentence-pool comparisons
  at 5k/20k/50k sentences. The 69 other runs retain the full curated pool.
- The sweep now draws **5,000 training examples with replacement per epoch**,
  independently of batch size, for the same **150 epochs per fit**. The final
  partial batch is retained (313/79/20 updates for batch sizes 16/64/256).
  `--samples-per-epoch` configures this; legacy runs still default to full passes.
- Fixed monitoring samples of up to 2,048 train/validation examples keep epoch
  reporting bounded. Their RNG is independent of training draws. Sampled-epoch
  checkpoints omit full-corpus embeddings but retain resumable optimizer/RNG and
  splits. Full validation is scored once per completed fold; the full held-out
  test is scored after refit. Final CV metrics use full folds, while curves use
  explicitly labelled monitoring samples.
- An outer **80% development / 20% held-out test** split by sentence text, then
  **five-fold CV inside development** (approximately 64/16/20 train/val/test).
  Duplicate sentences and their windows stay together. Each fold starts with the
  same seeded weights and fresh optimizer; test examples are absent from folds.
  A fresh final fit trains on all development examples, then scores test once.
  Each configuration therefore runs six fits; the full matrix has 450 fits.
- Fold train/validation histories, per-epoch CV mean/std, final held-out metrics,
  split indices, and test embeddings are saved separately. `resume-cv RUN_DIR`
  resumes unfinished fits from saved epochs and skips completed fits. Legacy
  archives/`continue` remain compatible. Plots distinguish validation from test,
  and progress now prints BCE to eight decimal places.
- Encoded-state retention is bounded by `--state-cache-mib` (default 256 MiB).
  Small corpora retain the device cache; large corpora use a host LRU and transfer
  simulation-sized batches. Encoding, simulator kernels and optimizer are unchanged.
- Slurm layout: **15 jobs × 5 sequential experiments**. Arrays of ten then five
  tasks use `aftercorr`; at most ten jobs run at once. A failed group stops its
  chain. Submit using `bash scripts/submit_sweep.sh` after `uv sync --locked`.

## Earlier pipeline validation (before the longer CV sweep)

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

## Validation of the longer CV sweep (before sampled epochs)

- `.venv/bin/python -m pytest -q` with actual MPS access: **72 passed, 17 CUDA
  cases skipped**. Tests cover sentence-group isolation, every development
  example validating exactly once, fresh fold initialization, test scored only
  after refit, and no repeat scoring on a completed resume.
- Interrupting a fold after epoch 1 and resuming to epoch 2 gives identical CPU
  weights, histories and RNG state to uninterrupted CV. Completed fold archives
  remain intact. Legacy continuation retains its evaluation policy.
- Streamed and resident contexts agree within CPU/GPU tolerances at 14 qubits,
  including parallel SPSA and two-epoch training; the forced small cache stays
  within its capacity. No simulation or optimizer changes were necessary.
- Real-data MPS checks completed for experiment IDs **30 (causal, 2 layers)** and
  **109 (CBOW, 64 layers)** using
  `scripts/training_sweep.py --experiment-id ID --device mps --max-sentences 16
  --epochs 2 --output outputs/cv-validation`. Both used all three sources and the
  full vocabulary; 117 causal / 133 CBOW token examples, with six fits each.
  Checked finite archives, complete histories, model/checkpoint weight equality,
  split coverage, and fold/refit plots. Evidence: `outputs/cv-validation/validation.json`.
- Refit training BCE moved from 0.70164428 to 0.70124904 (causal), and from
  0.68197415 to 0.68191256 (64-layer CBOW). Exact-word accuracy stayed zero in these
  small checks; successful execution does not establish useful predictive accuracy.
- Ruff formatting/lint, Pyright, shell syntax, `git diff --check`, and the updated
  training skill validator passed. Full-data 150-epoch fits and CUDA execution
  remain unmeasured locally; the cluster preflight includes the new CUDA CV/cache tests.

## Validation of sampled epochs

- `.venv/bin/python -m pytest -q` with actual MPS access: **88 passed, 23 CUDA
  cases skipped**. The sweep manifest keeps 150 epochs and a 5,000-example epoch
  budget for every batch size; all full-data pools remain uncapped.
- Tests count training draws/optimizer updates including partial batches, check
  that draws stay inside the training partition, reject hidden full-corpus
  prediction during sampled epochs, and verify stable monitoring subsets.
  CPU/MPS resumed weights and histories match uninterrupted sampled training.
- Sampled CLI CV, completed/partial resume, and standalone continuation preserve
  sampling settings on CPU/MPS. Full final validation scores are checked against
  direct inference; final test exports cover every held-out example, not only the
  monitoring sample. The cluster preflight includes these CUDA cases.
- Real-data MPS experiment IDs **30 (causal, 2 layers)** and **108 (CBOW, 8 layers)**
  completed two sampled epochs per fold/refit with 16 sentences, 33 draws per
  epoch, and 16 monitoring examples per split. Used the complete 11,428-word
  vocabulary; checked finite checkpoints, optimizer step counts, all five full
  validation reports, and 22/25 complete held-out test embeddings.
  Command: `scripts/training_sweep.py --experiment-id ID --device mps
  --max-sentences 16 --epochs 2 --samples-per-epoch 33 --eval-examples 16
  --output outputs/sampled-cv-validation`.
  Evidence: `outputs/sampled-cv-validation/validation.json`.
- Ruff formatting/lint, Pyright, shell syntax, diff checks, and the training skill
  validator passed. No actual CUDA timing or 150-epoch cluster fit is claimed.

## Focused causal sweep update

- Replaced source/cleaning ablations with 45 alpha/LR/window, 24 depth/batch, and
  six sentence-sampling comparisons. All 75 configurations use phrases + cleaned,
  causal training, 150 epochs and 5,000 draws per epoch. Job grouping is now 15
  groups of five, with arrays of ten and five tasks and matching dependencies.
- Measured the two active sources: 1,000 phrase rows and 244,280 cleaned rows;
  202,172 deduplicated sentences, 1,785,128 tokens and 10,864 vocabulary words.
- Validation is limited to the 12 targeted data/sweep tests, Ruff and shell syntax;
  no repeat training/backend runs for this manifest/source-selection change.

## Cluster handoff

The user supplied an A100-SXM4-80GB with a 9,728 MiB MIG allocation and driver
580.173.02. Retained host states are now bounded to 256 MiB by default; one
two-lane, 256-context float32 SPSA buffer at 14 qubits uses 64 MiB, excluding other
intermediates/library overhead. Host corpus/example arrays, embeddings, and archive
serialization still scale with data size. No CUDA memory peak or cluster execution
has been measured locally. Re-encoding evicted contexts can make full-data training
very slow. Regular epochs now have fixed sampling/monitoring budgets; preparation,
checkpoint serialization and final full evaluations still scale with the corpus.
There is no claim that five long CV configurations fit within 12 hours.
Checkpointing is per completed epoch, so a timeout can lose part of an epoch.

Use the checkout's locked environment, retain Slurm's `CUDA_VISIBLE_DEVICES`,
and use the confirmed `gpu:1g.10gb:1` GRES for `prod10` (the batch script now
requests it by default). Each
scheduled job checks real CUDA inference, training, CV, bounded-cache parity, and
checkpoint resume before its five configurations. Slurm retains 12-hour limits,
four CPUs, and logs/errors under `logs/`. [README.md](README.md) contains the
commands, matrix, and history-rewrite checkout instructions.

Keep subsequent edits minimal and driven by observed failures. The batching
implementation already passes the reference comparisons. Do not rewrite it to
address an unconfirmed CUDA problem; inspect the cluster preflight failure first.
