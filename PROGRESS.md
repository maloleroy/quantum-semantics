# Progress — 2026-09-14

## Cluster sweep archive review — 2026-09-15

Reviewed `/Users/ethan/Documents/Ecole/3A/Filière recherche/Papers/sweep.zip`,
the cluster handoff for the 45-configuration production QCSE sweep. The archive
contains 44 completed two-fold 64/16/20 CV runs and experiment 039 still running
after epoch 0 of its first fold; 54 older non-CV/legacy attempt directories were
excluded. The earlier BCE-based report selected experiment 031 (alpha 1,
learning rate 0.0003, window 2, 8 layers, batch 16): validation BCE
0.545285 ± 0.001124 and held-out test BCE 0.541256. Batch 16 receives 313 Adam
updates per 5,000-draw epoch, versus 79 for batch 64 and 20 for batch 256, so the
apparent batch advantage is update-count confounded.

The primary reanalysis now ranks by mean full-validation word top-1, defined for
this 14-bit decoder as exact agreement of all thresholded bits with the target
word ID. It selects experiment 035 (64 layers, batch 64, window 2) at 1.121%
validation top-1 and 1.068% held-out test top-1; experiment 031 remains the
secondary BCE winner. Training samples 5,000 example IDs with replacement from
the training partition, so repeats are allowed, while sentence-grouped split
checks found no sentence-level leakage.

Added `scripts/cluster_sweep_report.py`, which validated all 44 completed archive
layouts, split/group isolation, fold aggregation, fixed-monitoring histories and
recomputed held-out metrics from exported probabilities. All 44 passed; the ZIP
integrity check passed. It also generated detailed and short reports, a CSV/JSON
aggregate and 22 plots under `results/cluster-sweep/`. The local full suite was
rerun: **79 passed, 46 skipped** because CUDA and MPS are unavailable; Ruff,
Pyright, shell syntax, diff checks and the CPU 14-qubit/64-layer backend preflight
passed. See [TEST_INVENTORY.md](TEST_INVENTORY.md) for the complete ledger and
explicit remaining gaps.

## Unified 50-epoch production targets — 2026-09-15

The active launch defaults are now 50 epochs for all three experiment families:
the 45-configuration QCSE causal/GPT-like sweep, the ten-configuration semantic
encoding sweep with three seeds per configuration, and the QCSE/classical quantum
attention runs plus their attention cross-validation sweep. The 30-to-50 extension
gate is removed for new semantic jobs because every configuration now targets 50
epochs directly. Unit and smoke tests retain short explicit epoch counts for speed;
they are not production training targets. Existing 10-, 25- and 30-epoch result
folders remain historical evidence and are not silently relabelled.
The DGX launch gives each QCSE configuration its own 12-hour job; the old grouped
wrapper nearly filled that limit for the deepest five-configuration group. Check
the site's approved wall time before launch, but no five-configuration grouping is
now required.

## Local validation rerun — 2026-09-15

Reran the complete local validation battery from the repository root. The command
`.venv/bin/python -m pytest -q -rs` completed with **79 passed, 46 skipped in
18.78 s**; the 46 skips are the CUDA/MPS parameterizations unavailable on this Mac.
Also reran `.venv/bin/ruff check .`, `.venv/bin/pyright`,
`bash -n scripts/*.sh slurm-*.sbatch`,
`.venv/bin/python -m compileall -q src scripts tests`, `git diff --check`, and
`.venv/bin/python scripts/check_backend.py --device cpu`; all passed. The CPU
14-qubit/64-layer backend preflight reported maximum absolute errors of
`3.3306690738754696e-15` against both Qiskit and sequential evaluation. No CUDA
execution is claimed from this host.

## 50-epoch local execution audit — 2026-09-15

After setting the production targets to 50, completed capped CPU end-to-end runs:
one causal QCSE configuration with two validation folds plus a development refit
(3 fits), all ten semantic configurations with seeds 42/43/44 (30 fits), and all
five attention-CV ablation setups with two folds plus refit (15 fits). Every one of
the **48 fits** reached epochs 0–50 with finite histories and final held-out scoring;
the causal saved artifacts also passed an explicit outer/inner sentence-group
disjointness check. The audit used 16 sentences; QCSE/attention used 17 draws per
epoch to keep CPU runtime bounded, while semantic used its configured 5,000 draws.
It therefore validates execution and protocol handling, not full-corpus 50-epoch
quality or CUDA performance. See [TRAINING_AUDIT.md](TRAINING_AUDIT.md).

The semantic and attention sweep selectors now choose by validation retrieval top-1
first, then top-5, lower cross-entropy and earliest epoch as deterministic tie-breaks.
BCE/cross-entropy remain the differentiable training objectives and secondary
diagnostics. The selector policy has a direct regression test.

## Bounded final evaluation — 2026-09-15

Final validation and test inference now use fixed, seeded, no-replacement samples
of up to 10,000 example IDs from each already-separated split. Per-epoch monitoring
remains bounded at 2,048 examples for QCSE/semantic runs and 512 for attention.
The full sentence-grouped 80/20 outer split and inner validation folds are unchanged;
the cap affects inference/reporting cost only. `--final-eval-examples 0` restores
exhaustive final scoring. The semantic and attention runners record both the source
split size and evaluated sample size, while QCSE persists sampled IDs in `splits.npz`
and exports only those test predictions. Repeated windows from a sentence may appear
within one evaluation sample, but no normalized sentence group can cross a split.

The proposed 98/1/1 alternative was not adopted: it would change the estimand and
make the requested repeated-fold comparison non-comparable. The implementation was
covered by deterministic/no-replacement sampling tests, capped CV resume tests, and
the complete local suite rerun after the change (**84 passed, 46 skipped**).

## DGX A100 10 GB launch preparation — 2026-09-15

Prepared independent Slurm launches for the site's `prod10` partition (DGX A100
10 GB MIG) with `gpu:nvidia_a100_1g.10gb:1`, 50 epochs, four CPUs and no
`--dependency` options.
The QCSE sweep is now one configuration per array task (`0-44%10`) rather than five
sequential configurations inside each task. Semantic uses `0-9%10`, and attention
uses `0-1%2`; each task performs its own CUDA/backend preflight.

The all-family submitter is `scripts/submit_dgx_a100_10gb.sh`:

```bash
uv sync --locked
bash scripts/submit_dgx_a100_10gb.sh
```

It submits three independent arrays and never waits for one family before submitting
another. The batch files carry the confirmed partition/GRES directives, while the
submitters pass each array specification exactly once. If the site names resources
differently, pass `sbatch` options through the submitter, for example
`bash scripts/submit_dgx_a100_10gb.sh --partition=other --gres=gpu:other:1`.
The individual QCSE submitter is `scripts/submit_sweep.sh`; it also submits one
experiment per task. Nothing was submitted from this workstation.

## Full-corpus 5-fold attempt — 2026-09-15

Started the requested reference run locally with 5 exhaustive development folds,
50 epochs per fit, 5,000 replacement draws per epoch, a development-only refit and
one held-out test evaluation. It prepared the full 202,172-sentence / 1,582,956-
example corpus and completed fold 1 through epoch 1 before interruption. The first
fold took approximately 4.5 minutes per sampled epoch on this eight-core CPU,
projecting roughly 20–25 hours for the six 50-epoch fits before exhaustive final
validation/test scoring. The last complete checkpoint is retained under
`outputs/full-cv-50/`; no partial result was reported.

## Quantum attention comparison

Implemented on `qcse-classical-attention`, based on `semantic-decoder-prototype`
at `e8376c0`. Fixed QCSE context encoding and learned classical amplitude encoding
feed the same single-head quantum Q/K/V circuits and causal bilinear attention.
The postselected Torch action matches a small Qiskit block-encoding reference.
The register defaults to four qubits for both inputs; this is separate from the
production vocabulary-sized binary decoder. Details and commands: [ATTENTION.md](ATTENTION.md).

- Reused the semantic runner's grouped splits, complete shared vocabulary,
  replacement sampling, atomic checkpoints, resume and word exports. Added a
  local comparison launcher and a two-job `prod10` MIG array with CUDA preflights.
- Actual local runs: `.venv/bin/python scripts/semantic_experiment.py --model attention
  --encoding ENCODER --datasets phrases --epochs 5 --samples-per-epoch 5000
  --output outputs/attention-validation/ENCODER`, for `qcse` and `classical`, then
  `--resume RUN --epochs 10 --evaluate-test` for each. All 1,000 phrases and the
  full 10,864-word vocabulary were retained; vocabulary, splits and sampler states matched.
- Epoch-10 validation CE: **5.520 QCSE / 4.995 classical**. Test CE:
  **5.677 / 5.150**. These are single-seed pipeline results; the classical encoder
  has additional learned embedding parameters. Both cosine top-1 scores remained
  below the frequency baseline. Generated runs are ignored by Git.
- **12 focused tests passed**: `pytest -q tests/test_attention.py
  tests/test_attention_pipeline.py tests/test_semantic.py` (11 cases), then
  `pytest -q tests/test_attention_pipeline.py -k legacy` after adding the legacy
  checkpoint case. Tests cover Qiskit encoding/gate/postselection parity, finite nonzero
  gradients, causal/padding masks, reload and bit-exact CPU resume including Adam
  and sampling RNG. Existing semantic-model tests and legacy checkpoint resume pass.
  Targeted Ruff/Pyright, shell syntax and `git diff --check` also passed.
- `scripts/check_backend.py --device cpu` passed at 14 qubits/64 layers with
  maximum Qiskit error `3.33e-15`. The comparison shell launcher completed both
  encoders with 17 draws and batch 8; its local invocation set
  `UV_CACHE_DIR=/tmp/attention-uv-cache` for the restricted environment.
- CUDA and Apple MPS were unavailable; no GPU or scheduler execution is claimed.

## Circuit contribution ablation

Repeated the same 25-epoch, two-fold protocol with a matched `classical-no-circuit`
setup: learned classical embeddings and overlap/readout retained, trainable quantum
Q/K/V circuits disabled. Its CV validation CE was **4.6414 ± 0.0401** and test CE
**4.6504**, versus **4.6028 ± 0.0235** and **4.6084** for the one-layer circuit.
The circuit slightly improved cross-entropy but reduced test cosine top-1/top-5 from
15.04%/33.72% to 9.94%/30.48%. This run does not show a retrieval advantage from the
quantum circuit; the learned classical encoder carries most of the signal. Updated
plots include fold-SD bands and error bars in `results/attention-cv-25-ablation`.

## 25-epoch attention comparison

Ran `scripts/attention_cross_validation.py` with both named datasets, balanced
sampling, `--max-sentences 5000`, `--epochs 25`, and `--samples-per-epoch 5000`.
The cap retains all 1,000 phrase rows and 4,000 cleaned rows. It produced four
setups (QCSE/classical × one/two quantum layers), each with two independent 20%
validation shuffle splits inside the 80% development data and a fresh development
refit before one held-out test evaluation. Raw run folders are ignored under
`outputs/attention-cv-25`; tracked plots/report are in
`results/attention-cv-25`.

| Setup | CV validation CE mean ± SD | Test CE | Test cosine top-1 / top-5 |
| --- | ---: | ---: | ---: |
| QCSE, one layer | 5.6910 ± 0.0311 | 5.7268 | 4.35% / 18.07% |
| Classical, one layer | **4.6028 ± 0.0235** | **4.6084** | **9.94% / 30.48%** |
| QCSE, two layers | 5.6932 ± 0.0578 | 5.7180 | 4.11% / 15.45% |
| Classical, two layers | 4.6085 ± 0.0326 | 4.6209 | 9.74% / 31.30% |

The classical input encoder is superior in this comparison at both depths. The
one-layer classical setup has the best mean validation cross-entropy; two-layer
classical has the highest test top-5. Full curated-pool and CUDA execution remain
unmeasured locally.

## Hyperparameter sweep

Ran eight configurations for 25 epochs each with both folds, 5,000 replacement
samples per epoch, both datasets and the same 5,000-sentence balanced cap. The
tested learning rates were `0.001`, `0.003`, and `0.01`; attention scales were
`0.01`, `0.05`, and `0.2`; one/two layers and the no-circuit control were included.
Every setup trained two independent folds and a fresh development refit; no fold
was discarded as “best”. Results and figures are tracked in
`results/attention-hyper-25`.

The best mean validation CE was **4.5048 ± 0.0293** for classical one-layer,
learning rate `0.01`, alpha `0.05`; its held-out test CE was **4.4105** with
**17.44%/38.27%** cosine top-1/top-5. Fold CEs were 4.4841 and 4.5255. The
`0.001` learning-rate setup was worse (**5.2538 ± 0.1376**). Alpha changes near
the `0.003` reference were small. The report includes fold-level scores and
error bars; selection uses the validation mean, not the test result.

## Earlier work

Branch: `semantic-decoder-prototype`, based on Claude's `corpus-training-sweep` at `d71b6dc`.
Reference `main`: `87fa9a7cef03522957d442485a9bcee35c749a71`.

## Brief review of Claude's completed changes

- Focused data/sweep checks: **12 passed** (`pytest tests/test_data.py tests/test_sweep.py -q`).
- Current sweep defaults are 45 causal configurations, 50 epochs per fit, and
  two independent 64/16/20 shuffle-splits followed by a refit on development data.
  Repeated validation sets can overlap; this is not exhaustive two-fold CV.
- Updated stale README, CLI help and training skill descriptions to match the code.
  Training behavior and the Slurm submission logic are unchanged by this review.

## Report-inspired semantic decoder experiment

- Added a separate `SemanticModel` and `scripts/semantic_experiment.py`: trainable
  word embeddings and angle encoder, a four-qubit fixed chain circuit, XYZ
  expectations and a tied word decoder trained with cross-entropy/autograd.
  Vocabulary IDs only address rows. The existing production model is unchanged.
- CPU runs completed **10 epochs, then resumed to 50**, on 96 repetitive sentences
  and all 1,000 phrases (full shared 10,864-word vocabulary), with 5,000 sampled
  examples per epoch and a single grouped 64/16/20 split. Final test scored once.
- Repetitive validation CE: 3.128 → 0.756 → 0.751 at epochs 0/10/50; final test
  top-1 62.4%. Phrases validation CE: 9.325 → 4.754 → 6.765; final test top-1
  18.5% versus a frequency-only baseline of 9.5%. Phrases overfits after epoch 10.
- Two focused model tests passed; Ruff and targeted Pyright passed. Both real
  checkpoint resumes and decoded word exports completed. New-model GPU execution,
  semantic benchmarks and a matched classical comparison remain unmeasured.
- Architecture, commands, exact output directories and results are recorded in
  [SEMANTIC_PROTOTYPE.md](SEMANTIC_PROTOTYPE.md). Generated outputs remain local.

## 25-epoch semantic ablations

- Four phrase runs used the same seed/split and 5,000 sampled examples per epoch:
  trainable ansatz, frozen ansatz, zero fixed ansatz, and decoder-only training.
- Cosine is now the primary retrieval metric. At epoch 25, the trainable-ansatz
  run reached 14.7% held-out cosine top-1 and 18.5% dot top-1; decoder-only
  reached 2.1% cosine and 8.4% dot, below the 9.5% frequency baseline. The
  decoder alone does not explain the result.
- Frozen and zero ansatz controls matched or slightly exceeded trainable ansatz;
  this setup does not establish ansatz usefulness. Validation CE and cosine
  retrieval were best around epoch 10 and worsened by epoch 25. Plot and full
  table: `SEMANTIC_ABLATIONS.md`.

## Dataset/setup matrix

- Ran full trainable, no circuit, no encoding/decoding, frozen circuit, frozen
  encoder/decoder and frequency baseline on phrases-only, cleaned-only and both
  sources for 25 epochs. Cosine top-1/top-5 are the only reported metrics.
- Both sources gave the strongest full model in this capped local check:
  **16.8% / 35.2%** test cosine top-1/top-5. The direct baseline scored
  **7.6% / 21.2%**; no encoding/decoding scored **2.7% / 13.2%**.
- Each local selection was capped at 1,000 curated sentences for speed; the
  vocabulary still used both complete active files. Full details and plots:
  [SEMANTIC_MATRIX.md](SEMANTIC_MATRIX.md).
- Added 25-epoch checks for LR 0.001/0.01, ansatz scale 0.01/0.2 and four layers.
  LR 0.01 gave the strongest tested test cosine top-5 (36.3%). Added a direct
  mean-embedding baseline (0.6% test cosine top-1) and frozen encoder/decoder
  pathway (7.9%); details are in `SEMANTIC_GRID.md`.

## Implemented pipeline

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
- The sweep has **45 causal configurations, 50 epochs per fit**: 27 alpha/LR/window
  combinations (0.1/1/3 × 0.0001/0.0003/0.001 × 2/4/8) at 8 layers/batch 64;
  12 matched depth/batch comparisons; six uniform/balanced sentence-pool comparisons
  at 5k/20k/50k sentences. The 39 other runs retain the full curated pool.
- The sweep now draws **5,000 training examples with replacement per epoch**,
  independently of batch size. The final
  partial batch is retained (313/79/20 updates for batch sizes 16/64/256).
  `--samples-per-epoch` configures this; legacy runs still default to full passes.
- Fixed monitoring samples of up to 2,048 train/validation examples keep epoch
  reporting bounded. Their RNG is independent of training draws. Sampled-epoch
  checkpoints omit full-corpus embeddings but retain resumable optimizer/RNG and
  splits. Full validation is scored once per completed fold; the full held-out
  test is scored after refit. Final CV metrics use full folds, while curves use
  explicitly labelled monitoring samples.
- An outer **80% development / 20% held-out test** split by sentence text, then
  **two independent shuffle-splits inside development** (approximately 64/16/20 train/val/test).
  Duplicate sentences and their windows stay together. Each fold starts with the
  same seeded weights and fresh optimizer; test examples are absent from folds.
  A fresh final fit trains on all development examples, then scores test once.
  Each configuration therefore runs three fits; the full matrix has 135 fits.
  Exhaustive k-fold CV remains available by omitting `--val-fraction`.
- Fold train/validation histories, per-epoch CV mean/std, final held-out metrics,
  split indices, and test embeddings are saved separately. `resume-cv RUN_DIR`
  resumes unfinished fits from saved epochs and skips completed fits. Legacy
  archives/`continue` remain compatible. Plots distinguish validation from test,
  and progress now prints BCE to eight decimal places.
- Encoded-state retention is bounded by `--state-cache-mib` (default 256 MiB).
  Small corpora retain the device cache; large corpora use a host LRU and transfer
  simulation-sized batches. Encoding, simulator kernels and optimizer are unchanged.
- Slurm layout is now **45 independent QCSE array tasks**, one experiment per task,
  array `0-44%10`; semantic and attention remain independent arrays. No scheduler
  dependencies or five-configuration sequential jobs are used. Submit all three
  families with `bash scripts/submit_dgx_a100_10gb.sh` after `uv sync --locked`.

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
and use the confirmed `gpu:nvidia_a100_1g.10gb:1` GRES for `prod10` (the batch script now
requests it by default). Each
scheduled job checks real CUDA inference, training, CV, bounded-cache parity, and
checkpoint resume before its five configurations. Slurm retains 12-hour limits,
four CPUs, and logs/errors under `logs/`. [README.md](README.md) contains the
commands, matrix, and history-rewrite checkout instructions.

Keep subsequent edits minimal and driven by observed failures. The batching
implementation already passes the reference comparisons. Do not rewrite it to
address an unconfirmed CUDA problem; inspect the cluster preflight failure first.
## 30-epoch full-trainable versus no-circuit sweep

Completed 18 local causal runs on `phrases.csv + cleaned_sentences.csv` (Tatoeba excluded), with 30 epochs, 5,000 replacement-sampled examples per epoch, and cosine top-1/top-5 evaluation. The local speed check used 1,000 curated sentences; the same runner accepts the full corpus for cluster jobs. Alpha, learning rate, layer count, and batch size were varied one at a time. The best full-trainable run used learning rate 0.01, alpha 0.05, two layers, batch 64 (20.2% / 39.6% test cosine top-1/top-5). The alpha 0.01, learning rate 0.003 run reached 19.4% / 36.4%. No circuit remained competitive and won the four-layer and alpha 0.2 comparisons. See `SEMANTIC_HYPERPARAMS.md` and `outputs/semantic-hyper-report/`.

Added optional inference-only Qwen references in `scripts/qwen_baselines.py`: Qwen2.5-0.5B next-token logits and Qwen3-Embedding-0.6B vocabulary cosine ranking. They do not fine-tune or alter QCSE checkpoints.

## Focused DGX semantic protocol

Prepared a ten-job independent Slurm array for the report-inspired causal decoder. Each job runs one configuration and three seeded sentence-grouped 64/16/20 repeats (seeds 42, 43 and 44) on the `gpu:nvidia_a100_1g.10gb:1` MIG resource. The manifest varies only the reference/no-circuit control, alpha, learning rate, context window, layer count and batch size; all runs use both active datasets, the full curated sentence pool, 5,000 replacement draws per epoch and cosine retrieval metrics. The first pass is 30 epochs. Continue to 50 only when the median `best_epoch` across the three repeats is 30; jobs have no scheduler dependencies. See `README.md`, `scripts/semantic_cluster_sweep.py` and `slurm-semantic-prod10.sbatch`.

The full 30-epoch matrix was also run locally with the same ten configurations and three repeats, using a 1,000-sentence cap because CUDA and MPS are unavailable here. Ten processes completed in 2m59s wall time (25m43s total CPU). Median best epochs were 4–26, so no configuration met the reproducible 50-epoch extension rule. The local helper is `scripts/run_semantic_local.sh`; full-pool timing remains an extrapolation.
