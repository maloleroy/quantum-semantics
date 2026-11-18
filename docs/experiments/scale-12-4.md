# 12 vocabulary qubits and 4 ancilla qubits

Declared 17 November 2026 at the user's request. This is a single-seed scaling
and training feasibility experiment, not a controlled demonstration of a benefit
from more vocabulary or ancilla qubits. Completed results are below.

## Fixed protocol

- Model: `adaptive`, direct Born word measurement, no classical vocabulary head.
- Vocabulary: 4,096 byte-BPE outcomes (12 qubits); ancilla dimension 16 (4 qubits).
- Two Hamiltonian layers, rank 128, phase scale 1, recent window 16, context 128.
- Seed 11; float64/complex128; Adam at constant .003; gradient clipping at 5.
- Thirty epochs, 524,288 training examples per epoch, batch 512: 30,720 updates.
- NLL loss and validation NLL checkpoint selection; factor normalization disabled.
- Pinned WikiText-2 raw snapshot from `data/wikitext-2-raw-v1/manifest.csv`.
  Train tokenizer/statistics on training articles only. Preserve official article
  partitions, normalized overlap guard, lossless byte encoding and UNK policy.
- Prepare 524,288 train, 4,096 validation and 8,192 test targets using the existing
  deterministic preparation. Test is stored but never scored by this job.
- Report token NLL, bits per byte, sampling success and argmax separately. Different
  vocabulary token NLL is not directly comparable to previous 512-token runs.

## Execution

Isolated DGX checkout: `~/quantum-semantics-scale-12-4-20261117`.
Generated runner: `output/scale_12_4/tools/run.py`; execute with
`uv run python -m output.scale_12_4.tools.run` from the checkout root.

Use one `prod10` A100 `nvidia_a100_1g.10gb` slice, four CPU support threads,
24-hour allocation. An actual CUDA preflight uses the full model and batch,
resident training data, Adam moments and a saved-best model copy. Six optimizer
steps verify CUDA probabilities, finite CUDA gradients and changed weights;
measure the last four steps and a validation forward pass. Discard the preflight
model and optimizer and reinitialize seed 11 before the scientific fit.

Record peak allocated/reserved VRAM and project all 30,720 updates, per-epoch
validation and final scoring. Require a projection below 22 hours before training.
If this allocation fails memory or time requirements, retain the scientific
configuration and document a justified `prod80` full-A100 allocation instead.
No automatic batch, rank or update-budget reduction.

Record source/dataset hashes, GPU metadata, preflight JSON, live metrics CSV and
backward-pass progress JSON. Final weights and metric arrays use NPZ; final
histories use CSV. Selection is locked before any future held-out evaluation.
Start the local `dgx start` notifier after submission and stop work after
verification that CUDA training is ongoing, as requested.

## Submission

Slurm job **11711**, `qcse-12plus4`, submitted on 17 November at 15:35 DGX
time to `prod10` with one 10 GB slice, four CPUs and 15 GiB host memory.
Initially queued for resources; all seven 10 GB slices were occupied.
The local completion notifier is active. Source corpus hashes are verified
against the pinned manifest during data preparation. Project tests run inside
the GPU allocation before preflight. Data preparation can run ahead of the
allocation under a shared setup lock; model computation runs only on CUDA.

The user subsequently requested stopping once preparation is complete even if
the job remains queued. At handoff, job 11711 is pending resources; GPU tests,
preflight and scientific training have not yet run. The queued launcher performs
these gates automatically. Data-only tokenizer preparation is already in progress;
the shared lock prevents duplicate preparation if the GPU job starts meanwhile.

## Setup failure and resubmission

Job 11711 started at 15:53:24 DGX time on 17 November and failed after 15 seconds
with exit 1:0. The GPU test suite reported 96 passes and 62 setup errors because
`data/phrases.csv`, required by its fixtures, had not been copied to the isolated
checkout. The experiment's GPU preflight and scientific training never started;
this failure provides no evidence about model memory or throughput.

Data preparation separately completed. The 4,096-token dataset SHA-256 is
`9b499cfde14ff1e83f3ffcf7fa819516703b1896dd60313361ca8a8427f38544`.
Copied the missing fixture, refreshed the input hash manifest, and resubmitted
the unchanged protocol as job **11716**. It is pending resources at handoff;
the same test and resource gates will run automatically. The local notifier is
active. No test scoring or scientific optimizer updates have occurred.

## Remove the sentence corpus dependency

At the user's request, delete the old sentence CSV from the repository and this
experiment's checkout. Tests now generate a tiny temporary text corpus through
a shared fixture and no longer require a research dataset. `prepare` requires
an explicit `--data` argument; usage examples select WikiText.
Temporarily hold queued job 11716 while synchronizing these changes, then release
the same job with its original WikiText dataset, model and training budget.

## CUDA test failure, verified correction and relaunch

Job 11716 started at 16:16:26 DGX time and failed after 13 seconds, exit 1:0.
The suite passed 157 tests and failed the `prior` and `prior_position` CUDA
checkpoint tests on argmax accuracy arrays. Their probability, gradient and
checkpoint comparisons passed. The generated corpus produces tied priors;
identity dynamics and device roundoff can split those ties, changing the exact
tie-count accuracy from 1/2 to 1. Scientific training and its resource preflight
had not started.

The cross-device test now exercises nonzero learned dynamics for every prior
model and explicitly checks a margin above 1e-10 before comparing argmax scores.
A separate CUDA test checks analytically specified exact ties. Production models,
scoring rules, tolerance thresholds and experiment settings are unchanged.

Before submitting another training job, verification job **11726** ran the full
suite and actual 12+4 preflight on one `prod10` 10 GB slice. It completed with
exit 0:0 in 23 seconds: **160 tests passed**, all six full-size Adam updates
completed with finite CUDA gradients and changed weights. The measured model has
35,520,800 parameters; peak allocated/reserved memory was **6.123/6.811 GiB**
on the 9.5 GiB usable slice. Mean measured update time was **0.6845 seconds**;
projected fit time including scoring was **6.04 hours**, an estimate rather than
a completion guarantee. Preflight weights and optimizer are discarded.

The unchanged scientific fit is resubmitted as **11727**. Its launcher repeats
all CUDA tests and resource checks before fresh seed-11 training. Original
logs and verified preflight metadata are retained locally under
`output/scale-12-4/retrieved/`. The completion notifier is active.
At handoff, job 11727 remains pending scheduler allocation; scientific training
has not started. This is distinct from the successfully completed verification.

## Completed results — audited 18 November 2026

**The 16-qubit run fits one 10 GB slice and improves NLL and argmax over a
training-count bigram baseline, but does not establish better coherent generation
or a sampling advantage over bigram.** No matched neural baseline or other
ancilla/vocabulary configuration was fitted in this experiment.

Slurm **11727** completed with exit **0:0** in **5 h 53 min 38 s**. All 160 tests
passed again before preflight and scientific training. The fit itself took
21,188.65 seconds and completed all **30,720 updates / 15,728,640 row exposures**.
The run has **35,520,800 parameters**, with peak allocated/reserved CUDA memory
reported as **6.123 / 7.059 GiB** across preflight and training on a 9.5 GiB usable
A100 MIG slice. The observed runtime agrees with the approximately six-hour
preflight projection; it is simulator throughput, not quantum hardware speedup.

Dataset, all source files and selected-weight hashes verify. The 31 history rows
cover epochs 0–30, with batch 512, 524,288 examples per epoch and constant .003
learning rate throughout. NLL selection locks **epoch 19**, after 19,456 updates.
The selected weights are finite. GPU analysis job **11815** completed with exit
0:0 in 1 min 42 s, reproducing every saved validation array with **zero maximum
absolute difference**. No optimizer updates, checkpoint edits or test scoring
occurred during analysis. The original held-out test remains unscored.

### Same-data validation

All rows below use the same 4,096 targets from all 60 official validation articles
and the same 4,096-token tokenizer. Count distributions use only training
statistics. The modal control always emits token 33, the single ASCII space,
chosen from the training unigram. Its unsmoothed NLL is infinite because every
other target has zero probability.

| Predictor | Token NLL ↓ | Bits per byte ↓ | Sampling S ↑ | Argmax ↑ |
|---|---:|---:|---:|---:|
| 12+4 adaptive quantum | **3.17362** | **2.10859** | 37.431% | **49.365%** |
| Training bigram | 3.41465 | 2.26874 | 37.311% | 44.971% |
| Training unigram | 4.64186 | 3.08411 | 16.891% | 41.089% |
| Always training-modal space | ∞ | ∞ | **41.089%** | 41.089% |

Against bigram, quantum NLL improves by **−0.24104**, with a post-hoc paired
article-bootstrap 95% interval **[−0.28903, −0.19160]**. Argmax improves by
**4.395 percentage points [3.721, 5.060]**. Sampling improves by only
**0.120 points [−0.197, 0.434]**, which does not establish a sampling improvement.
Always emitting space scores higher on S than the quantum model, while providing
a much worse distribution for NLL. These are distinct objectives.

Intervals use 2,000 resamples of the 60 validation articles with seed 20261118.
They are conditional on this selected seed-11 checkpoint and reused validation
set; they do not include training-seed or model-selection uncertainty. They are
exploratory diagnostics, not an independently confirmed selection rule.

Whitespace comprises **1,713 / 4,096 targets (41.82%)**. On the remaining 2,383
targets, quantum reaches **4.750% sampling / 14.645% argmax / 5.26054 NLL**, versus
bigram **3.113% / 7.428% / 5.68944**. The non-whitespace NLL difference is
**−0.42890 [−0.50714, −0.34818]**. Thus improvement beyond whitespace is present,
but absolute non-whitespace sampling remains low. Quantum is slightly worse than
bigram on whitespace NLL and sampling, offsetting much of the aggregate S gain.
Whitespace means a nonempty token piece whose decoded text is entirely whitespace;
it is broader than the single-space modal control.

Evidence: [validation](scale-12-4-validation.csv), [paired intervals](scale-12-4-paired.csv),
[buckets](scale-12-4-buckets.csv), [bucket intervals](scale-12-4-bucket-paired.csv).

### Learning and comparison with the smaller register

Selected training NLL is **2.59196**, versus validation **3.17362**: a gap of
**0.58165**. Validation plateaus after the early epochs and is best at epoch 19;
epoch 30 is worse at **3.19825 NLL**, even though online training loss keeps
falling. This is consistent with a generalization limitation; it does not support
assuming that extending this run alone will improve generation. Epoch-30 sampling
and argmax are slightly higher than at the selected epoch, but NLL was the declared
selection criterion. Do not retroactively select a checkpoint for another metric.
The [history CSV](scale-12-4-history.csv) preserves all checkpoints' scores.

For descriptive context, the earlier 512-token, 9+1 adaptive constant-rate fit
reported **2.43222 bits per byte**, versus **2.10859** here, approximately 13.3%
lower. Token NLL and token-level accuracies are not comparable across these
tokenizers. Even bits per byte here comes from separately sampled token targets,
not paired common byte spans. Vocabulary, byte coverage of the 128-token context,
training target positions, auxiliary width and parameter budget all change;
parameters increase from 751,892 to 35,520,800 (about 47.2 times). Consequently,
this is not an isolated benefit of extra ancillas or a matched-size comparison.
No neural competitor with the new vocabulary and parameter budget was trained.

### Unfiltered generation

Before looking at the generated output, analysis fixed the earlier study's three
prompts, seeds 20261030–20261032 and 64 new tokens per prompt. Both direct sampling
and simulator argmax are retained in the [complete sample CSV](scale-12-4-samples.csv)
(its `text_escaped` column uses JSON string escapes for controls and newlines),
with token arrays in `output/scale-12-4/analysis/sample-tokens.npz`. There is no
temperature, rejection, replacement of bad outcomes or retry. Argmax is a
separate simulator diagnostic requiring distribution access, not a one-shot
quantum generation procedure. Repeating the first sampled continuation exactly
reproduces its tokens on CUDA.

Sampled continuations remain incoherent. The first begins
“The history of its winning almost essrews , butvcomeJuan” and contains one
U+0018 control character. The third contains invalid UTF-8 rendered as two
replacement characters. All argmax continuations are valid UTF-8 but degenerate
into repeated phrases such as “of the first of the first” and “of the of the”.
No UNK/non-text outcome occurred. More qubits have not yielded useful long-form
generation in this screen.

### Interpretation and next comparison

Retain this as a successful **simulation scaling** result and an exploratory
NLL/argmax improvement over count baselines. Do not claim coherent generation,
quantum advantage, or a causal gain from four ancillas. A useful next experiment
would hold the 4,096-token dataset fixed, compare one versus four ancilla qubits,
and include a strong matched neural baseline. Use common byte-span evaluation
for a controlled comparison with the old tokenizer. No further training was
launched as part of this analysis.

The learning plot and numerical audit are under `output/scale-12-4/analysis/`.
Authoritative weights remain in the isolated DGX run recorded above; the local
checkpoint copy was moved to temporary storage for auditing when the laptop disk
filled. The analysis tools are retained under `output/scale_12_4/tools/`.
