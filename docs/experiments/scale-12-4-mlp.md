# Matched MLP control for the 12+4 quantum instance

Declared 18 November 2026 at the user's request, after the quantum result was
known. Compare the existing `adaptive_mlp` control with the locked
[12+4 adaptive quantum fit](scale-12-4.md#completed-results--audited-18-november-2026).
Do not change hyperparameters in response to this control's validation scores.

## Fixed comparison

- Use the exact prepared 4,096-token, 128-context WikiText-2 byte-BPE NPZ from
  quantum job 11727, SHA-256
  `9b499cfde14ff1e83f3ffcf7fa819516703b1896dd60313361ca8a8427f38544`.
- Reuse training-only tokenizer/statistics, official article partitions, UNK
  policy and the identical 524,288 training / 4,096 validation / 8,192 test rows.
  The test remains unscored.
- Model `adaptive_mlp`, configuration size 4,096, window 128, layers 2,
  ancilla-size feature setting 16, rank 128, recent window 16, phase scale 1.
  All other configuration fields match the locked quantum fit.
- The MLP uses the same types of prepared real amplitudes, local controls and
  spectral shifts, with independently learned preparation/controllers. It uses
  a 65,680 → 480 → 4,096 network with Tanh and softmax. The feature
  setting does not imply that the classical network contains qubits.
- **35,463,168 trainable parameters**, versus quantum **35,520,800**: 57,632 fewer,
  a **0.1623%** difference from integer hidden-width rounding. Encoders/controllers
  count toward both totals. No padding parameters are added to equalize counts.
- Seed **11**, float64/complex128 feature preparation, Adam constant .003,
  gradient clipping 5, no dropout and no factor normalization.
- Thirty epochs, batch **512**, **30,720 updates / 15,728,640 row exposures**.
  With the identical source code, seed and dataset, NumPy's training-row shuffle
  schedule matches quantum. Initialization draws differ between architectures.
- NLL loss; choose minimum validation NLL, then maximum sampling success, retaining
  the earliest epoch on an exact tie. There are the same 31 validation checks.
  Report selected most-likely-token accuracy, sampling success, NLL and bits per
  byte separately. Do not select by accuracy after looking at the results.

This is a single-seed, equal-exposure comparison with almost equal parameter
count and matching input features and optimizer settings. It is not a three-seed
comparison or a claim that the chosen hyperparameters are optimal for either model.

## Execution and checks

Use an isolated checkout `~/quantum-semantics-scale-12-4-mlp-20261118`, copying
the quantum run's source snapshot and locked CUDA dependencies. Preserve the
original quantum checkpoint and run. Generated tools are in
`output/scale_12_4_mlp/tools/`; results go to `output/scale-12-4-mlp/`.

Start with one `prod10` A100 `nvidia_a100_1g.10gb` slice and four CPU support
threads. A short verification job must pass the full GPU/project suite and six
full-batch Adam updates on the actual MLP, checking finite CUDA probabilities,
CUDA gradients, changed weights and exact parameter count. Keep resident data,
Adam moments and a best-model copy in the memory measurement. Require projected
training and scoring below 22 hours before submitting a 24-hour training job.
Preserve the batch/update budget if a larger allocation is needed.

The runner checks model and training metadata against the original quantum
record, as well as the exact dataset hash. Discard all preflight updates and
reinitialize seed 11 before training. Record live validation CSV, backward-pass
progress, device and peak-memory metadata, source/dataset hashes, NPZ weights
and final CSV history. Activate the laptop `dgx start` completion notifier.
Stop work after confirming training is ongoing; if allocation remains queued,
the user's earlier instruction allows leaving it pending without waiting an hour.

## Verified and launched — 18 November 2026

Short verification job **11816** completed with exit 0:0 in 24 seconds. All
**160 GPU/project tests passed**. Six full-batch CUDA Adam updates verified
finite probabilities and gradients, changed weights and exactly 35,463,168
parameters. The runner's checks confirm the dataset hash, quantum source hashes
and matching model/training metadata. Mean measured update time was **0.1665 s**,
projecting **1.44 hours** including scoring. Peak allocated/reserved memory was
**2.988 / 3.781 GiB** on one 9.5 GiB usable A100 MIG slice. This is a simulator
resource estimate, not a result about prediction quality or hardware speedup.

At launch, full training job **11817**, `qcse-12plus4-mlp`, was running on `prod10`,
one 10 GB MIG slice, four support CPUs, 15 GiB host memory, 24-hour allocation.
It passed all 160 tests and the preflight again, discarded those updates, and
started the fresh seed-11 fit. At handoff, live progress confirms at least
**64 scientific backward passes**, finite CUDA gradients and peak reserved
memory **3.781 GiB**. Initial validation NLL is **8.30112** before training;
this is not a selected predictive result. The completion notifier is active.

Logs, progress and metadata are in the isolated remote checkout under
`output/slurm/` and `output/scale-12-4-mlp/`. Verification logs and preflight
metadata are also retained locally under `output/scale-12-4-mlp/retrieved/`.
Training results and the paired most-likely-token comparison remain pending.

## Completed comparison — audited 18 November 2026

**Most-likely-token accuracy is essentially tied: quantum 49.37%, MLP 49.27%.**
The MLP clearly improves expected sampling success, including on non-whitespace
targets. The small aggregate NLL difference is unresolved. Neither model produces
coherent long continuations in the fixed sample set. This is one seed, not a
replicated quantum/classical ranking or evidence of quantum advantage.

### Completion and audit

Training job **11817** completed with exit **0:0** in **1 h 23 min 4 s**; fit time
was **4,959.32 seconds**. All **30,720 updates / 15,728,640 row exposures** completed.
Peak allocated/reserved CUDA memory was **2.991 / 3.781 GiB** across preflight and
training. Both neural fits used the same 10 GB MIG profile and four support CPUs.
Quantum fit time was 21,188.65 seconds, about **4.27 times** the observed MLP fit
time, with peak reserved memory 7.059 GiB. These are observed simulator costs
from separate jobs, not a controlled throughput benchmark or quantum hardware
speed comparison.

The two dataset hashes, source snapshots, seed, model feature configuration and
training budgets match; only architecture kind changes. The parameter difference
is the declared 0.1623%. All 31 history checks, per-epoch example counts/rates,
NLL checkpoint selection and paired validation target/group/byte arrays audit.
Quantum selects epoch **19**, MLP selects epoch **4**, after respectively
**19,456 / 4,096 updates**. Both still completed their identical 30-epoch budget;
selected-checkpoint exposures naturally differ under the shared selection rule.

GPU analysis **11823** completed with exit 0:0 in 12 seconds. Loading the locked
MLP verifies its source and weight hashes; model parameters are finite. Its
validation arrays reproduce with maximum absolute error **1.60e-14**, within
the unchanged 1e-10 tolerance. Sample repeatability and unchanged weights also
verify. No test-set scoring, fitting, calibration or checkpoint edits occur in
analysis. The authoritative checkpoint remains on the DGX; local retrieval omits
weights to conserve disk. The byte-identical local dataset copy was reused from
the quantum retrieval and checked against the shared hash.

An initial helper upload was interrupted; analysis-only job 11822 failed before
model loading because its module was absent. The transfer was repaired, and the
uploaded file's existence and compilation checked before submitting successful
job 11823. This did not affect training or either scientific checkpoint.

### Selected validation results

All rows use the same **4,096 targets across 60 official validation articles**,
the same 4,096-token vocabulary and the declared minimum-NLL checkpoint rule.

| Predictor | Parameters | Epoch | Most-likely accuracy ↑ | Sampling S ↑ | NLL ↓ | Bits/byte ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Quantum adaptive | 35,520,800 | 19 | **49.365%** | 37.431% | 3.17362 | 2.10859 |
| Matched MLP | 35,463,168 | 4 | 49.268% | **42.307%** | **3.14687** | **2.09082** |

For **MLP minus quantum**, paired article-bootstrap 95% intervals are:

| Metric | Difference | 95% interval | Interpretation |
|---|---:|---:|---|
| Most-likely accuracy | −0.098 percentage points | [−1.037, +0.866] | Unresolved |
| Sampling success | +4.876 percentage points | [+4.447, +5.286] | Favors MLP |
| NLL | −0.02674 | [−0.06280, +0.01144] | Unresolved |

The accuracy lead is only **four correct targets**: 2,022 quantum versus 2,018
MLP. It does not establish a quantum advantage. Intervals use 2,000 paired
resamples of articles, seed 20261118. They are post-hoc and conditional on these
selected checkpoints and the reused validation set. They omit training-seed,
hyperparameter-search and checkpoint-selection uncertainty.

The MLP's sampling S also exceeds the same-data always-space score of 41.089%
numerically. Most-likely accuracy, expected sampling success and NLL measure
different properties and must remain separate.

### Whitespace explains the aggregate accuracy tie

Whitespace is defined identically to the quantum analysis: a nonempty token piece
whose decoded text is entirely whitespace. There are 1,713 whitespace targets
and 2,383 other targets.

| Target bucket | Quantum accuracy | MLP accuracy | Quantum sampling | MLP sampling |
|---|---:|---:|---:|---:|
| Whitespace | **97.665%** | 94.980% | 82.894% | **89.088%** |
| Non-whitespace | 14.645% | **16.408%** | 4.750% | **8.679%** |

On non-whitespace targets, MLP accuracy improves by **1.762 points
[0.409, 3.016]**, and sampling by **3.929 points [3.369, 4.466]**. Quantum's
46 extra correct whitespace targets offset MLP's 42 extra correct other targets,
leaving the four-target aggregate quantum lead. Non-whitespace NLL is similar:
quantum **5.26054**, MLP **5.24141**, difference **−0.01913 [−0.07779, +0.04684]**.
The small aggregate accuracy difference therefore masks an MLP benefit on the
harder non-whitespace subset; absolute performance there remains poor.

Evidence: [validation](scale-12-4-mlp-validation.csv),
[paired intervals](scale-12-4-mlp-paired.csv), [buckets](scale-12-4-mlp-buckets.csv).

### Learning and generation

The MLP's validation NLL bottoms out at epoch 4 (**3.14687**) and then rises to
**9.13896** at epoch 30. Meanwhile, online training NLL falls from **2.60565** in
the fifth training epoch to **0.28721** in the thirtieth. This is a strong
overfitting pattern, not a failed or unfinished fit. Selected training-pool NLL
is **2.48534**, versus validation **3.14687**. All reported comparisons use the
locked epoch-4 checkpoint; the final epoch's higher S does not replace the
declared NLL selection. The [history](scale-12-4-mlp-history.csv) preserves every
epoch, and `output/scale-12-4-mlp/analysis/comparison.png` shows both learning curves.

Generation uses the same three fixed prompts, seeds 20261030–20261032 and 64
new tokens as the quantum audit, in both direct sampling and simulator argmax.
No temperature, rejection, retries or postprocessing changes predictions. All
six MLP samples have valid UTF-8, no non-text outcomes and no disallowed control
characters. The third begins “Quantum mechanics describes his father in November
Patcom the churchesminer”; the argmax examples include “womanwoman” and repeated
quotation marks. They contain locally plausible fragments but remain incoherent
and do not demonstrate useful long-form generation. Quantum's retained sample
set includes a control character and an invalid UTF-8 continuation. This small
qualitative comparison is not a blinded generation-quality benchmark.

The [complete sample table](scale-12-4-mlp-samples.csv) stores JSON-escaped text
without deleting any generated content. Raw text and token arrays remain under
`output/scale-12-4-mlp/analysis/`.

### Scope of the conclusion

The original question now has a matched-size answer for this larger instance:
**49.37% quantum versus 49.27% MLP most-likely accuracy**, with no resolved
accuracy difference and a clear **sampling advantage for MLP**. The earlier
44.10% versus 50.35% comparison used a different 512-token recipe and three-seed
means. This new result cannot establish that extra qubits causally closed that
gap: tokenization, architectures, parameter budgets and optimizer recipes differ.

Both current runs share the constant-rate recipe; no MLP-specific tuning or
regularization was selected after seeing these results. Equality of optimizer
settings is not evidence that the chosen setting is optimal for either model.
Seed replication, stronger classical tuning and independent confirmation would
be needed for a broader family-level claim. No further training was launched,
and the held-out test remains unscored.
