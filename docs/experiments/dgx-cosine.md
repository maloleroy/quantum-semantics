# CUDA cosine comparison with fresh constant-rate controls

3 November 2026. Declared before training. This carries forward the pending
[cosine protocol](adaptive-cosine.md) under the new authorization to run long
research experiments on the DGX. No new model architecture is introduced.

The retained adaptive quantum reference reaches 2.7007 NLL versus matched MLP
2.4393. Test fixed cosine learning-rate decay and retained-context length. Fresh
constant-rate fits on the same CUDA backend prevent the primary schedule effect
from being confounded with the CPU-to-GPU transition. Historical CPU scores remain
context, not the primary schedule controls.

## Locked comparison

Seven fresh fits, all seed 11, 30 epochs, batch 512, Adam initial rate .003,
NLL loss and NLL selection, two stages, auxiliary size 2, recent window 16,
phase scale 1, float64/complex128, Born readout and no frozen parameters:

| Label | Model | Rank | Schedule | Context |
|---|---|---:|---|---|
| constant_adaptive | adaptive | 128 | constant .003 | 128 |
| cosine_adaptive | adaptive | 128 | cosine .003 to .0003 | 128 |
| constant_static | product | 176 | constant .003 | 128 |
| cosine_static | product | 176 | cosine .003 to .0003 | 128 |
| constant_mlp | adaptive_mlp | 128 | constant .003 | 128 |
| cosine_mlp | adaptive_mlp | 128 | cosine .003 to .0003 | 128 |
| cosine_short | adaptive | 128 | cosine .003 to .0003 | final 16 |

All receive 30,720 optimizer updates and 31 validation checks including epoch zero.
The schedule formula and limits remain as declared in the original cosine protocol:
one rate per epoch, no warmup/restarts, final ratio .1. Its average rate also changes,
so this does not isolate the cosine shape from a smaller average learning rate.
Amplitude decay .9 is recorded but unused by these models. Counts and parameter
budgets remain adaptive 751,892, static 751,988, MLP 750,460.

Use the exact prepared 524,288-position dataset at
`output/product-coverage/batch-RofZOb/data/full.npz`, SHA256
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The paired short dataset only masks columns before the last 16. Verify original
vocabulary, BPE merges, counts, unigram, causal target/group/byte-length arrays
and retained context across all partitions. Keep the 128-column shape. No new
positions, source data or tokenizer/statistic fitting. Test is not scored.

## GPU execution and reproducibility

Use an isolated remote checkout `~/quantum-semantics-cosine-20261103`, created
without replacing the CUDA verification checkout. Install the locked CUDA extra
with the default CPU group disabled. Submit one full A100-SXM4-80GB in `prod80`,
four CPU support threads and an eight-hour allocation ceiling. Fits run sequentially
on that GPU; do not run a CPU-only research fallback. The larger allocation ceiling
is a limit, not a prediction of wall time. Planning estimate for all seven fits is
2–6 hours plus queue/transfer time, extrapolated and not yet measured.

Before any fitting, the GPU gate checks CUDA-enabled PyTorch, A100 identity, all
model parameters/buffers, a real batch-512 forward pass, finite CUDA gradients and
nonzero gradient signal for adaptive, static and MLP families. It saves small
`gpu-gate.json` metadata. The CUDA parity tests must also pass. Every research
train/build/evaluation call explicitly selects CUDA; run/fit metadata must record
CUDA and A100. Allocation alone is insufficient proof.

All source, dependencies, tools and full-data inputs are checksummed locally and
remotely. Generated short data get a separate checksum before fitting. Source and
data hashes are checked again at completion; selected weights are independently
reloaded on GPU and validation arrays reproduced to 1e-10. No code/dependency
changes beneath the running allocation. An archived local source copy and the
remote checkout preserve the exact implementation for later audits.

## Reporting and interpretation

Save ten metric rows, 20 dominant/non-dominant target buckets and 32 paired group
interval rows. Compare cosine-adaptive with the other nine fits/baselines and
separately compare cosine-static/MLP with their constant counterparts. Baselines
are all-training bigram, unigram and training-modal space. Do not bootstrap
infinite NLL against the modal predictor. Sampling, argmax and NLL stay distinct;
classical argmax is an available practical strategy.

Save 24 fixed-prompt, unfiltered 64-token continuations: six new full-context fits
plus bigram/unigram, using the existing three prompts and seeds 20261030–20261032.
Save raw token IDs in NPZ and text/tables in CSV. CUDA RNG draws need not match
historical CPU draws at the same seed. Short-context generation is omitted because
the generic generator uses the full window. Retain all invalid/UNK outcomes.
No held-out test scores or archived-reference resampling.

This is a single-seed reused-validation screen, with descriptive paired article
bootstrap intervals that exclude selection uncertainty. Better sampling alone is
not useful text generation. Useful longer context, coherent prose, strong-classical
superiority and quantum hardware advantage remain unproven.

## Submission and completion

Generated tools live in ignored `output/dgx-cosine/`. The local controller runs
`submit.sh`, which copies pinned inputs, installs the remote CUDA environment,
submits with `sbatch --wait`, records Slurm accounting and retrieves results/logs.
The codex-long-jobs skill supplies durable state and automatic completion delivery;
no model polling while the job runs. The remote marker is `DGX_COSINE_COMPLETE`.
The local success marker `DGX_COSINE_RETRIEVED` additionally requires successful
Slurm completion, copied artifacts and the expected seven selection locks.
If a fit fails, preserve completed runs; audit the failure before resubmitting the
unchanged remote runner. Reading this protocol does not claim the job completed.

### Restart on 4 November 2026

Slurm job 10844 failed after 12 seconds before any research fit. The GPU forward
and gradient gate passed, but the upload omitted `data/phrases.csv`, required by
the CUDA test fixtures. No selection locks or research results were produced.
Failed logs and the original manifest are preserved in
`output/dgx-cosine/failed-10844/`.

The corrected upload and checksum manifest include that fixture. The retry uses
the same verified source, dataset and seven-fit protocol in the existing remote
checkout. Slurm job 10875 passed the GPU gate and all 34 CUDA tests, prepared and
verified the short-context dataset, and started `constant_adaptive`. Its training
process was observed on the A100 using 1,284 MiB and 86–89% SM utilization.
This verifies startup, not successful completion or a scientific result.

The local controller is `qcse-dgx-cosine-20261104-03`; its log is
`output/dgx-cosine/controller-retry-20261104-03.log`. An earlier retry controller
stopped at a Slurm-accounting preflight formatting check without submitting a job.
The corrected controller retains automatic completion delivery and artifact
retrieval. The research goal is to be paused after startup verification at the
user's request; the Slurm experiment continues independently.

## Completed results

4 November 2026. Slurm job **10875 completed with exit 0:0 in 02:08:25** and
emitted `DGX_COSINE_COMPLETE`. The local controller lost its SSH connection and
then failed retrieval with exit 255; this was not a training failure. Recovery
retrieved the completed artifacts without restarting any fit. Remote input and
generated-data checksums passed, and every downloaded result file matched a
fresh remote SHA256 manifest. Raw artifacts remain in
`output/dgx-cosine/retrieved/` and the isolated remote checkout.

The GPU summary reloaded all selected weights and reproduced validation arrays
to 1e-10. The local audit additionally checked source/data/weight locks, model and
training configuration, the 31 history entries and actual NLL-selected epoch,
all ten metric rows, 20 target buckets, 32 paired article intervals and 24 saved
sample renderings. Count baselines were independently recomputed from training
statistics. No held-out test was scored.

| Fit | Validation NLL ↓ | Sampling S ↑ | Argmax ↑ | Selected epoch | Fit minutes |
|---|---:|---:|---:|---:|---:|
| Constant adaptive | 2.702110 | 25.774% | 40.576% | 29 | 16.99 |
| Cosine adaptive | 2.701256 | 25.685% | 40.088% | 30 | 16.98 |
| Constant static | 2.755989 | 24.610% | 39.551% | 29 | 10.06 |
| Cosine static | 2.755683 | 24.672% | 39.453% | 30 | 10.07 |
| Constant MLP | 2.439315 | 38.437% | 46.826% | 14 | 14.92 |
| Cosine MLP | **2.407749** | **39.180%** | **47.925%** | 20 | 14.93 |
| Cosine adaptive, final 16 tokens | 2.701010 | 25.763% | 39.844% | 29 | 42.28 |
| Training bigram | 3.059255 | 23.158% | 34.033% | — | — |
| Training-modal space | ∞ | 30.054% | 30.054% | — | — |

The unexpectedly slower short-context fit is recorded as observed. Its padded
shape remains 128 columns; this study does not establish why its wall time differs.
These fit timings are not a controlled CPU/GPU benchmark or quantum speedup.

Cosine minus constant adaptive NLL is **−0.000854**, with descriptive 95% paired
article interval **[−0.010813, +0.008458]**. Static NLL is similarly unchanged.
The MLP improves by **−0.031566 [−0.051679, −0.013244]**, widening the best
classical/quantum gap. Cosine adaptive trails cosine MLP by
**+0.293507 [+0.228581, +0.349908]** NLL. Full minus short adaptive NLL is
**+0.000245 [−0.010452, +0.009418]**: this comparison does not establish a
benefit from the older 112 tokens. Single-seed, reused-validation intervals remain
descriptive and omit training/selection uncertainty.

The target buckets expose the continuing language deficit: cosine adaptive
samples non-space targets correctly only **6.08%** of the time, versus **19.30%**
for cosine MLP. Quantum NLL on these targets is 3.6762 versus 3.3255 for MLP.
The stronger MLP argmax option also remains available. No gain in quantum advantage,
useful long-context prediction or coherent long-text generation is demonstrated.

All fixed-prompt quantum continuations remain fragmented. The three cosine-adaptive
samples contain five control characters and two UTF-8 replacement characters;
the three constant-adaptive samples contain six control characters. No sampled
token was rejected. The cosine MLP samples have no such characters but still do
not form coherent long text. Saved text is diagnostic, not a language-quality score.

Retain constant adaptive as the quantum reference: cosine's tiny NLL difference
does not justify replacing it, and its sampling/argmax are lower. Retain cosine
MLP as the stronger classical reference. The next representation experiment must
address conditional non-space prediction and demonstrate useful quantum dynamics;
more schedule tuning alone is not supported as the main route forward.

Compact evidence: [metrics](dgx-cosine-validation.csv),
[paired intervals](dgx-cosine-paired.csv), [target buckets](dgx-cosine-target-buckets.csv),
[unfiltered samples](dgx-cosine-samples.csv), and
[sample diagnostics](dgx-cosine-sample-diagnostics.csv).
