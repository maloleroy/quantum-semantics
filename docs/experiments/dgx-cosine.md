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
