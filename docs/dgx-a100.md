# DGX-A100 operations

The supported robust-study controllers use `prod10`, one
`gpu:nvidia_a100_1g.10gb:1` per task, and **at most four concurrent instances**.
The site rejects interactive `srun` allocations on this partition: use `sbatch`.
Do not replace the Slurm-provided `CUDA_VISIBLE_DEVICES`, including MIG UUIDs.

## Keep the frozen runtime separate from published history

Received robust studies ran from 483bea6. Their manifests hash executed code,
data and parent studies. Current branch code includes later audits, prototypes
and newly committed legacy encoder/selection changes. Pulling/checking out that
code into the frozen runtime can break source guards or active jobs.

Publish/read documentation through Git without changing the working tree:

```bash
cd ~/quantum-semantics
git fetch origin codex/qcse-bert-robust-benchmark
git show origin/codex/qcse-bert-robust-benchmark:backlog.md
```

For a new implementation, use an isolated versioned checkout/environment and
new manifests, or the documented copied-tool extension mechanism. Never edit
hash guards/manifests to make an incompatible trainer appear identical.
Install dependencies with `uv sync --locked` between jobs, not in array tasks.

## Inspect the existing causal cohort

These are read-only commands from the original DGX checkout:

```bash
cd ~/quantum-semantics
COHORT=outputs/robust-causal-controls/causal-controls-20261022T012543Z-q1n83tfn
.venv/bin/python scripts/robust_benchmark.py status "$COHORT/L4"
squeue --me --partition=prod10 --states=all \
  --format='%.18i %.24j %.12T %.12M %R'
sacct -j 10400 --parsable2 \
  --format=JobID,State,ExitCode,ElapsedRaw,Start,End,AllocTRES
```

The received export records 8/8 completed fits. Their array is 10400; pilot 10399.
No jobs in `squeue` can mean completion; use `status`, accounting and saved
ledgers to distinguish completion from failure. `sacct -j JOBID` literally
fails—use the actual numeric job ID. A missing `history.json` is expected for
some parameter-free controls; the timing reporter supports that case.

## Submission rules and scripts

| Entry point | Scope |
| --- | --- |
| `submit_robust_benchmark.py` | Immutable prepared study; fit/pilot/refit stages, concurrency 1–4, resource override restrictions, terminal-state queue handling and shared lock. |
| `run_benchmark_followup.sh` | Sequential refinement/confirmation workflow with waited profiles/arrays and validation-frozen choices. Completed cohort, not the next plan. |
| `submit_gradient_replications.sh` | One-MIG replicated gradient diagnostic with preflight. Completed job 10324. |
| `run_target_controls.sh` | Copied-tool 32-fit common-target study. Completed; do not repeat. |
| `run_causal_controls.sh` | Copied-tool pilot plus missing eight controls. Completed; not a micro 64 adoption launcher. |
| `submit_focused_sweep.sh` | Historical focused protocol; now waited sequential arrays, concurrency 1–4, shared lock, active/unknown queue guard, fixed one-MIG resources. Not a modern 100×20k study. |

The current drivers share `logs/benchmark-submission.lock`, clear inherited
`SBATCH_*` overrides and block when existing user `prod10` jobs could overlap.
Terminal completed/failed/cancelled rows are accepted; unknown states block.
The pilot finishes before arrays start. An array `%4` limits that array, not
all independently submitted arrays; sequential drivers/queue checks are needed
to respect the combined ceiling. Manual/unrelated submissions remain outside
those locks.

Inherited pre-branch submitters (`submit_sweep.sh`, `submit_semantic_sweep.sh`,
`submit_dgx_a100_10gb.sh`) retain older multi-array policies. **Do not use them
for new work under the four-MIG contract.** They need retirement/guarding before
reuse. Archived ten-job instructions describe historical configurations, not
the supported current launch path.

Keep Slurm's four-CPU allocation and explicit thread count separate. Supported
modern jobs use four numerical threads; local checks use one. The focused
batch script now defaults to its allocated CPU count rather than 32 threads.
More threads are not a demonstrated optimization.

## Resume, selection and export

Resume only the same study/source/target/seed recipe. The robust submitter
skips valid completed tasks and preserves execution attempts. Check status and
logs before submitting; do not cancel/relaunch successful fits to regenerate
reports. Fixed 100-epoch comparisons select final weights; older family-winner
selection/refit workflows are separate and remain validation-only.

Keep on DGX: prepared arrays, manifests/source snapshots, prediction NPZs,
checkpoint/selected states, histories and accounting. Compact diagnostic ZIPs
include hashes and sentence sums but intentionally omit large arrays/states.
Those omissions prevent independent token/logit/resume checks on the Mac.

Export source/frequency/position and paired rank-transition sums from existing
predictions before buying more training. Saved NPZs retain top-five IDs and
per-token metrics, not full logits. Full MRR/confidence/calibration needs a
checkpoint inference pass. Conversation exports belong only in ignored local
outputs, never in diagnostics deliveries or Git.

## Next batch readiness

The [16-fit micro 64 trajectory study](next-experiments.md) is specified but its
launcher is not implemented. New GPT/feature models still need runner/resume/
CUDA integration. There is no new launch command yet. Full budgets remain
100×20k, and a fresh one-MIG training/evaluation profile must precede an array.
Use [compute notes](optimizations.md), not the old screen forecast, for planning.
