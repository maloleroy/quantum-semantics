# Launch the missing causal controls — 2026-10-22

This implements the next step of the
[two-track plan](BENCHMARK_DUAL_TRACK_PLAN.md), following the
[completed 32-fit audit](results/robust-target-controls/README.md).
It does not repeat that 32-fit cohort or implement the later GPT/readout/sampler
studies. Both research directions remain active.

## What runs

1. A waited one-MIG 10GB `sbatch` pilot after the existing CUDA parity preflight.
   Both L4 and L2R2, four MLP/BERT/CRZ/identity anchors, microbatches 8/16/32/64,
   full 64 and final-partial 32 example batches:64 conditions, five measured
   timing repetitions after one warmup. Fixed train inputs and all seeds/source
   metadata are retained. Loss/logits, accumulated gradients, one clipped Adam
   update and first/second moments are compared with microbatch 8. Eval-mode
   parity disables dropout; timed steps use normal dropout. Engineering
   tolerances and maximum differences are recorded. Failed numerical comparison
   is evidence against adopting that change, not a reason to change the budget.
2. Fresh CUDA timing profiles for the four missing causal models. The forecast
   includes 100×20,000 draws, monitoring and final full validation,1.10 planning
   factor and 20s/job. A forecast over 23h per fit aborts before a 24h submission.
3. **Eight new fits:** L4 × {q8 local, frozen CRZ, XX, ZZ} ×seeds 47/48.
   Fixed 100epochs,20,000 draws/epoch, Adam batch 64/microbatch 8, LR 0.003,
   depth 2/std 0.05. Final-state scoring, complete eligible training pool and
   complete validation; no selection, refits or new test scoring.

The pilot never automatically changes a training setting. Keeping microbatch 8
preserves the original mechanism comparison. Later optimization adoption
requires a declared new execution recipe and trajectory check.

The completed L4 main anchors and L2R2 eight-model controls are reused only as
saved evidence. Their arrays, results and manifests are not overwritten. The
new preparation checks parent source/target hashes, budgets and actual result
ledgers. It copies the exact L4 prepared artifacts and the corresponding
bidirectional control definitions, then records new task/tool fingerprints.
Compact parent sentence statistics accompany the new export for paired review.

The controller and shared submission lock prevent overlapping benchmark
submissions. Pending/running/unknown prod10 states block submission; retained
completed records are accepted. Inherited `SBATCH_*` options are cleared.
The pilot requests one instance; the subsequent waited array has `%4` and
each task requests one `nvidia_a100_1g.10gb`. Combined maximum:four instances.
The Slurm-provided CUDA device mask is preserved. No interactive `srun`.

## Transfer and run

Use the new `causal-controls-next.zip` delivery, uploaded into
`~/quantum-semantics`. It contains three copied tools, checksums, a run note
and a Git bundle. The bundle is available for publication/history; **no Git
checkout/pull is needed to run**. The original frozen 483bea6 environment stays
in place. The old review-only transfer archive is not this launcher.

```bash
cd ~/quantum-semantics &&
unzip -n causal-controls-next.zip -d outputs/causal-controls-tools-20261022 &&
(cd outputs/causal-controls-tools-20261022 && sha256sum -c SHA256SUMS) &&
{
  nohup bash outputs/causal-controls-tools-20261022/run_causal_controls.sh \
    > outputs/causal-controls-driver.log 2>&1 < /dev/null &
  echo "Driver PID: $!"
}
```

The parent pointer `outputs/robust-target-controls-study.txt` must still identify
the completed 32-fit cohort, with raw prepared arrays and ledgers present on DGX.
This is the normal state after the previous driver. The compact downloaded ZIP
alone cannot substitute for those DGX artifacts. No manually entered study path
or hardcoded cluster absolute path is required.

```bash
tail -n 60 outputs/causal-controls-driver.log
squeue --me --partition=prod10 --format="%.18i %.24j %.12T %.12M %R"
```

After preparation, check just the eight-fit study:

```bash
.venv/bin/python scripts/robust_benchmark.py status \
  "$(cat outputs/robust-causal-controls-study.txt)/L4"
```

The driver can survive disconnects. To resume after a failed job/controller,
rerun the same `nohup bash ...` command with unchanged tools. It retains the
new cohort, skips the completed pilot/forecast, and resumes incomplete fits
through the existing launcher. Completed results are checked by final reporting.
Do not delete pointers, use a fresh preparation or submit a second array around
the driver. If the pilot failed before writing its forecast, restarting reruns
that single-MIG stage; partial diagnostics remain available.

## Duration and outputs

Prior measured bidirectional costs imply about 15.6 MIG-hours and 4–6h for the
eight causal fits at four slots, plus the pilot and queue waits. This is a proxy,
not a measured new causal ETA or confidence bound. The pilot has a 1h allocation;
fresh `compute.json` supersedes the proxy. More constrained Slurm availability
can extend elapsed time. Each actual fit still has a 24h allocation.

The new pointer is `outputs/robust-causal-controls-study.txt`.
The final ZIP appears automatically only after all eight fits/report succeed:

```bash
ls -lh outputs/robust-causal-controls/*-diagnostics.zip
```

Download that ZIP for the next review. It includes both-track pilot data,
fresh profiles/forecast, source snapshots, all eight result/history/runtime/
timing records, rank counts and float64 sentence aggregates, plus compact
parent anchors. Full prepared arrays, token predictions and checkpoints remain
on DGX and are listed as omitted. No manual `select` or test refit is required.
