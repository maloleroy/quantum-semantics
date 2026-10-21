# Executable target/context controls — 2026-10-20

**Completed 2026-10-21:** all 32 fits received and
[independently audited](results/robust-target-controls/README.md). Keep this
protocol as the historical executable 32-fit batch. The next work is the
[full causal/bidirectional plan](BENCHMARK_DUAL_TRACK_PLAN.md), including the
eight missing causal mechanism fits. Rerunning this driver does not implement
that new design.

This development batch follows the
[replicated gradient audit](results/gradient-replicated-10324/README.md) and the
[prediction audit](results/robust-followup-prediction-review/README.md). It tests
context information and circuit mechanisms while keeping target composition,
training exposure and split identities fixed. It reuses the original frozen
trainer without modifying its source, checkpoints or completed studies.

## Matrix and questions

| Recipe | Context slots | Models | Seeds | Fits |
| --- | --- | --- | --- | ---: |
| L4 | Four preceding words | MLP-32, BERT-32, q8 CRZ, q8 identity | 47,48 | 8 |
| L2pad | Two preceding words, two right PAD slots | Same four | 47,48 | 8 |
| L2R2 | Two preceding, two following words | Same four + q8 local/frozen CRZ/XX/ZZ | 47,48 | 16 |

The main development contrast is **L2R2 versus L2pad**, with the same slots and
middle BERT mask position. It isolates providing right words under this recipe.
L4 versus L2pad changes left information and slot/mask placement, so it is a
different contrast. L4 versus L2R2 changes both sides and is not alone a clean
right-context intervention. Use top-5 as the primary development ranking and
retain top-1/NLL and all per-seed results. There is no LR/model winner selection
or test refit in the driver.

The L2R2 circuit block shares q8 encoder/head initialization and training draws:
identity, local rotations, frozen CRZ, trainable CRZ, XX and ZZ. Local rotations
can be absorbed into the unrestricted linear decoder at the function-class
level; different training dynamics remain possible. CRZ and ZZ are different
gate generators. This block is not the graph-Hamiltonian experiment.

## Data, fairness and provenance

- Sources remain the complete prepared phrases/cleaned union, with the same
  vocabulary and sentence groups as the original studies. Vocabulary remains
  transductive; no new vocabulary-efficiency claim is made.
- All recipes use **positions > 0 in training as well as validation**.
  Canonical IDs come from the saved causal arrays; CBOW contexts are matched by
  identical ordered `(group, position, target)` keys. Right slots are explicitly
  padded for L2pad. The target word at the target position never enters context.
- Reuse saved causal fold 0: **129,390 training groups / 1,014,085 eligible
  training targets; 32,348 validation groups / 252,120 validation targets**.
  This is approximately a 64/16/20 grouped split. The already-observed original
  40,434 test groups are not rescored in this batch. `folds=1` in the derived
  manifest means one saved
  fold, not a newly computed one-fold cross-validation procedure.
- Every training target stays eligible. Uniform-token draws with replacement
  use the unchanged sampler and identical canonical train IDs/seed across
  recipes. All 100 epochs draw 20,000 examples, batch 64/microbatch 8: **2M draws /
  31,300 updates per fit; 64M draws / 1,001,600 updates overall**. Final epoch 100
  is scored on complete validation; sampled curves use the same fixed monitor.
- LR .003 and q8/depth 2/init std .05 are fixed from previous development work.
  This estimates fixed-recipe/context/mechanism effects, not each architecture's
  independently optimized performance. Changed tasks can need later equal LR
  tuning. Two fresh seeds address some initialization sensitivity without
  repeating five folds for every configuration.
- MLP/BERT remain approximately capacity-matched references with their actual
  architecture/init/dropout recipes. Hybrid capacity differs. This is not
  exact FLOP matching, pretrained-BERT replication or quantum hardware advantage.
- Original sources and prepared parent artifacts are hash-checked. New arrays,
  parent manifests, split target keys, copied tools and source snapshots are
  bound in derived manifests. Old results are not relabelled as new fits.

The new Python script handles preparation, profile-based planning, completeness
and final reporting. The existing trainer still handles Adam, sampling,
checkpoints, interrupted resume, gradients and fit validation. No training loop
is duplicated and no old source guard is disabled.

## Submit on DGX

From the original checkout, fetch/copy the tools. Do not pull changed training
sources into the frozen checkout. The two files must come from the same commit.

```bash
cd ~/quantum-semantics &&
git fetch origin &&
mkdir -p outputs/target-controls-tools &&
git show origin/codex/qcse-bert-robust-benchmark:scripts/target_control_study.py \
  > outputs/target-controls-tools/target_control_study.py &&
git show origin/codex/qcse-bert-robust-benchmark:scripts/run_target_controls.sh \
  > outputs/target-controls-tools/run_target_controls.sh &&
{
  nohup bash outputs/target-controls-tools/run_target_controls.sh \
    > outputs/target-controls-driver.log 2>&1 < /dev/null &
  echo "Driver PID: $!"
}
```

The small background controller prepares new artifacts and waits for Slurm;
all profiling and training execute through `sbatch`. First, one **single-MIG**
job runs the existing CUDA parity preflight and five-repeat batch profiles for
each recipe. A fingerprint-matched forecast with a 1.10 engineering margin is
required before fits. A predicted fit above 23 hours aborts before launching
the arrays; training is never shortened to meet the 24-hour allocation limit.

Then three arrays run **sequentially**, each capped at four one-MIG tasks.
The shared submission lock/queue guard and waited stages prevent intentional
overlap. Pending/running/unknown prod10 jobs block a new submission; retained
completed queue entries do not. Inherited `SBATCH_*` overrides are removed.
No interactive `srun`, independent extra arrays or test-refit stage is used.

Historical medians imply 33.9 allocation-hours and 10.9 sequential-array hours
before margins. Allow roughly **12–16 hours plus queueing** until the fresh
CUDA forecast appears. This proxy is not a confidence interval. Queue delays,
checkpoint I/O and restarts are not included in the batch-time extrapolation.

```bash
tail -n 50 outputs/target-controls-driver.log
squeue --me --partition=prod10 --format="%.18i %.22j %.12T %.12M %R"
```

The pointer `outputs/robust-target-controls-study.txt` records the new cohort.
Its `compute.json` contains the profile-based forecast, with **32 fits and zero
refits**. To resume after a failed controller/job, rerun the same copied driver;
it keeps the cohort, reuses profiles, skips complete arrays and resumes partial
fits through the original launcher. Keep the copied tools unchanged. `--new`
explicitly creates a separate cohort and should not be used for a restart.

## Outputs and next decisions

Successful completion creates
`outputs/robust-target-controls/target-controls-*-diagnostics.zip` automatically.
It includes full fit ledgers, histories, runtime/timing records, profiles,
prepared JSON/source metadata, top-1-through-top-5/miss rank counts and
float64 per-sentence metric sums for every seed/model. The report validates
all token predictions against prepared targets before packaging. Full prepared
binary arrays, raw token predictions and final/continuation checkpoints stay
on DGX and are explicitly listed as omitted. Predictions are checksum-bound.
There are no intermediate epoch 25/50 weight snapshots; per-epoch gradient
aggregates and final checkpoints are retained by the original trainer.

Download the diagnostic ZIP after completion. Compare paired per-seed effects
and sentence-macro/token-weighted estimates before choosing further fits.
Two seeds and a reused validation split are discovery evidence. Bootstrap
draws do not add independent training runs, and no held-out significance claim
should be made from this study. Decide later whether equal tuning/new-seed
confirmation, data-mixture sampling or the train-only graph-Hamiltonian block
addresses the remaining scientific uncertainty most effectively.

The paper QCSE adapter's fixed encoding and independent-bit word decoder remain
a separate representation question. Its matched-position results are already
near the frequency reference. A later encoding/readout study should retain this
target protocol and include a suitable classical counterpart; it requires
implementation and a fresh profile before assigning training fits. The current
32-fit block studies the learned hybrid and its classical anchors.
