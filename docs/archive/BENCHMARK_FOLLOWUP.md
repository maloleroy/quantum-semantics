# Focused follow-up batch — 2026-10-16

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


Completed on 2026-10-20: [audited results, robustness and next steps](../../results/robust-followup/README.md).
That review also documents a reporting-only NLL precision fix and how to apply
it without changing the completed studies' frozen training source.

The [156-fit audit](../../results/robust-screen-main/README.md) supports a smaller
set of architectures, better classical capacity controls and objective-specific
calibration. The completed screen stays frozen. This protocol fixes choices
before observing held-out results; it does not rerun the whole 52-arm grid.

## Questions and fixed decisions

1. Does a trainable eight-qubit CRZ circuit improve over its matched analytic
   identity control once both use the same LR, encoder/head initialization,
   corpus, targets, optimizer draws and evaluation examples?
2. Does a width-32 MLP close BERT-32's advantage at similar parameter count?
   They are near-capacity controls, not exactly parameter/FLOP matched.
3. Do rankings or the circuit effect change with bidirectional information?
   Causal and CBOW studies have separate inputs, profiles, selections and reports.
4. Do local/frozen/XX/ZZ circuits or small-std mean embeddings provide useful
   exploratory alternatives? These controls use LR 0.003; they do not have the
   same tuning budget as the four primary architecture groups and cannot be
   called fully optimized winners.

All trainable fits retain **100 × 20,000 replacement draws**, Adam batch 64,
microbatch 8, final-state scoring, full final validation/test scoring and four
CPU threads. Use both complete curated sources and the same shared vocabulary.
Source hashes, dependencies, split seed 2026 and original artifacts stay frozen
for the whole follow-up. No early stopping, emergency short runs or reduced
training pools are used.

| Stage | Setups | Folds | Training seeds | CV fits | Purpose |
| --- | ---: | ---: | --- | ---: | --- |
| Refine causal | 18 | 3 | 43 | 54 | Equal LR grids and eight-qubit controls |
| Refine CBOW | 18 | 3 | 43 | 54 | Same model/tuning budget, bidirectional task |
| Confirm causal | 5 | 5 | 44, 45, 46 | 75 | New seeds and a small fixed architecture set |
| Confirm CBOW | 5 | 5 | 44, 45, 46 | 75 | Same confirmation design for the second task |
| Final refits | 5 per objective | Full 80% development | 44, 45, 46 | 30 refits | Held-out scores and paired intervals |

Total: **258 CV fits + 30 refits**, 576 million training draws / 9,014,400 Adam
updates. Four pilots are reused as their first CV fits. This is more fits than
before but avoids the expensive unproductive deep-QCSE grid. The three-fold
screen trains on about 53.33% of groups; five-fold confirmation trains on about
64%. Absolute score changes across stages can reflect that data difference.
All architecture comparisons within a stage share the same splits.

The 18 refinement setups are four equal LR grids (0.0003/0.003/0.01): BERT-32,
MLP-32, hybrid q8 CRZ depth 2/init 0.05, and hybrid q8 identity; plus q8 local,
frozen CRZ, XX, ZZ and mean-pooling at default std 1 and std 0.1. The latter
rescales the same random draws, retaining paired initialization directions.
The original mean/MLP defaults stay unchanged. Identity readout now evaluates
exact product-state XYZ moments analytically; logits/gradients are checked
against the former statevector implementation. This is a classical feature
control, without a claim of quantum computational advantage. Float32 operation
ordering can change tiny rounding errors, so preserve the old implementation
through its saved source rather than pooling old and new trajectories.

## Selection and inference rules

`prepare_benchmark_followup.py confirm --from-study STUDY` reads only the frozen
refinement selection. Choose BERT-32 and MLP-32 LRs independently by mean top-1,
then top-5, then NLL, then a deterministic name tie-break. For circuit/identity,
choose **one common LR** by their equally weighted mean top-1 (same tie-breaks).
This pairs the optimizer setting rather than comparing independently tuned
quantum/classical variants and attributing every difference to the circuit.

The fifth architecture is the previously selected angular QCSE reference.
It was selected from the causal discovery screen; it is a fixed reference in
CBOW, not the result of a new broad CBOW QCSE search. The local/frozen/XX/ZZ/mean
controls remain exploratory and are not silently substituted into the primary
CRZ-versus-identity hypothesis after seeing their scores.

All five confirmation arms are prespecified `--refit-controls`, so family
selection cannot discard either member of a contrast. Each appears exactly once
per seed in held-out reports, keyed by configuration name rather than family.
Both confirmation selections freeze before either test stage begins. The driver
never runs the refinement studies' proposed refits.

The primary contrast is **CRZ minus identity top-1**, separately for causal and
CBOW. Primary intervals are **97.5% paired seed/sentence percentile intervals**
(Bonferroni across those two planned contrasts); all metrics for the primary pair
are retained at that level, with top-1 as the formal primary metric. Other model
comparisons use exploratory 95% intervals. Retain the negative and null
results. An interval spanning zero means inconclusive evidence; it does not
prove equivalence. There is no post hoc "best seed" or test-based LR choice.
The bootstrap resamples paired training seeds and complete sentence groups,
retaining token weighting and overlapping windows within their sentence.
It is conditional on the selected architectures, fixed split and evaluated
population; three seeds give limited initialization uncertainty. The vocabulary
protocol remains transductive and this is not original BERT pretraining or a
new-domain semantic evaluation.

## Compute expectation and optimizations

The completed screen used 370.506 measured fit job-hours and an array span of
94.326 hours at peak four MIG instances. Its raw batch-cost forecast was 5.56%
low. New prepared settings explicitly apply a **1.10 planning factor** to fresh
CUDA batch measurements; this is an engineering allowance, not a statistical
upper bound. Scheduling tails and pilot reuse are included. Queue waits,
checkpoint I/O, failures and human/site interruptions are not guaranteed by it.

Before new profiles, causal timing proxies suggest about 56 allocation-hours
for each 54-fit refinement study, 115 for each 75-fit confirmation study, and 23
for each 15-refit study. Proxies use measured q8 CRZ/identity and BERT-32 costs,
2× MLP-16 cost for MLP-32, the measured q8/q4 CRZ ratio for new q8 controls,
and old mean cost for its initialization variant. These are explicitly unmeasured
proxies for the new shapes/objective, not benchmark observations or bounds.
The two-objective ideal four-slot total is about 97 hours; pilots, scheduling
tails and margin suggest **roughly 4–5 days plus queue delays**. Analytic identity
may reduce that, but no CPU-to-GPU speed factor is assumed.

Each pilot also measures actual **train-fold CE gradient variance**, using a
fixed eight-example training batch, a fixed encoder/head per register size,
2/4/8/12 qubits, depths 2/8, initialization std 0.05/1 and 32 circuit seeds.
It saves raw gradients, input IDs, common-state hashes and summaries under
`nlp-trainability-probe/`. Only circuit initialization varies within each
condition. Across qubit counts the encoder/head shape changes, so this remains
a conditional finite-size probe, not proof of an asymptotic barren plateau.
It uses no validation/test labels and is excluded from repeated-fit estimates;
its own phase duration is retained alongside the older synthetic-cost probe.

Each study gets its own single-MIG CUDA pilot and `compute.json` before its
array. The driver stops if a forecast fit exceeds 22 hours, leaving two hours
within the requested 24-hour allocation. Slurm still enforces the actual limit;
completed epochs are resumable if an estimate proves insufficient. CUDA parity
is checked inside each allocation. Keep microbatch 8: a larger microbatch is a
future profile, with new BERT dropout/numerics validation and a new study.

`sbatch --wait` holds the submitter until an array terminates and propagates a
failed task status ([Slurm documentation](https://slurm.schedmd.com/sbatch.html)).
The existing queue/resource guard and a driver lock prevent overlapping runs in
this checkout. Arrays run sequentially; each requests one 10 GB MIG per task
and at most four tasks at once. The pilot uses one. Independent manual jobs or
other checkouts remain outside this ceiling; use this driver for this batch.
The driver waits on the login host; GPU training runs only through `sbatch`.

## DGX commands: launch, inspect, resume, export

Run from the existing checkout after the old allocations finish. The previous
outputs and their `source/` snapshots remain available; new studies use unique
directories. Do not edit/pull source while the follow-up is active.

```bash
cd ~/quantum-semantics &&
git switch codex/qcse-bert-robust-benchmark &&
git pull --ff-only &&
uv sync --locked
```

```bash
squeue --me --partition=prod10
```

Once there are no active `prod10` jobs, launch the complete staged batch:

```bash
mkdir -p outputs/robust-followup
nohup bash scripts/run_benchmark_followup.sh \
  >> outputs/robust-followup/driver.log 2>&1 < /dev/null &
echo $! > outputs/robust-followup/driver.pid
```

The login driver survives closing SSH. It prepares/runs both refinements,
freezes/runs both confirmations, then runs the final refits and reports. Every
fit records its metrics, history, checkpoints, predictions, phase timings,
actual device and scheduler receipts. The two refinement reports are timing /
development selections; held-out reports are in the two confirmation studies.

Inspect without interrupting the driver:

```bash
tail -n 60 outputs/robust-followup/driver.log
squeue --me --partition=prod10
```

```bash
for pointer in outputs/robust-followup/*-study.txt; do
  test -f "$pointer" || continue
  .venv/bin/python scripts/robust_benchmark.py status "$(cat "$pointer")"
done
```

Any failed submission/task/report stops the driver. Inspect its log and Slurm
errors. Once queued/running jobs are terminal, rerun the same `nohup` command
without editing source or deleting the pointer files. It resumes unfinished
fits and skips validated completed phases through the same selection/report
checks. A second active driver exits immediately through its lock. Completed
runs can be audited again without launching new fits.

Success ends with `Follow-up complete: 108 refinement CV fits, 150 confirmation
CV fits, 30 refits.` Then export the diagnostics (keep all binary checkpoints
and predictions on DGX):

```bash
find outputs/robust-followup -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.psv' \
     -o -name '*.txt' -o -name '*.log' \) -print |
zip -q outputs/robust-followup-diagnostics.zip -@
```

Download `outputs/robust-followup-diagnostics.zip` to Papers for analysis.

## Optional next data cohorts

Full source/window ablations remain prepared choices, not an automatic
Cartesian expansion. After refinement, use one of `window-2`, `window-8`,
`phrases`, `cleaned` with `--from-study` pointing to the matching refinement
study and `--objective causal` or `cbow`. Each holds the selected five-arm set
fixed and uses three folds / seed 43 / the full 100 × 20,000 budget (15 fits).
For example, after the driver is terminal:

```bash
SOURCE=$(cat outputs/robust-followup/refine-causal-study.txt)
PREPARED=$(.venv/bin/python scripts/prepare_benchmark_followup.py window-8 \
  --objective causal --from-study "$SOURCE") &&
STUDY="${PREPARED##* CV tasks at }" &&
test -f "$STUDY/manifest.json" &&
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage pilot \
  --wait --sbatch-option=--time=24:00:00 &&
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --concurrency 4 \
  --wait --sbatch-option=--time=24:00:00 &&
.venv/bin/python scripts/robust_benchmark.py select "$STUDY"
```

Source-only cohorts change the evaluated population and splits; compare models
within each cohort. They do not isolate training-source quality on a fixed
external evaluation population. Graph/Laplacian semantic Hamiltonians, noise,
real quantum hardware and downstream semantics remain research beyond this
batch; the present XX/ZZ controls do not implement those claims.
