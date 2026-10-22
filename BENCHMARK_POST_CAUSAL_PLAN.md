# Next experiments after the causal completion — 2026-10-22

The [new audit](results/robust-causal-controls/README.md) completes the missing
eight causal controls. Both left-only and both-sided studies remain active.
Do not repeat the 40 completed target/control fits or run a broad Cartesian
grid. The next experiments should distinguish specific explanations and use
the measured execution improvement before increasing volume.

This is a prospective specification. The GPT/classical-feature prototypes
have CPU behavioral checks; **no next-batch launcher or frozen-runner
integration is delivered by this review**. The old causal launcher performs
the completed workflow and is not a launcher for this plan.

## 1. Validate microbatch 64 on full training trajectories

First batch: **16 fits** = {MLP-32, BERT-32, q8 CRZ, q8 identity} × {L4, L2R2}
× seeds 47/48. Retain 100×20,000 draws, Adam batch 64, LR 0.003, depth 2/std 0.05,
complete final validation and final-state scoring. Only execution microbatch
changes 8→64; retain the final 32-example optimizer batch. Reuse the matching
micro 8 histories/results as anchors, without altering their manifests.

Why full-budget repetitions are justified here: the pilot measures 64 one-step
conditions and shows 4.9–7.7× faster training steps, but does not measure
100-epoch behavior. Float reductions/Adam and BERT dropout consumption can
change trajectories. A tiny no-dropout multi-update test should precede the
full runs; it cannot replace their quality/time check. No emergency short
budget is substituted for scientific fits.

The preparation must bind parent manifests/target arrays and reproduce each
anchor's actual initial state. Preserve split/vocabulary/sampler policy,
model/optimizer settings, source guards and raw anchors. Hash prepared arrays,
common initial parameters and draw streams. Record normal-dropout trajectories
as well as deterministic numerical checks; do not try to force identical
dropout masks by changing the old model's training behavior.

Before submission, a single waited one-MIG `sbatch` job must:

1. Run CUDA preflight plus multi-update forward/gradient/Adam-moment parity on
   tiny full/partial batches, with declared dtype/tolerances and dropout mode.
2. Measure micro 64 training **and evaluation** for both tasks/four arms, warmup
   and repeated synchronized timings, memory and runtime/source hashes.
3. Estimate 31300 training steps, 101 sampled monitors plus full final validation,
   setup/checkpoint cost, array tails and planning allowance separately.
4. Reject nonfinite/parity/resource failures and any per-fit forecast above
   the selected Slurm limit. Keep the global ceiling at four one-10-GB MIG jobs.

Profile-derived **training-only** work for these 16 fits is about 1.46 MIG-hours:
roughly 0.91min/MLP, 4.07min/BERT, 15.72min/CRZ and 1.13min/identity fit, using
31300×the two-task average measured micro 64 step medians. These are neither
whole-fit durations nor a launch ETA. Evaluation, diagnostics, transfers,
saves/setup, contention and queue waits require fresh measurement. Do not
reuse the old global forecast divided by 7.7.

Analysis: paired endpoint/trajectory drift per seed/model/task; NLL/top-1/top-5,
gradient and update norms, epoch timing, fit/wrapper/allocation time and memory.
Use fresh seed confirmation for any subsequent quality claim. Two paired
seeds cannot certify statistical equivalence of stochastic recipes. Retain
both results if rankings change; choose one declared execution recipe before
new model comparisons and run all relevant controls under it.

An exact matrix-product-state backend is a separate optimization hypothesis:
product input +fixed-depth nearest-neighbor rank-two gates bounds the bond
dimension by 2^depth (four here). It need not outperform vectorization at q8.
First implement/test state, XYZ, input/parameter gradients and Adam/resume
parity, then profile. Never switch it into existing runs automatically.

## 2. Complete the classical comparison with the correct architecture

Two useful prototypes now exist in `scripts/research_model_controls.py`:

| Prototype | Controlled behavior | Full-vocabulary parameters |
| --- | --- | ---: |
| Scratch GPT decoder | width 32, two layers/two heads, causal mask, last-left-state next-word logits, tied head, explicit PAD mask, no library label shift |373280|
| Classical angle feature map | same hybrid lookup/encoder/decoder/tied head, 16 encoded angles→16 GELU→24 bounded features, replacing only the circuit |186824|
| Existing hybrid q8 CRZ | two layers, 24 XYZ features |186190|
| Existing scratch BERT-32 | middle MASK for L2R2; appended MASK for old L4 reference |385394|
| Existing MLP-16 / MLP-32 | ordered tied-head feedforward controls |187312 /368880|

GPT is near BERT's capacity but differs in query token, normalization, positional
capacity and dropout; it is an architecture comparison, not a pure mask
ablation. No pretrained weights or tokenizers are used. The classical feature
map adds 634 parameters over the hybrid (<0.4% total); its 24 bounded features
use 680 map parameters versus 46 in the circuit. This component capacity
difference matters even though the word embedding dominates total counts. The
features do not obey all quantum Bloch-vector constraints. Report this, rather than
calling it exactly capacity/geometry matched. MLP-32/BERT have about twice
the hybrid's parameter count. Add MLP-16 (only 1122 more parameters than q8,
about 0.6%) to separate total capacity from circuit architecture. Existing
screen MLP-16 scores used a different protocol and cannot substitute for these
matched-target fits.

CPU tests verify causal prefix invariance, tied/output vocabulary, PAD masks,
gradient flow and shared hybrid initialization. Remaining: versioned factory
integration, initialization fingerprints independent of circuit RNG, checkpoint/
optimizer/RNG resume, CUDA preflight/profiles and export support. No corpus
fits have been run for these prototypes.

After micro 64 validation, a small fixed-recipe discovery block adds **ten fits**:
GPT causal ×two seeds, classical features and MLP-16 ×two seeds ×both tasks. Reuse
compatible MLP/identity/CRZ/BERT anchors. It diagnoses representation/architecture;
it does not establish best-tuned performance. If selecting LRs, use an equal
grid .0003/.003/.01 for the relevant competing families. For these five
additional model/task cells alone that is **fifteen one-seed tuning cells**, followed by
fresh-seed confirmation; inherited baseline tuning must be compatible before
making an equally tuned claim. Do not launch both variants redundantly.

Add smoothed train-only count/backoff references in each task. Bidirectional
gap-context counts should use their allowed context, rather than handicap
them with a causal-only n-gram. Distinguish complete-pool counts from counts
of the exact sampled training multiset. Deterministic count references do
not need repeated training seeds.

## 3. Explain the gate-coordinate and endpoint effects

The exact per-layer mapping
`CRZ(c, t, g)=RZ_t(g/2) RZZ_ct(-g/2)` makes current CRZ/ZZ chains equivalent
in function class. Therefore a larger ansatz sweep cannot explain their
observed difference as greater expressivity. The focused counterfactual is
**four mapped-initialization ZZ fits**, two tasks ×two seeds: start from
the corresponding CRZ function, same encoder/head, then optimize in ZZ
coordinates. Reuse CRZ anchors only under the adopted compatible recipe.
Report physical coupling scale/local-field correlations and update norms.

A tiny ZZ implementation parameterized through CRZ latent coordinates should
reproduce CRZ training up to floating tolerances. That is an algebraic test,
not a reason to pay for four duplicate full trajectories. If mapped ZZ still
differs under coordinate-wise Adam, that supports optimizer-coordinate effects;
if the default ZZ difference disappears, initializer distribution was important.
Neither result alone proves a general superiority of one optimizer or gate.

Before further training, export additional statistics from existing DGX
predictions: source/frequency/length/position-conditioned sentence sums and
top 1→rank 2–5→miss transitions. Preserve targets, predictions and pairing IDs.
The current `predictions.npz` stores top-five IDs and per-token NLL/accuracy,
not full logits: full-rank/MRR and confidence/calibration require another
inference pass from retained checkpoints, not a training rerun or an inference
from missing ranks. Bind that pass to the old checkpoint/source and verify
that reaggregated top-k/NLL agree within declared numeric/tie tolerances.
Top-5 improves while top-1 falls causally; sentence-macro and token NLL even
reverse ZZ's average sign. These motivate error analysis but do not establish
its mechanism. Compact rank 1–5 totals cannot reconstruct full MRR or calibration.

## 4. Complete the representation, plateau, data and Hamiltonian studies

All blocks below retain **both tasks** and require separate implementation/
profiles. Counts are alternatives/modules, not one approved full grid or ETA.

| Question | Next discriminating study | Interpretation/control |
| --- | --- | --- |
| QCSE fixed decoder bottleneck | Same paper encoder: independent-marginal fixed-code head, learned word head on those features, and joint-Born valid-word distribution; retain encoding-only/product/classical references. | Joint Born retains correlations discarded by marginals without adding a word head; learned heads add capacity. Profile q14 separately. A final Z-diagonal block is invisible to every computational-basis probability, including joint Born. |
| Barren plateau or another optimization failure? | q4/8/12, controlled depth/init, fixed-dimensional head/projection, 64-target gradients at initialization and trained checkpoints, multiple batches/heads/circuit seeds. | Record signed means, variance/RMS, loss concentration, per-parameter activity and update/weight ratios. Separate exact dead gates, saturation, mini-batch cancellation, overfitting and poor coverage. No exponential-scaling claim from three small q values. |
| Corpus mixture/sampling | Uniform-sentence and 10% phrase-mass policies vs unchanged uniform-token anchors; start with MLP/identity/selected circuit, two seeds in each task (24 incremental fits). | Current phrase mass≈0.4%;10% is about 25×,50% about 125×. Fixed evaluation, fractional shared-source membership and all eligible targets nonzero; distinguish changed objective from importance weighting. |
| Exposure/convergence | Explicit 100→200 epoch continuation on a small fixed subset in each task; preserve optimizer and RNG and score full validation at both endpoints. | Current 100×20k is≈1.97 expected corpus passes,≈86% unique targets under the replay. Monitor improvements are not full-validation convergence evidence. Do not edit frozen epoch manifests. |
| Context scaling | Pair L4/L8/L16 and L2R2/L4R4/L8R8, same target population, modest model subset. | Equal total visible words; direction is a separate factor, not a reason to drop either task. New input capacity and costs need tuning/profiles. |
| Semantic Hamiltonian | Train-only symmetric graph, declared normalized Laplacian/isolated-node convention; identity, unitary evolution, heat and matched classical graph filtering. | Gate generators are not learned semantic graphs. Check Hermiticity, padding, t=0, norm/sparse parity and graph leakage; sparse exponential action, no dense propagator. |
| Generalizable semantics | Reserve external/source/template-held-out evaluation and later native GPT generation/BERT downstream representations. | Short curated word prediction and reused splits do not establish semantic understanding, pretrained-LM parity or quantum advantage. |

Prioritize readout/coordinate/classical controls before expensive qubit/depth
grids. Use existing checkpoints and predictions before buying more fits.
Keep negative results and the distinction between initialized probes, trained
fits, deterministic controls, reused reports and independent confirmations.

## Robustness and resource contract

Prespecify a primary contrast in each task before fresh confirmation, top-5
primary with NLL/top-1 secondary. Use three fresh seeds for a short list,
not five folds ×three seeds for every screening cell. Report seed-wise effects
and token/sentence weighting; adjust valid tests for the two-primary family
if making a joint confirmatory claim. The many post-hoc exploratory contrasts
remain unadjusted descriptive results. More bootstrap draws cannot create
new training variability or erase validation reuse.

Both scientific arms receive the same tuning/exposure opportunities. Full
eligible pools and 100×20k remain the comparison budget. Every new protocol
has source/data/split/initialization/sampler hashes, raw predictions, checkpoint
and optimizer/RNG state, task-level quality/memory/timing ledgers and Slurm
accounting. Pilot and arrays execute sequentially through `sbatch`, with at
most **four concurrently allocated 10 GB MIG instances** including other active
benchmark jobs. No local full-corpus training, interactive `srun`, automatic
test selection or updates to the frozen 483bea6 trainer are part of this plan.
