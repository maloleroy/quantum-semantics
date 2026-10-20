# Next studies: top-5, mechanisms and data — 2026-10-20

This is a concrete staged design following the
[completed audit](results/robust-followup/README.md), not an instruction to
submit another full batch now. The copied float64 reporter's export has now
been received and
[independently checked](results/robust-followup-prediction-review/README.md):
CBOW top-5 gains persist across three seeds and common positions, while top-1
remains inconclusive. This closes the corrected-statistics gate. The replicated
gradient diagnostic is the next implemented mechanism; the remaining
neural/data blocks still need the implementation listed below.

The replicated gradient probe is implemented and locally tested. The neural,
sampling and graph blocks below are **designs requiring implementation and
fresh CUDA profiles**. They are not launchable by adding settings to the old
manifest. In particular, the existing runner hardcodes top-1 selection,
uniform token sampling and the original objective-specific target populations.

## Order, budgets and evidence boundaries

1. Completed: saved prediction aggregates, top-5 intervals, matched-position
   contrasts and seed sensitivity. No fits or GPU allocation were needed.
2. Run cheap replicated NLP gradient diagnostics with raw-gradient export.
3. Prepare a common split/target/sampling infrastructure and a small neural
   development screen. Freeze the population before model selection.
4. Compare Hamiltonian generators and mixture/sampling policies in separate
   blocks, reusing identical anchor fits. Develop an actual graph-Hamiltonian
   prototype separately before budgeting its full-corpus inference.
5. Confirm only prespecified contrasts worth resolving, on a new external
   population if the goal is independent generalization. Do not automatically
   multiply every configuration by five folds and three seeds.

All trainable benchmark fits retain **100 x 20,000 draws**, full eligible pools,
batch 64 / microbatch 8 and final-state scoring. A longer convergence pilot
must be an explicit separate budget. Never shorten training to squeeze a job
into a time estimate. Every production allocation gets a measured profile;
each requests one 10 GB MIG, with a combined maximum of four instances.
No experiment proposed here establishes quantum hardware speedup.

For an economical discovery screen, use two fresh training seeds (47,48) and
one fixed grouped development holdout, shared across arms. This is exploration,
not two seeds proving robustness. A five-fold final confirmation is reserved
for a small frozen set. Any new random outer split of these same already-used
corpora measures split sensitivity; it does not create a new independent or
previously unseen corpus. Reserve a truly external test before choosing data
preprocessing and graph/tuning details if an independent claim is intended.

## A. Top-5: distinguish retrieval from probability quality

The existing top-5/NLL findings are exploratory because top-1 selected the
configurations and was the prespecified primary endpoint. First regenerate the
float64 summaries and paired seed/sentence intervals. Retain per-seed results,
leave-one-seed-out means and original negative results. More bootstrap draws
do not replace more trained models or new evaluation populations. The
[prediction review](results/robust-followup-prediction-review/README.md) now
completes this reaggregation and interval check.

A prospective CBOW top-5 study should retain full-task scoring and prespecified
boundary strata. Choosing only noninitial targets after seeing the boundary
top-1 deficit changes the question. The common-target task in block B is a
separate information-control experiment, not a replacement for the original
endpoint.

The saved refinement grid already selects **LR 0.003 under either top-1 or
top-5 for BERT, MLP, CRZ and identity, in both objectives**. Repeating those
unchanged fits just to change the selection metric is not justified. Reuse
that development decision for unchanged arms; spend additional compute on
fresh seeds, controlled targets and missing mechanisms. The finding remains
exploratory because the original primary endpoint was top-1. Changed target
recipes, samplers and new model classes need their own development checks.

For a new top-5-focused study:

- Select LRs by **validation top-5, then NLL, then top-1, then name**, using the
  same three-LR grid (0.0003/0.003/0.01) when new tuning is needed. For unchanged
  tasks/arms, reuse the endpoint-consistent saved development grid above.
  Use one common LR for the circuit/identity contrast within each task. Report
  a separate comparison of independently optimized models if that is the question; do
  not attribute its whole difference to adding an entangler.
- Keep cross-entropy training. Top-5 is a nondifferentiable evaluation/selection
  endpoint, not an interchangeable smooth training objective.
- Prespecify the primary pair and objective. If both causal and bidirectional
  top-5 are primary, keep nominal family-wise correction across those two.
  Top-1/NLL and all additional pairs remain secondary. Choose a useful effect
  threshold before new test scoring; 0.5 percentage points is a proposed
  planning value, not a retrospectively established equivalence margin.
- Inspect the top-5-minus-top-1 gap and, using saved top-k IDs, target rank
  within the five retrieved words. That is not full MRR: ranks below five are
  censored. Full vocabulary MRR, confidence calibration, ECE and entropy need
  additional logits/probability summaries; current top-k/NLL archives do not
  contain enough information to invent them.
- Use target-frequency buckets built from training counts and fixed sentence
  length/source buckets for error analysis. Treat these as descriptive unless
  subgroup hypotheses and multiplicity handling were fixed before testing.
  Token-weighted top-5 and sentence-macro top-5 answer different questions;
  report both explicitly rather than silently changing the estimand.

## B. Classical NLP comparison on identical targets

Keep two clearly labeled tracks: controlled from-scratch word prediction, and
eventual pretrained/downstream semantics. They cannot share a single headline
claim about training-data efficiency or quantum advantage.

Controlled neural references: MLP-32, BERT-32, a tied word-level bidirectional
LSTM with approximately matched total capacity, q8 CRZ and q8 analytic identity.
Add a smoothed causal n-gram reference using training-only counts and the same
maximum left context. Smoothing and model order need validation, as emphasized
by [Chen and Goodman](https://aclanthology.org/P96-1041/). Do not count repeated
seeds of a deterministic count model as independent fits.
The count reference scans the full training pool; that is not identical to the
neural replacement-draw exposure or optimizer budget. Label that difference,
or add a count reference built from exactly the same sampled context/target
multiset when exposure matching is the question.

For all neural models use the same word IDs, full-word CE, target set, paired
draw lists, eligible pool, optimizer batch and evaluation examples. Log total
and active parameters, FLOP/profile estimates, full allocation time, memory and
quality. Near-capacity, equal-update/exposure and equal-compute comparisons are
separate results. For equal compute, compare quality at measured training-time
checkpoints; do not pretend it is the same intervention as equal epochs.

A recurrent CBOW reference must actually receive both sides of the target.
The forward state at a middle mask in a unidirectional LSTM would not. Define
the gap, left/right padding, bidirectional combination and tied output head
before running. The original
[BERT paper](https://aclanthology.org/N19-1423/) includes pretraining and a
different tokenization/objective recipe; our small scratch MLM remains a
controlled architectural comparison, not reproduction of original BERT.

Fix the target-position confound in the new comparison: every task uses the
same sentence groups and **positions > 0**, including training and validation,
with hashes of `(group, position, target)` saved per split. Keep the full active
sentence pool. Compare L4 (four preceding words) with L2R2 (two left/two right)
as two specified tasks. To isolate adding right context, add an L2-only control
in exactly the same four slots/mask position as L2R2, padding its right slots.
Comparing L4 against L2R2 alone changes both left and right information.

Suggested neural development block: five neural models x three LRs x two seeds
x two tasks = **60 fits**, plus the deterministic n-gram reference. The
L2-only information control on the four existing main models adds **8 fits**.
The new recurrent model and target recipe need fresh profiles before an ETA.
Do not silently reuse old fits with different target populations or splits.

## C. Hamiltonians: two different experiments

### C1. Gate-generator mechanisms

At q8/depth2/init std0.05 compare identity, local-only, frozen CRZ, trainable
CRZ, ZZ and XX, with identical encoder/head initialization, input/sampler and
readout. Include MLP-32 and BERT-32 quality/cost anchors. Keep LR0.003 fixed in
this mechanism block. A claim about the best tuned architecture would require
equal LR search, not just interpreting its fixed-LR result as optimal.

Eight models x two seeds x two tasks = **32 development fits**. Identical main
anchors can be reused from block B only if their complete source, split,
target-population, sampler, LR, seed and budget fingerprints agree. Do not
combine this with a depth/width/data Cartesian expansion.

The previous q8 medians imply roughly **44 allocation-hours**, or 11 ideal
four-slot hours before tails/pilots, for that block alone. A rough 12–16-hour
wall allowance is a proxy, not a measured ETA for the new protocol. Local-only
unitaries have the same function class as identity followed by an unrestricted
linear decoder; optimization trajectories can still differ. CRZ includes a
local-Z term in addition to ZZ, so it is not the same Hamiltonian as RZZ.

### C2. A data-derived graph Hamiltonian

The existing context matrices/gate generators do not define a learned semantic
Hamiltonian. Build a word co-occurrence graph from **training sentences only**,
using a declared context radius, symmetric nonnegative edge weights and zero
self-edges. First use raw co-occurrence; PPMI weighting is a later separately
validated choice. No validation/test graph edges or frequency statistics.

For adjacency A and degree D, a candidate is
`H = diag(degree > 0) - D^(-1/2) A D^(-1/2)`, with inverse zero on isolated nodes.
This is real symmetric, has defined isolated-node behavior and bounded scale.
Vocabulary words index basis states, not individual qubits. A 10,864-word graph
needs a 14-qubit basis embedding if interpreted as a register. Padding states
are isolated; probabilities are normalized over valid words.

With context distribution p0, set the initial amplitude to sqrt(p0). Compare:

1. Identity (`t=0`), giving p0 under the same amplitude/Born readout.
2. Coherent propagation `exp(-i t H) sqrt(p0)` followed by squared magnitudes.
3. Classical heat propagation `exp(-tau H) sqrt(p0)`, squared and normalized,
   isolating unitary versus dissipative propagation under a common readout.
4. A classical probability-flow diffusion on the same graph, plus the strong
   learned MLP reference. State its generator/normalization explicitly; it is
   not automatically the same operator/readout as the amplitude controls.

Use the same declared frequency smoothing for zero-probability outcomes and
validation-selected positive time/smoothing grids, with equal tuning effort.
Test Hermiticity, norm conservation for coherent evolution, isolated nodes,
t=0, global-energy-shift invariance and sparse/dense agreement on tiny graphs.
The continuous-time walk idea has a long literature, including
[Farhi and Gutmann](https://arxiv.org/abs/quant-ph/9706062); applying it to this
training graph is our new hypothesis, not a result established by that paper.

Prototype the algebra on tiny synthetic graphs, then profile **sparse**
matrix-exponential actions at the full vocabulary. Do not form a dense
16,384-square exponential per context or reduce the real corpus/vocabulary and
call it a full benchmark. Cache repeated contexts with measured memory limits.
This is a research implementation gate; no honest DGX ETA exists yet. Any
classical simulator quality gain is not evidence of quantum runtime advantage.

## D. Plateau: replicated batches and heads

`scripts/probe_nlp_trainability.py` now accepts `--batch-seeds` and
`--head-seeds`, retaining the historical default 619 for each. Proposed CUDA
grid: q2/4/8/12, depth2/8, std0.05/1, **three batches x three encoder/head seeds
x 32 circuit seeds**, separately for each objective. This is 144 conditions /
4,608 gradient evaluations per objective, not 4,608 training fits.
Proposed batch seeds are 619/620/621 and head seeds 619/719/819, retaining the
old batch/head 619 condition as a direct implementation check.

Within each `(batch, head, q)` condition, only circuit initialization varies;
non-circuit weights remain identical across depths/scales. Shared circuit seeds
pair conditions rather than making them independent. Save selected train IDs,
raw gradients, losses, hashes and the executed analyzer alongside the frozen
training source snapshot. Copied execution can preserve strict old manifests.
The probe reads training fold 0 only and no validation/test labels.

The old 16-condition CUDA probe took about 28 seconds. Linear replication gives
about **4.2 minutes per objective**; allow roughly 5–15 minutes per objective
until measured. Use one MIG for this diagnostic, preserving the overall cap.
Actual new CUDA performance is unverified; only tiny CPU replications are tested.

Report initial-gradient variance within condition, variability across batches /
heads, first-RX and all-parameter summaries, structural zeros and raw scale.
Do not pool raw gradients from different heads into one variance and call it
variance over circuit initialization. Register size still changes encoder/head
shape, so the current design is conditional finite-size evidence, not a clean
isolation of all qubit effects or proof of asymptotic scaling. The assumptions
in [Cerezo et al.](https://arxiv.org/abs/2001.00550) must be checked before
applying theorems to learned NLP CE. Top-5 plateaus and loss-gradient plateaus
are different observables.
Initialization probes also do not diagnose what happens later in optimization.
If that becomes the question, evaluate fixed train batches at saved epochs
25/50/100 and compare active-gradient scales with quality. The existing runs
do not retain all those weight snapshots; histories alone cannot reconstruct
their past gradient distributions. A new run must save that checkpoint schedule
explicitly instead of implying those measurements already exist.

## E. Data mixture and sampling on a fixed evaluation population

Build one full-union sentence partition before changing training mixtures.
Preserve exact/near-duplicate grouping policy across all arms; save source
membership and report shared sentences explicitly. Never let a sentence from
one source enter training while its duplicate from the other enters test.
Report natural-population and source-stratified test scores on the same heldout
sentences. A source-only corpus with a new test split answers a different question.

Distinguish:

- **Uniform token sampling**: existing training chooses one eligible target
  uniformly, so long sentences contribute more draws.
- **Uniform sentence sampling**: choose a training sentence uniformly, then an
  eligible target in that sentence uniformly. Target probability is
  `1 / (number_of_train_sentences * eligible_targets_in_sentence)`.
- **Source-mixture token sampling**: keep unique target IDs and fractional
  source membership. Set a=1 for phrases-only, b=1 for cleaned-only, and
  a=b=0.5 for shared sentences. For target i use
  `q(i) = p*a(i)/sum_train(a) + (1-p)*b(i)/sum_train(b)`, at p=0.25/0.5/0.75.
  These p values retain all union examples as eligible. Shared sentences have
  one identity; their two fractional contributions are combined in q(i).

Here p is the mass of the two weighted sampling components, not necessarily the
fraction of draws from each mutually exclusive source bucket. Report the
realized phrases-only, cleaned-only and shared proportions as well.

Source-balanced *sentence* sampling would be another policy, not equivalent to
this source-balanced *token* policy. The current loader's `sampling=balanced`
only affects an actual `max_sentences` subset; with the full pool it does not
change epoch draw probabilities. The benchmark training loop must implement
the proposed distributions explicitly, not just rename a loader flag.

All policies keep the same 20,000 draws/epoch, number of updates, vocabulary,
weights initialization and evaluation target IDs. Save per-epoch target-ID
hashes, realized source/token/length/frequency histograms, unique exposure,
replacement counts and sampler RNG state. Reuse paired random uniforms where
appropriate, while acknowledging that changing the distribution intentionally
changes the actual sampled targets. Test weight normalization, positive support,
source overlap, empirical frequencies and interruption/resume parity.

Suggested block: four main models x five sampling policies x two seeds x two
tasks = **80 development fits**, including 16 natural-sampling anchors that can
be deduplicated when fingerprints agree. Current timing proxies suggest around
60 allocation-hours / 15 ideal four-slot hours before profiling and tails;
source-policy CPU preparation overhead is unmeasured. Do not cross these
policies with every ansatz/LR in the same first study. After discovery, choose
the scientifically relevant policy contrasts rather than the largest test gain.

## What must exist before submission

The next neural/data runner needs validated common-target preparation,
source-membership artifacts, explicit epoch sampler settings with RNG resume,
top-5 selection/report metadata, the recurrent reference, timing profiles and
deduplication by complete experimental fingerprints. The graph experiment has
its own algebra/sparse-compute gate. These are implementation requirements,
not claims that the old launcher already supports the designs above.

Run the small diagnostic first, then size a combined batch from measured costs
and the corrected top-5 analysis. Preserve atomic commits, original artifacts,
strict source hashes, raw per-example metrics and the four-MIG ceiling. No
jobs are submitted by this document. The neural/data blocks remain designs
until those implementation checks are complete.
