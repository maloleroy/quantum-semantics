# Full causal and bidirectional studies — protocol design, updated 2026-10-22

User scope: **study both left-only (GPT-like information) and both-sided
(BERT-like information) prediction fully**. Every substantive research block
below has both tracks. A weak or positive effect in one does not cancel the
other. This document supersedes the prioritization in the older
[next-studies proposal](BENCHMARK_NEXT_STUDIES.md); it does not relabel old fits.
The [completed 32+8-fit audits](results/robust-causal-controls/README.md) supply
the current evidence. This is a research design, not a claim that all proposed
model/sampler/graph implementations or launchers already exist.

**Completed 2026-10-22:** the one-MIG microbatch pilot and eight L4 mechanism
fits passed the new export audit. No resubmission is needed. The
[post-causal plan](BENCHMARK_POST_CAUSAL_PLAN.md) specifies the next 16 micro 64
trajectory checks, then focused architecture/coordinate work. It supersedes
the initial launch priorities below. GPT/classical-feature CPU prototypes exist;
runner/resume/CUDA integration and later sampler/continuation/graph modules
remain unfinished. The completed pilot did not adopt a new microbatch.

## Common protocol and the two estimands

| Property | Causal track | Bidirectional track |
| --- | --- | --- |
| Primary task | Predict word t from four preceding words, L4 | Predict masked word t from two preceding and two following words, L2R2 |
| Future information | Never supplied, including preprocessing/features | Supplied as explicitly allowed context; target itself excluded |
| Transformer reference to add/retain | Actual small causal decoder, GPT architecture family | Existing small scratch BERT masked-token model |
| Native extension | Longer prefixes and eventual autoregressive likelihood/generation | Wider masked contexts and eventual representation/downstream evaluation |
| Primary within-track ranking | Validation top-5, retain top-1 and full-word NLL | Same endpoints; independently reported and tuned |

Keep the common positive-position target protocol for controlled comparisons:
same vocabulary, sentence groups, target-key hashes and paired training draws.
Native first-word/BOS modeling and all-position MLM are later, separately labeled
extensions. Do not silently mix those populations into the controlled results.
Masked conditional NLL is not autoregressive sentence likelihood: do not compare
its exponentiation to GPT perplexity as though they normalized the same model.
The present BERT baseline with left-only input is a useful historical reference,
but has full attention over its available context, not GPT's causal mask.

All trainable base fits retain **100 epochs ×20,000 draws**, batch 64, full
eligible pools, Adam reset per fit, full final validation and fixed final 100.
Keep monitor sampling separate from full evaluation. Report trainable/total
parameters, active circuit parameters, exposure, update counts, memory,
fit/wrapper/allocation times and quality for each track. Word IDs/vocabulary
are presently transductive; a train-only-vocabulary/OOV protocol is separate.

After the four-slot core, study context length in paired blocks: L4/L8/L16
versus L2R2/L4R4/L8R8 at equal total visible words, with unchanged target
populations. This tests each track's context scaling; it changes model input
dimensions and Transformer position capacity and needs new profiles/tuning.
Do not cross this width series with every sampler, qubit count and gate family.

Pair the encoder/head initialization independently from circuit initialization,
and log both states, split IDs and sampler fingerprints. Current depth/width
comparisons can consume different numbers of RNG draws before initializing
the head; same integer seed alone is not sufficient evidence of pairing.
Do not change source guards or update the frozen DGX trainer in place.
Use copied extensions with hashes, or an explicitly isolated checkout and
new environment/provenance. Old artifacts remain readable and resumable.

## 0. Close the missing causal mechanism block first

**Completed; retained here as the original protocol.** Eight fits: L4 × {q8 local, frozen CRZ, XX, ZZ} ×seeds 47/48,
depth 2/std 0.05/LR 0.003. Reuse the existing four L4 anchors and all eight L2R2
arms only where source, data, sampler, initialization and budget match.
The resulting paired matrix has eight models in each track, two seeds each;
the completed L2pad intervention remains a supplementary control.

Questions: does frozen CRZ match trainable CRZ in each track? Does ZZ improve
on identity in the causal setting too? Does entanglement add anything beyond
local rotations? Report seed-wise effect sizes, not one pooled task ranking.
Selection of ZZ after seeing L2R2 is exploratory; do not count a new L4 result
as independent confirmation of the earlier bidirectional claim.

Actual wrappers used 15.357 MIG-hours /3h52m33s, peak four, versus fresh forecast
16.391 MIG-hours /4.152h. Pilot/queueing/packaging are excluded. See the
[audit](results/robust-causal-controls/README.md) for endpoint-specific results.
The old drivers must not be used to repeat this completed block.

## 1. Measure useful optimizations before multiplying fits

**Pilot completed; adoption pending full trajectories.** One waited `sbatch` job, **one 10 GB MIG**, both tasks and representative models:
microbatch 8/16/32/64 with unchanged optimizer batch 64, fixed inputs and initial
weights. Test full and final partial batches. Compare forward values, loss,
all accumulated gradients and one Adam update; then synchronized CUDA timings
with warmup/repeats and peak memory. For dropout-bearing BERT, random-mask
consumption and floating reductions can differ; characterize this explicitly.
Retain deterministic no-dropout parity checks and normal training profiles.

The large cost difference between q8 circuits and analytic identity makes
launch/accumulation overhead a better first target than adding qubits. Do not
change the number of optimizer steps or shorten epochs as an optimization.
Any adopted microbatch change is a new declared execution recipe. Keep a
small trajectory comparison before using it in future matched studies.

Recorded one-update checks passed all 64 conditions. Micro 64 training steps
were 4.9–7.7× faster; whole-fit speedups and stochastic long-run equivalence were
not measured. Next 16 full-budget matched fits and fresh evaluation profiles
are specified in the [post-causal plan](BENCHMARK_POST_CAUSAL_PLAN.md).

## 2. Rigorous classical architecture and feature controls, in both tracks

Core references per track: ordered MLP-32, a capacity-near Transformer of the
correct architecture, q8 identity, q8 CRZ, and one simple classical nonlinear
feature replacement. Record best-tuned quality separately from fixed-recipe
mechanism effects. The nonlinear replacement must share the hybrid's learned
lookup, angle encoder, feature dimension 24, decoder and word head; compare
a small nonlinear map of the 16 encoded angles against circuit XYZ features.
Account for its extra parameters and include a capacity-matched variant if
the difference is material. A generic MLP-32 is a quality anchor, not this
feature-matched control. Local-only rotations already share the identity
function class under an unrestricted decoder; a slower statevector simulator
is not a stronger classical reference.

The causal decoder receives only left words, uses triangular attention and
predicts the next word from the final left state. Do not accidentally use the
library's shifted-label loss on already paired single-target examples. Padding
positions and vocabulary exclusions must be explicit. Test future-token
perturbations, target removal, padding, attention masks, tied heads, gradients
and exact checkpoint/RNG resume on tiny data. BERT predicts from its middle
MASK. Their query-token, normalization/dropout and parameter-count differences
must be reported; they are not a pure attention-direction ablation. A separate
same-backbone attention-mask ablation can isolate directionality if needed.

Use the same LR grid .0003/.003/.01 and equal tuning opportunities when claiming
best-tuned performance. A manageable design is five models ×three LRs ×one
development seed ×two tracks = **30 recipe cells**, reusing truly identical
existing development cells. This is LR exploration, not a robustness result.
Freeze choices, then confirm five models ×three fresh seeds 49/50/51 ×two
tracks = **30 fits**; do not tune on their confirmation scores. If this is too
costly after profiling, reduce the number of models/questions explicitly,
never training exposure. Historical LRs can be reused for unchanged mechanisms,
but that alone is not equal tuning of new architectures or samplers.

The 30-cell designs above are optional full-family blocks, not the next automatic
submission. The current smaller proposal adds GPT, classical features and
capacity-near MLP-16 (ten fixed-recipe fits, or fifteen one-seed LR cells for
the five additional model/task combinations). Reuse compatible old tuning only where
it supports the actual comparison; follow with a short frozen contrast list.

Add train-only smoothed count references before expensive neural expansion:
causal backoff n-grams, and a normalized gap-context/backoff conditional count
model for the bidirectional task. A left-only n-gram is not a matched
bidirectional baseline. Count models trained on the complete pool and those
trained on the exact sampled multiset answer different exposure questions;
label each. Deterministic references do not need repeated "training seeds."

## 3. QCSE representation and readout, in both tracks

Separate this from the learned hybrid. For the current independent-bit word
decoder, with valid word code b_w and marginal probabilities p(x):

```
logit_w = sum_j [b_wj log p_j + (1-b_wj) log(1-p_j)]
        = constant(x) + b_w · logit(p(x)).
```

Word softmax removes the common constant. It is exactly a **fixed binary-code
linear head on q log-odds features**, including when some bitstrings are invalid.
The code limits available output directions; it is not the joint Born
distribution of the circuit, and bitwise BCE is not valid-word CE. New tiny
tests verify probability/gradient equivalence against the actual adapter and
show identical marginals can come from different joint distributions.

Design a two-encoder ×two-head factorial in each task: paper context/state
encoder versus a declared classical feature encoder; fixed binary-code head
versus learned word head. Match q-dimensional features and evaluation targets,
report head capacity, and retain analytic encoding-only/frequency controls.
That is **16 discovery fits** at two seeds across both tracks, before any
extra encoding variants. Profile q14 simulations first; do not quote q8 costs.
Changing a learned decoder and adding many word parameters changes model
capacity; success would diagnose a bottleneck, not prove quantum advantage.
The old ID permutation changed encoder IDs while retaining target codes; it
does not isolate output-code assignment. A new code assignment control must
declare which mapping changes and apply it consistently.

Also distinguish the **joint Born distribution conditioned on valid word
codes**, using the same paper state. It retains correlations discarded by
independent marginals without adding a learned word head. Compare it with
encoding-only/product controls and record valid-word mass/clamping. It is a
third readout question, not the same object as the learned-head factorial;
avoid multiplying every encoder/ansatz unnecessarily. A final diagonal layer
remains invisible even to joint computational-basis probabilities.

## 4. Data mixture, sampling and exposure, in both tracks

The verified training fold contains 650 phrase-only sentences/4,036 positive
targets, 128,739 cleaned-only/1,010,044 targets and one shared sentence/five
targets. Phrase supervision is only about 0.4% under uniform-token sampling.
The validation fold has 159 phrase sentences/998 targets; fine-grained source
claims will have much less support than whole-corpus scores suggest.
See [source composition](results/robust-target-controls/source-composition.json).

Start with models MLP, identity and a frozen selected circuit per track. Compare
uniform-token with uniform-sentence replacement and one explicit source mixture
(10% phrase mass). Two new policies ×three models ×two seeds ×two tasks =
**24 incremental fits**, with unchanged uniform-token anchors reused only when
fingerprints agree. Consider 50% phrase mass only after that dose-response
check; it is a large intervention relative to the observed 0.4% baseline.

Split/deduplicate before constructing pools. For shared sentences, distribute
source membership fractionally (1/k per source), normalize within each source,
then combine the requested source weights into one probability per target.
A shared sentence must not become two dataset entries. Sampled repeats remain
allowed. For uniform-sentence sampling choose a sentence then a positive target
uniformly inside it; all eligible examples retain nonzero probability.

Without importance weights, a new sampler deliberately changes the training
objective. With p/q weights it can instead estimate the old objective with
different variance. Do not silently conflate these two experiments. Hash the
probability vector, draw stream/seed and membership rule; keep validation and
target populations fixed. Report source/frequency/length/position strata and
both token-weighted and sentence-macro scores. The current compact export lacks
token ranks for these strata: obtain them from DGX raw predictions first.

## 5. Optimization, qubit scaling and longer budgets, in both tracks

Use actual NLP loss/gradients at initialization and later training states.
The replicated 9,216-evaluation initialization probe is complete but uses tiny
eight-target batches and varying head shapes. The new 100-epoch q8 histories
show active gradients and continuing improvement, not a qubit-scaling theorem.

For q4/8/12, depths 2/8 and scales 0.05/1, first run a no-training gradient probe
with a fixed-dimensional surrounding encoder/head and an explicit q-dependent
projection; keep its normalization controlled. Use optimizer-sized 64-target
batches, multiple training batches, heads and circuit initializations in both
tasks. Log signed mean, variance, RMS, per-parameter distributions, loss,
gradient noise across batches, clipping and actual update/weight ratios.
Separate structural zero gradients, saturation, overfitting and noisy minibatch
gradients from exponential concentration with qubit count. A fitted slope over
three small q values is descriptive, not an asymptotic proof. Do not add noise
models until ideal-simulator behavior is understood.

A small convergence block: {MLP, identity, CRZ} ×two fresh seeds ×two tasks =
**12 prospective fits to 200 epochs ×20,000**, recording full validation at 100
and 200 and preserving optimizer/sampler state between them. Monitor curves
are not final full validation. The old 100-epoch artifacts cannot be silently
relabelled or extended by altering their manifests. An explicit continuation
protocol with provenance, checkpoint validation and prospective endpoints is
required. Use new trajectories if that cannot be guaranteed. Budget the full
200 epochs, and keep the result separate from the fixed 100-epoch comparison.

## 6. Hamiltonian construction, in both tracks

Gate generators and a semantic graph Hamiltonian are distinct. CRZ includes
a local Z field plus ZZ; XX and ZZ trials test gate families, not a learned
semantic Hamiltonian. A final diagonal block cannot affect Z-only readout,
but the hybrid's XYZ readout can observe phase changes.

For the implemented directed chain with free per-layer RZ fields, CRZ and ZZ
have exactly the same function class: map each CRZ angle g to ZZ−g/2 and add
g/2 to its target's RZ field. Their default initialization and coordinate-wise
Adam differ. The [new state/gradient probe](results/robust-causal-controls/gate-coordinate-probe.json)
checks this conversion. The next gate experiment should isolate those effects,
not claim a more expressive Hamiltonian from the current ZZ ranking.
Product-state depth 2 nearest-neighbor chains also have Schmidt rank≤4 across
every cut; exact-MPS simulation is a separate implementation/profile candidate.

Build a symmetric nonnegative train-only word co-occurrence graph with a
declared radius, zero self-edges and an explicit isolated-node convention.
Use a normalized graph Laplacian with zero rows/diagonal for isolated nodes.
Compare identity, unitary exp(-itL), heat exp(-tL) and a matched classical graph
feature/filter under the same input features and prediction heads. A symmetric
graph may be shared when its construction uses the same training sentences;
task-specific initial context states still use only allowed left/both words.
If directional causal transitions are studied, a nonsymmetric matrix is not
itself a valid Hermitian Hamiltonian: define the Hermitian construction.

Verify Hermiticity, isolated nodes, padding, t=0, norm behavior, sparse/dense
agreement on a tiny graph and absence of validation/test graph edges. Use
sparse exponential action; do not materialize a dense 16,384² propagator or
assume gate-simulator timing applies. Profile setup, graph storage, evolution,
training and repeated inference before choosing a batch. Time/weight/PPMI
tuning and graph data exposure belong in the comparison cost. No full-data
ETA is established yet, and a quantum walk does not by itself imply useful NLP
semantics or computational advantage.

## Robustness, selection and launch policy

Prespecify a primary within-track contrast for the next confirmation; top-5
primary with top-1/NLL secondary, and a useful effect threshold agreed before
new evaluation. If both tracks support a joint claim, control the two-primary
family, for example with Holm-adjusted valid paired tests. The many exploratory
cells are not independent hypotheses with confirmatory intervals. With too
few seeds, report effect ranges and uncertainty limitations explicitly rather
than claiming bootstrap precision. Five folds ×three seeds is not the default.
Additional grouped splits measure split sensitivity; they do not create an
unseen corpus. Reserve a genuinely external/source/template-held-out test
before tuning new preprocessing. Keep both exact and near-duplicate audits.

Word prediction does not establish semantic similarity, entailment or useful
generation. Later native GPT-like likelihood/generation and BERT-like
representation/downstream studies need their own held-out protocols. Pretrained
BERT/Qwen comparisons are an external-utility track with disclosed pretraining
and tokenization/exposure, not an equal-data advantage claim.

**Next order:** one-MIG optimization profile, eight missing L4 mechanism fits,
then the paired classical/representation study in both tracks. Data/sampling,
scaling and graph modules follow as explicit measured batches, not a Cartesian
product. Maintain a combined ceiling of four one MIG of 10 GB tasks, sequential
waited arrays and source/runtime fingerprints. No interactive `srun` and no
parallel manual submissions around the controller. Newly designed modules
require implementation/parity checks and fresh CUDA profiles before launch
commands or reliable ETAs can be issued.
