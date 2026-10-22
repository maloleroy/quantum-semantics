"""Assemble the final report from committed quantitative result records."""

# Keep long Markdown citations in the report template intact.
# ruff: noqa: E501

import json
from pathlib import Path

import numpy as np

from .analyze import LABELS, aggregate_test, records

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def ci(row):
    return f"{row['difference']:+.5f} [{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}]"


def build():
    summary = read("summary.json")
    if not np.isclose(summary["walk"]["mean"]["nll"], 4.2964, atol=5e-5):
        raise ValueError("New outcomes require revising the report narrative before rendering")
    paired = read("paired_comparisons.json")
    mechanisms = read("mechanisms.json")
    corpus = read("tatoeba_data/dataset.json")
    small = read("phrases_data/dataset.json")
    theory = read("theory_diagnostics.json")
    semantic = read("semantic_probe.json")
    all_runs = [
        r
        for stage in [
            "screen",
            "refinement_screen",
            "main",
            "depth",
            "phrases",
            "full_classical",
            "calibration",
            "born_control",
        ]
        for r in records(stage)
    ]
    worker_minutes = sum(r["seconds"] for r in all_runs) / 60
    main_table = (RESULTS / "test_table.md").read_text().strip()
    intervals = [
        "| Comparison (first minus second) | Test NLL difference [95% interval] |",
        "|---|---:|",
    ]
    for label, key in [
        ("Graph walk - QCSE categorical", "walk - qcse_nll"),
        ("Graph walk - QCSE joint Born", "walk - qcse_born"),
        ("Graph walk - graph removed", "walk - no_graph"),
        ("Graph walk - scrambled graph", "walk - shuffled"),
        ("Graph walk - classical diffusion", "walk - diffusion"),
        ("Quantum residual - plain bigram", "predictive_residual - bigram"),
        ("Quantum residual - mixed bigram", "predictive_residual - mixed_bigram"),
        ("Full classical embedding - bigram", "full_embedding - bigram"),
    ]:
        intervals.append(f"| {label} | {ci(paired[key]['nll'])} |")
    interval_table = "\n".join(intervals)
    depth = records("depth") + [r for r in records("main") if r["config"]["method"] == "walk"]
    depth_lines = [
        "| Alternating layers | Parameters | Validation NLL, mean ± seed SD |",
        "|---:|---:|---:|",
    ]
    for layers in [1, 4, 8, 16, 32]:
        scores = [r["validation"]["nll"] for r in depth if r["config"]["layers"] == layers]
        depth_lines.append(
            f"| {layers} | {2 * layers + 1} | {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f} |"
        )
    depth_table = "\n".join(depth_lines)
    small_summary = aggregate_test("phrases")
    small_lines = ["| Model | Small-corpus test NLL | Top-1 |", "|---|---:|---:|"]
    for method in ["unigram", "qcse_nll", "walk", "embedding", "bigram", "predictive_residual"]:
        row = small_summary[method]
        label = "Classical embedding" if method == "embedding" else LABELS[method]
        small_lines.append(
            f"| {label} | {row['mean']['nll']:.4f} | {100 * row['mean']['top1']:.2f}% |"
        )
    small_table = "\n".join(small_lines)
    mech = {}
    for method in ["walk", "predictive_residual"]:
        rows = [v for k, v in mechanisms.items() if k.startswith(method + "_count")]
        mech[method] = {k: float(np.mean([v[k] for v in rows])) for k in rows[0]}
    resource_lines = [
        "| Toy vocabulary | Binary qubits | Compiled depth | CX gates |",
        "|---:|---:|---:|---:|",
    ]
    for r in read("circuit_resources.json"):
        resource_lines.append(
            f"| {r['vertices']} | {r['qubits']} | {r['compiled_depth']} | {r['compiled_counts']['cx']} |"
        )
    resource_table = "\n".join(resource_lines)
    sem_lines = ["| Representation / similarity | SimLex Spearman correlation |", "|---|---:|"]
    for label, pattern in [
        ("Fixed-walk pure-state fidelity", "static_walk_state_fidelity"),
        ("QCSE singleton probability overlap", "qcse_nll_probability_overlap"),
        ("Train-count PPMI row cosine", "ppmi_cosine"),
        ("Classical bigram probability overlap", "bigram_probability_overlap"),
        ("Graph walk probability overlap", "walk_probability_overlap"),
        ("Graph walk context-dependent state fidelity", "walk_state_fidelity"),
        ("Predictive quantum state fidelity", "predictive_residual_state_fidelity"),
        ("Full classical token table cosine", "embedding_table_cosine"),
    ]:
        scores = [
            v["spearman"]
            for k, v in semantic["results"].items()
            if k == pattern or k.startswith(pattern + "_s")
        ]
        values = [v for v in scores if v is not None]
        value = "Undefined (constant)" if not values else f"{np.mean(values):.3f}"
        if len(values) > 1:
            value += f" ± {np.std(values, ddof=1):.3f}"
        sem_lines.append(f"| {label} | {value} |")
    semantic_table = "\n".join(sem_lines)
    text = f"""# Hamiltonian encoders for causal quantum semantics

Research report · 16 September 2026 · QCSE branch `codex/hamiltonian-encoder-research`

## 1. Findings and recommendation

**The proposed Hamiltonian construction solves the vocabulary-label symmetry problem.
The experiments do not establish a predictive benefit specific to quantum evolution.**
A corpus graph gives the model much more useful structure than QCSE's numerical-ID
angles, but classical use of the same structure is stronger in these experiments.

The 16-layer graph walk obtains **4.2964 test NLL**, versus **5.1826** for the original
QCSE encoding trained with categorical loss. Lower NLL is better. However, classical
undirected diffusion on the same graph obtains **4.0350**, a smoothed last-word bigram
obtains **2.9679**, and a 16-layer classical embedding model trained on the full training
partition obtains **2.4388**. The graph walk's 8.72% exact accuracy is even below the
unigram control's 9.18%, despite its better probability quality. It is not yet a useful
next-word model by that metric.

The most promising quantum variant starts from square roots of classical bigram
probabilities and adds a small context-dependent walk. It reaches **2.9662 NLL**.
But a classical mixture containing just **0.2473% unigram probability** also reaches
**2.9662**. Their paired NLL difference is **+0.000014**, with a 95% interval of
**[-0.000851, +0.000880]**. Thus the small gain over the plain bigram does not demonstrate
a quantum improvement. This control materially changes the interpretation of the result.

My recommendation is to **keep the graph-based, label-equivariant design principle**,
replace the original arbitrary-ID angle map, and prioritize **directional predictive
statistics and explicit context order**. Retain a quantum residual as an experimental
component only if it beats its own identity/calibration control. Avoid investing in a
large hardware compilation until that comparison succeeds.

This study completed **{len(all_runs)} optimization runs**, including 16-layer main models,
a 1/4/8/16/32-layer sweep, three seeds for confirmation, two corpora, exact circuit checks,
and a restricted external similarity probe. Detailed derivations are in [THEORY.md](THEORY.md);
the original draft is preserved in [PROPOSAL.md](PROPOSAL.md).

## 2. What is wrong with arbitrary indexing, and what should remain ordered?

In the supplied [QCSE v2 paper](https://arxiv.org/abs/2509.05729v2), word indices become
angles, a context matrix is flattened into rotation blocks, and the circuit produces
marginal qubit probabilities. This creates several different design choices: vocabulary
numbering, flattening order, gate order, and sentence order. Only the last of these is
necessarily linguistic. Frequency-sorted IDs contain a frequency prior, but neighboring
IDs need not denote neighboring meanings.

Renaming vocabulary entries consistently should leave predictions for named words
unchanged. Swapping words inside a sentence should generally change them. The goal is
**equivariance to vocabulary relabeling**, not invariance to meaningful sentence order.
The Hamiltonian model achieves the former by deriving every matrix from corpus relations.
The original angle formula does not have that structural guarantee.

The present task is GPT-like in its **causal information flow**: predict the next token
from up to eight previous tokens, with no future words in the input. Neither this study
nor the original QCSE ansatz is a Transformer/GPT attention stack. This report evaluates
causal word prediction and associated representations, not a full modern language model.

![Encoding principles](results/encoding_principles.png)

Figure 1. Three independent, small numerical demonstrations. Left: 25 random vocabulary
renamings change the pairwise-fidelity geometry of QCSE's encoding, while consistently
renaming a graph encoder leaves it unchanged to numerical precision. These are toy
structural checks, not language-model scores. Middle: a shared unitary cannot change
pairwise fidelity. Right: the literal repeated-token amplitude sum assigns [a,a,b]
probabilities 0.8/0.2; frequency-preserving loading assigns 2/3 and 1/3.

## 3. Mathematical audit of the proposal

**The symmetry proof is sound.** If a permutation P acts on the context state and on both
Hamiltonians by conjugation, every exponential transforms consistently. The final state
and its Born probabilities are simply relabeled. The independent double-precision check
finds a maximum state error of {theory["permutation_max_state_error"]:.2e}. Tied graph-neighbor
weights must be handled without arbitrary ID tie breaks; the implementation keeps all
ties at the pruning threshold. Classical graph models can have the same symmetry, so
this property is valuable but not uniquely quantum.

**A fixed walk cannot learn a fidelity geometry.** Applying the same unitary to two states
preserves their inner product. Distinct basis words remain orthogonal. Repeating only
the same graph Hamiltonian collapses to one walk whose time is the sum of layer times.
Context-dependent interleaving can change cross-context geometry, because different
inputs then undergo different unitaries. Its semantic usefulness still requires testing.
The measured shared-unitary fidelity change is {theory["shared_unitary_max_fidelity_change"]:.2e}.

**Repeated words change the normalization.** In the draft, amplitudes from repeated
occurrences add before squaring. The correct normalization is the squared norm of the
summed amplitudes, not generally the sum of positional weights. The literal `bag`
control keeps this coherent counting. The main model instead sums weighted token
frequencies first and takes their square roots, preserving the intended empirical mass.

**A position register alone does not recover order.** If the graph evolution acts only
on words and position is ignored at readout, probabilities become a weighted average
over positions. With equal weights this is again a bag of words. Position must affect
operations, weights or measurement. We test explicit recency and a chronological
sequence of token-specific context potentials.

**Association is not a causal transition law.** A fixed real-symmetric Hamiltonian has
reciprocal basis transition probabilities; any fixed unitary gives a doubly stochastic
transition matrix after squaring magnitudes. Natural-language bigrams generally have
neither property. Context-dependent unitaries can escape these fixed-walk restrictions,
but an undirected co-occurrence graph still discards useful direction at construction.

**Some apparent alternatives are redundant.** For the exact normalized Laplacian I-H,
evolution differs from adjacency evolution by a global phase and reversal of time.
Signed trainable times absorb this change. A final diagonal phase is invisible to
computational-basis measurement; each implemented layer therefore applies the context
phase before graph mixing. A generic appended local-qubit ansatz would not automatically
preserve vocabulary equivariance.

**The unary hopping formula needs a factor correction.** For an undirected graph, use
one-half times the sum over unordered edges i < j of H[i,j](X_i X_j + Y_i Y_j). Keeping
one-half while summing both edge directions doubles the hopping strength. Restricting
the corrected Qiskit operator to the single-excitation subspace reproduces the graph
matrix exactly. The full proofs, isolated-node convention and phase details are in
[THEORY.md](THEORY.md).

## 4. Experimental design

### Data and leakage controls

Tatoeba contains **246,072 sentences**, **207,510 distinct normalized sentence groups**,
**2,059,720 tokens**, and 11,355 observed word types. Exact duplicates stay in the same
partition. The split is approximately 64% training, 16% validation and 20% test by
sentence group. Vocabulary, transition counts, co-occurrences and priors are fitted
using training sentences only. Lowercasing and tokenization follow the repository;
there is no subword model, lemmatization or stopword removal.

The main vocabulary has **512 entries: 511 training words plus UNK**. It covers
{100 * corpus["partitions"]["train"]["known_token_coverage"]:.2f}% of training tokens and
{100 * corpus["partitions"]["test"]["known_token_coverage"]:.2f}% of test tokens. The evaluated
test set has {100 * corpus["partitions"]["test"]["sample_unk_target_rate"]:.2f}% UNK targets.
Known-target metrics are therefore reported alongside the full distribution. Perplexity
is for this restricted vocabulary and must not be compared with full-vocabulary LLM scores.

All **1,162,743 training transitions** contribute to graph/conditional statistics.
To bound exact-gradient simulation, optimization uses a fixed random subset of **8,192
training targets**; validation uses 4,096 and final testing 8,192 targets from 7,228
sentence groups. Selection indices, targets and group identities are saved. The full-data
classical reference instead trains on all 1,162,743 targets, using the identical vocabulary,
statistics and held-out examples. It is an additional system reference, not an equal-update
comparison. The 8,192-target classical model is also included.

### Model definitions

The main graph uses distance-weighted causal co-occurrences, symmetrized as W = C + C^T,
with self edges removed, threshold-based 24-neighbor pruning and symmetric normalization
H = D^(-1/2) W D^(-1/2). Count, PPMI and normalized-PMI constructions were screened.
The input distribution c gives a word the sum of 0.7^age over its occurrences, normalized
to total mass one. The initial state has amplitudes sqrt(c). Each of 16 layers applies
`exp(-i gamma[l] diag(c))`, followed by `exp(-i beta[l] H)`.

The trained graph model has **33 scalar parameters**: 16 graph times, 16 context-phase
strengths and one unigram-mixture logit. Main probabilities are a learned classical
mixture of Born probabilities and the positive training unigram prior. This declared
smoothing departure avoids impossible-target log losses; it is not a coherent amplitude
mixture. Graph removal and graph scrambling preserve the rest of the setup.

The predictive variant initializes amplitudes from the smoothed bigram distribution of
the last context word. It starts near identity, with graph times around 0.02 per layer.
An earlier initialization around 0.6 per layer degraded the useful prior. Its matched
classical controls learn only the unigram mixture, or mixture plus temperature. Both
selected their initial checkpoint: mixture 0.002473 and temperature 1.

Classical graph controls use the same 16-layer budget and 33 parameters, combining
stochastic propagation with context restart and a unigram mixture. The undirected
version uses the same pruned W; the directed version uses the bigram transition table.
The classical embedding reference uses a 32-dimensional token table, learned position
weights, 16 residual dense layers and a categorical output: **50,464 parameters**.
Its embedding lookup is mathematically a one-hot-to-linear map.

QCSE controls retain the original fixed exponential encoding and 16 RX/RZ/CRZ ansatz
layers: **416 parameters** on nine qubits. One trains with the original bitwise BCE;
a second trains normalized valid-vocabulary independent-bit likelihood with categorical
NLL. A third uses the **full joint Born distribution**, avoiding the marginal-bit
bottleneck. The 512-class state space needs no invalid-state postselection. These are
three distinct readout/objective controls, not a literal reproduction of the paper's
training results or datasets.

### Optimization and evaluation

All main models use exact differentiable statevector or classical computations, Adam,
30 epochs, batch size 128, seeds 11/22/33, and gradient-norm clipping at 5. Learning rates
are 0.03 for graph/scalar models, 0.01 for QCSE and 0.003 for the classical embedding.
QCSE keeps its 0.001 angle L2 penalty; the other models have no extra L2. Validation NLL,
checked initially and every five epochs, chooses checkpoints. The full-data neural
reference uses 10 epochs, batch size 1,024 and validation every two epochs. Different
learning rates, regularization, parameter counts and data-processing paths prevent
interpreting this as a universal architecture ranking.

The initial 15-epoch screen used 2,048 training/validation targets and one seed. It selected
count graphs and motivated near-identity initialization. Separate selection manifests
were saved before held-out evaluations. Depth and unpromoted variants remain validation-only.
No test scores selected a graph, depth or checkpoint. Exact gradients replace the older
repository studies' SPSA estimates, so earlier numerical scores are not directly comparable.

NLL is mean negative log probability of the correct next word, in natural-log units;
perplexity is its exponential. Top-1 and top-5 use vocabulary probabilities, with fractional
credit for exact ties so arbitrary ID tie breaks cannot affect the score. The paper's
half-matching-bits accuracy is not used as word accuracy.

## 5. Main held-out results

{main_table}

Values are means across three fitted seeds where available; ± is the sample standard
deviation of NLL across those seeds. Perplexities are averaged per seed. The classical
bigram, unigram and pooled controls are deterministic training-count estimates. Known-target
NLL excludes UNK targets but retains the original probabilities, rather than renormalizing
a different prediction task.

![Held-out comparison](results/heldout_comparison.png)

Figure 2. Selected comparisons from the full table. The quantum residual overlaps its
classical mixed-bigram control. Error bars show optimization-seed SD, not uncertainty over
new corpora. The full-training classical model uses more gradient examples; the matched
8k classical reference and matched graph diffusion are listed in the table.

### What the uncertainty supports

{interval_table}

Negative differences favor the first model. Intervals use **3,000 paired bootstrap
resamples of held-out sentence groups**, averaging per-example differences over fitted
seeds before resampling. They are conditional on this corpus, this split and these models;
they do not include retraining on new splits or adjustment for the many exploratory
comparisons. They support the graph-versus-QCSE improvement and the advantage of classical
diffusion over the graph walk. They do not resolve a benefit of the quantum residual over
the one-scalar classical mixture.

The whole-system QCSE comparison still changes encoder, ansatz structure, statistical
preprocessing and, in some controls, readout. The joint-Born QCSE control narrows the
readout confound, but does not turn the study into a pure encoder-only substitution.

## 6. Depth, graph choice and initialization

{depth_table}

Depth helps the alternating model over this range. **32 layers perform best on validation**,
but remain worse there than the 16-layer classical undirected diffusion control. The
primary held-out comparison stays at the prespecified 16 layers. Longer training or
further depth might change results; these experiments do not establish a barren plateau.
The depth sweep holds epochs and per-layer initialization distribution fixed, not total
initial walk time, total FLOPs or total parameter count.

![Depth and learning](results/depth_and_learning.png)

Figure 3. Left: validation performance over three seeds at 1, 4, 8, 16 and 32 layers, with
seed-SD bars. Right: validation learning curves. Starting from a predictive distribution
changes the scale of the problem much more than adding layers to the raw context walk.

At the one-seed screen, count-based recency walks scored **4.3162**, versus **4.5368** for
PPMI and **4.4901** for normalized PMI. The literal coherent bag scored **4.4350**.
Chronological token-specific phase injection scored **4.3033**, only a small change in
that screen and not promoted for three-seed confirmation. These are validation findings,
not held-out rankings or proof that PPMI/order conditioning never helps.

A fixed-walk grid on the main validation set selected time 8 and unigram mixture 0.5,
with **4.4111 NLL**. Alternating context phases improve beyond that limited grid, while
their quantum character remains unnecessary for the best graph-based prediction. Without
context phases, repeating a fixed graph generator would only change its total time.

## 7. What the ablations actually establish

**The corpus-word correspondence matters.** Removing the graph after refitting increases
test NLL by 0.4643. Scrambling graph labels while keeping words/inputs/outputs fixed
increases it by 0.4631 and almost recovers the unigram. This is intentionally different
from consistently renaming every structure, which leaves the model unchanged.

**Coherence is used, but its use is not an advantage claim.** On a fixed random sample
of 256 validation targets, removing H with trained weights frozen changes the walk's
NLL by {mech["walk"]["fixed_weights_graph_removal_nll_change"]:+.4f}. Dephasing after every
fitted graph step changes it by {mech["walk"]["fixed_weights_dephasing_nll_change"]:+.4f}.
These interventions test a co-adapted fitted model; they are not retrained classical
baselines. The separately trained classical diffusion still wins on held-out NLL.

**Context sensitivity is present.** Reversing valid tokens while preserving padding
changes the graph walk's output by mean total-variation distance
{mech["walk"]["mean_reversal_tv"]:.3f} on that diagnostic sample. Its mean pairwise state-fidelity
change from initialization is {mech["walk"]["mean_fidelity_geometry_change"]:.3f}, consistent with
input-dependent evolution escaping shared-unitary fidelity preservation. Neither statistic
by itself measures whether the changed meaning is linguistically correct.

The graph walk retains an average **{100 * mech["walk"]["unigram_mixture"]:.1f}% unigram mixture**.
Its raw Born NLL on the diagnostic subset is
{mech["walk"]["raw_born_nll_clipped_1e30"]:.3f}, versus
{mech["walk"]["validation_subset_nll"]:.3f} after smoothing. Thus the literal unsmoothed proposal
is not the system producing the main score. Norm drift remains a few parts per million
in complex64 training; double-precision reference and Qiskit tests validate the exact
intended evolution independently.

The draft's graph-removal criterion should therefore be strengthened: **a degradation
when the graph is removed shows graph dependence, not a uniquely quantum contribution**.
Require improvements over classical diffusion, predictive identity and calibration
controls before making the stronger statement.

## 8. Smaller-corpus replication and semantic probe

### Original phrases corpus

The second corpus has 1,000 sentences and a full training-derived vocabulary of
**{small["vocabulary_size"]} entries including UNK**. It uses all 3,985 training, 1,004 validation
and 1,246 test targets, with the same 16-layer/30-epoch/three-seed protocol. Test UNK rate is
{100 * small["partitions"]["test"]["sample_unk_target_rate"]:.2f}%. Unlike the capped Tatoeba vocabulary,
this full training vocabulary has no observed training UNK tokens, making unseen-word
prediction especially constrained. Scores across the two corpora should not be compared
as if they represented the same vocabulary or distribution.

{small_table}

The main qualitative ordering survives: the graph walk improves over causal QCSE, while
the predictive prior accounts for most of the stronger quantum variant's performance.
The classical calibration controls were confirmed on Tatoeba only, so the tiny residual
versus bigram difference on this smaller corpus is not an independent quantum-gain claim.

### External word similarity

[SimLex-999](https://fh295.github.io/simlex.html) targets word similarity rather than mere
association. Only **36 of its 999 pairs** have both words in the already fixed 512-entry
training vocabulary. We did not refit vocabulary, train models or choose hyperparameters
on this benchmark. The probe uses singleton word contexts and therefore does not test
contextual word-sense disambiguation.

{semantic_table}

The weak positive graph-walk correlations are interesting enough to motivate a larger
semantic study, but **36 common-word pairs cannot establish a robust semantic advantage**.
Seed SD does not reflect sampling uncertainty over word pairs; several comparisons were
inspected and repeated words make pairs dependent. Do not compare these values with
published full-SimLex scores. Better next-word likelihood also does not imply better
word-similarity geometry, as the predictive models illustrate.

A fixed shared walk has constant zero fidelity between distinct basis words, so its
correlation is mathematically undefined. QCSE's singleton encoding likewise collapses
here: the sole nonzero angle initially drives RX on a plus state and contributes only
a global phase. Its near-constant numerical similarity is reported as undefined rather
than assigning meaning to floating-point rankings. This confirms a limitation already
identified by the earlier repository encoding study.

## 9. Directed quantum encoding and hardware feasibility

A Hermitian source/target dilation preserves causal direction. Build B by taking elementwise
square roots of a classical row-stochastic bigram table, and use the block Hamiltonian
with B and B-transpose off the diagonal, normalized by B's spectral norm. Start in the
source-word block and condition readout on the target block.

At short times, the conditional target probabilities approach the supplied bigram table,
while target-block success falls quadratically with time. The 16-layer directed screen
selected **epoch zero**, with validation NLL **2.9328** on its smaller screen and mean
success **0.000619**, approximately 0.062%. This is about one success per 1,617 attempts
using inverse mean success as a rough summary; the exact expected cost across contexts
is the mean of their inverse successes. It is not evidence that training discovered a
better conditional distribution. No held-out confirmation was claimed for this variant.

![Directed-walk tradeoff](results/directed_walk_tradeoff.png)

Figure 4. An independent eight-word example, not a corpus performance curve. At short
times, a directed dilation reproduces the input classical transition probabilities very
closely, but successful target-role measurements become rare. Longer times change both
the conditional prediction and success probability. Total variation measures distributional
difference, with zero meaning identical.

For the real 512-word count graph, the measured resources are **11,116 undirected edges**,
mean degree **43.42**, and maximum degree **510**, despite 24-neighbor thresholding.
Unary encoding needs 512 physical qubits. Binary encoding needs nine, but the direct
Pauli expansion contains **130,816 nonzero terms** at coefficient tolerance 1e-12.
One unary edge sweep already contains 22,232 XX/YY Pauli rotations, before multiplying
by layers or the product-formula repetitions needed for a target simulation error.

We compiled exact small binary examples, including state preparation and four alternating
layers, into RZ/SX/X/CX gates with Qiskit optimization level 1:

{resource_table}

These are illustrative generic compilations with no device routing/noise and no claim
of optimal gate counts. Statevector infidelity stays below 8e-15. Sparse exponential-multiply
and dense matrix-exponential references agree to {theory["sparse_vs_dense_max_error"]:.2e} in the
independent numerical test. Training uses repeated dense spectral evolution for these
bounded vocabularies; there is no Trotter approximation in the fitted model.

A sparse raw-context superposition can exploit its few nonzero entries. The predictive
initialization, however, loads a dense conditional distribution. Both cases still require
corpus preprocessing and Hamiltonian access. The learned 33 scalars do not account for
the V-by-V classical statistics, stored eigensystem or output-distribution extraction.
Semantic equivariance also does not imply invariant physical compilation cost: relabeling
binary basis states can change locality and routing.

Sparse-Hamiltonian algorithms have meaningful access assumptions; logarithmic qubit count
alone is not a speedup argument. This is a classically tractable single-particle simulation,
not evidence of quantum computational advantage or hardware scalability.

## 10. What to pursue next

**For a stronger causal model now:** replace arbitrary-ID angles with training-derived
directional conditional features, preserve explicit recency/position, and score a valid
categorical vocabulary. The strongest completed system here is the full-training classical
embedding model. Square-root predictive loading is a well-defined quantum-compatible
encoding, but its existing statistical work must be credited to the classical counts.

**For a quantum research contribution:** keep equivariance as an invariant and use a
near-identity quantum component with a precise identity control. Compare against an equally
well-tuned classical transformation of the same inputs, including smoothing and temperature.
Require a repeatable held-out gain beyond those controls before optimizing hardware gates.

**For semantic embeddings:** choose an observable representation or context-dependent
encoding whose similarity can change meaningfully. A shared fixed unitary cannot make
basis words closer in fidelity. Expand lexical coverage and evaluate contextual senses,
not just next-token likelihood. A directed/position-conditioned graph encoder or learned
spectral filtering is a plausible next experiment, not a validated winner here.

The proposal's minimal criteria are therefore only partly sufficient. Vocabulary equivariance
passes; graph dependence passes; context sensitivity passes. The stronger criteria of useful
semantic similarity, superiority over classical graph computation and practical hardware
benefit remain unestablished.

## 11. Limits, reproducibility and deliverables

This is an exploratory controlled study, not an exhaustive architecture search. It uses
one grouped split per corpus, a capped main vocabulary, local context windows, exact
simulation and no shot/noise model. Exact-duplicate grouping does not remove every near-duplicate
or shared Tatoeba translation/template. The supplied CSV lacks provenance metadata beyond
its contents and filename; no stronger corpus-origin claim is made.

Models have unequal learned parameter counts and preprocessing costs. Quantum models use
full-training statistics but only a subset for gradient updates. The full classical reference
has more gradient data; the matched subset and matched graph controls expose that distinction.
Deep models can be sensitive to initialization, learning rate and regularization. Failed
variants do not imply general quantum impossibility, and the tiny semantic subset does not
support a broad ranking of embedding quality.

The recorded runs used approximately **{worker_minutes:.1f} summed worker-minutes** on local CPU,
with one numerical thread per worker and several independent jobs in parallel. This is summed
job time, including per-job preparation/evaluation, not wall time, gate time or a hardware
speedup claim. Logs were kept to completed-job summaries. Dependencies are locked; datasets,
splits, configurations, validation histories, source digests and environment records are saved.

The report's code lives in `experiments/hamiltonian_study/`. [REPRODUCE.md](REPRODUCE.md)
gives the exact commands for screening, confirmation, depth, classical controls and the
small corpus. [PROTOCOL.md](PROTOCOL.md) records the original plan and validation-driven
adaptations. [SOURCES.md](SOURCES.md) explains literature scope. Raw weights and prediction
arrays remain local and ignored; compact numerical records, figures and reports are versioned.
The supplied Tatoeba file is not added to Git.

The validation suite checks train-only vocabulary/statistics, duplicate-safe splitting,
coherent versus frequency counting, graph pruning ties, permutation equivariance, finite-difference
gradients, exact exponentials, sparse evolution, QCSE gate parity, state normalization,
phase invisibility, binary padding, independent Qiskit circuits, and classical identity-control
agreement. All 75 repository tests pass; lint and type checks include the research code.
Small circuit
and float64 checks validate the intended operations; the large runs use complex64.

## 12. Sources

- Varmantchaonala et al. [QCSE, arXiv:2509.05729v2](https://arxiv.org/abs/2509.05729v2), 2026 revision. The supplied local PDF, especially equations 10-28, is the architecture reference.
- Mülken and Blumen. [Continuous-Time Quantum Walks](https://arxiv.org/abs/1101.2572), Physics Reports, 2011. Graph-generated coherent transport background.
- Mernyei, Meichanetzidis and Ceylan. [Equivariant Quantum Graph Circuits](https://proceedings.mlr.press/v162/mernyei22a.html), ICML 2022. Symmetry-aware quantum graph architectures.
- Liu et al. [GraphQWalk authors' repository](https://github.com/jitwooo/GraphQWalk), IEEE TNSE 13, 4779-4796, 2026. Transition-probability and spectral structural embeddings; not a pure-state fidelity claim. The draft's DOI was not independently publisher-verified.
- Schuld, Sweke and Meyer. [Encoding and expressive power](https://arxiv.org/abs/2008.08605), PRA 2021. Encoding determines accessible model functions.
- Pérez-Salinas et al. [Data re-uploading](https://arxiv.org/abs/1907.02085), Quantum 2020. Repeated input-dependent quantum operations.
- Levy and Goldberg. [Implicit matrix factorization](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html), NeurIPS 2014. Distributional-statistics motivation for PMI controls.
- Plesch and Brukner. [Quantum-state preparation](https://arxiv.org/abs/1003.5760), PRA 2011. The cost of loading arbitrary states.
- Berry, Childs and Kothari. [Sparse Hamiltonian simulation](https://arxiv.org/abs/1501.01715), FOCS 2015. Complexity under specified Hamiltonian access.
- [Chiral Quantum Walks](https://arxiv.org/abs/1405.6209), 2014. An untested alternative directionality mechanism; no result here is attributed to it.
- Hill, Reichart and Korhonen. [SimLex-999](https://fh295.github.io/simlex.html), Computational Linguistics, 2015. External similarity evaluation, restricted here to 36 covered pairs.
"""
    (ROOT / "REPORT.md").write_text(text)
    return len(text.split())


if __name__ == "__main__":
    print(f"Wrote report: {build()} words")
