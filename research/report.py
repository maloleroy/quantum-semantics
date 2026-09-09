"""Render the completed fixed pilot into reports, a figure, and the research log."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("artifacts/pilot")
REPORTS = Path("reports")


def main():
    rows = json.loads((ROOT / "results.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert len(rows) == protocol["runs"] == 57
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)
    assert all([r["seed"] for r in group] == [11, 23, 37] for group in grouped.values())
    audit = json.loads(Path("artifacts/audit/result.json").read_text())
    controls = json.loads((ROOT / "classical.json").read_text())
    geometry = json.loads((ROOT / "code_geometry.json").read_text())
    states = json.loads((ROOT / "state_geometry.json").read_text())
    REPORTS.mkdir(exist_ok=True)

    def values(name, key="bce", partition="test"):
        return np.array([r[partition][key] for r in grouped[name]])

    def stat(name, key="bce"):
        a = values(name, key)
        return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"

    def table(names):
        lines = [
            "| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in names:
            initial = np.mean([r["initial"]["test"]["bce"] for r in grouped[name]])
            lines.append(
                f"| {name} | {initial:.4f} | {stat(name)} | "
                f"{values(name, 'bit_accuracy').mean():.1%} | "
                f"{values(name, 'exact_word_accuracy').mean():.1%} | "
                f"{values(name, 'valid_code_word_accuracy').mean():.1%} |"
            )
        return "\n".join(lines)

    common = """
## Protocol and reading the numbers

177 train occurrences from 24 complete sentence-text groups; 57 test occurrences
from 8 disjoint groups. Full 729-word alphabetical vocabulary (transductive dictionary),
window 8, 10 qubits, 2 ansatz layers, 20 epochs, Adam-SPSA, learning rate 0.003,
L2 0.001, batch 32. Paired initialization/optimizer seeds 11, 23, 37. Settings were
fixed before outcomes; every run is retained. No test-selected checkpoints or tuning.

SD describes variability across only three runs on ONE split, not a confidence
interval or evidence of population significance. There are 99 distinct training targets
and 37 test targets; 26/57 test occurrences have targets absent from the selected
training subset. No identical context tuple crosses the split. Results are a sensitivity
pilot, not converged language performance or a replication of the paper's corpus.

BCE is mean binary cross-entropy (lower is better). Exact words require every bit
correct. Valid-code word decoding ranks the allowed codebook using a product of
marginals; it is not joint Born-probability decoding. The canonical train-only bit prior
has test BCE **0.6758**. All quantum conditions here have 0% threshold exact-word accuracy.

Sources and reproduction: [protocol](../artifacts/pilot/protocol.json),
[all results](../artifacts/pilot/results.json), [runner](../research/run_suite.py),
[full progress log](../PROGRESS.md). Each run folder stores weights, codebook, raw
probabilities, initial weights, complete history, and train/test indices.
"""

    def write(name, text):
        (REPORTS / name).write_text(text.strip() + "\n")

    audit_table = "\n".join(
        [
            "| Model | Test BCE | Bit accuracy | Exact words | Paper half-bits metric |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {name} | {s['bce']:.4f} | {s['bit_accuracy']:.1%} | {s['exact_word_accuracy']:.1%} | {s['paper_similarity_accuracy']:.1%} |"
                for name, s in [
                    ("QCSE initial", audit["initial"]["test"]),
                    ("QCSE epoch 50", audit["final"]["test"]),
                    *[(name, c["test"]) for name, c in audit["controls"].items()],
                ]
            ],
        ],
    )
    write(
        "01-paper-audit.md",
        f"""
# Experiment 1 — paper fidelity and honest baselines

**Verdict:** the core implementation is a defensible equation-level implementation
of QCSE. The audit found no core formula or gate bug requiring a circuit rewrite.
It is not an exact reconstruction of unspecified author code and does not reproduce
the paper's performance. Documentation overstated the isolation of full-corpus
frequency-ranked IDs; that wording and the broken PDF path were corrected.

## Paper-to-code audit

Read the supplied [paper4.pdf](../../paper4.pdf), 15 pages, arXiv:2509.05729v2,
10 March 2026. PDF SHA256 `{audit["paper_sha256"]}`.

| Paper item | Implementation and verdict |
|---|---|
| Eq. 24, p. 7 | Exponential positional decay, sine/cosine word angles and additive identity term match. |
| Eqs. 25–28, pp. 13–15 appendix | Alternative context equations have direct formula tests; square padding for the angular vector is a documented convention. |
| Reshape, p. 4 | Row-major sequential chunks, trailing zeros; explicit convention where paper is incomplete. |
| Eqs. 10–14, pp. 4–5 | One initial H per qubit; RX then RZ then ascending adjacent CNOT cascade for every upload. Independently checked with dense matrices. |
| Eq. 13 | Display swaps the two angle indices relative to Eqs. 11–12. Implementation follows the preceding gate definitions. |
| Eqs. 15–18, p. 6 | RX/RZ then adjacent CRZ per ansatz layer; independent CRZ matrix test. |
| Figures 2–3 vs equations | Controls point the other way in figures; reference supports both, default follows equations. |
| Parameter count | Explicit ansatz has M(3m−1), so 58 at m=10, M=2. Tables use 3m; no invented ring gate added. |
| Gate count, Eqs. 21–23 | Formula omits the initial m Hadamards; actual circuit includes them. |
| Eqs. 19–20 | Exact Z marginals (q0 first), not softmax probabilities or a complete quantum state. |
| Section V.A, p. 7 | BCE and L2, rate 0.0003, 50 epochs match the miniature audit. SPSA/Adam details, initialization and batch size are implementation choices. |
| Data/split | Local 1,000-phrase corpus differs from paper's 110 English sentences/80 words. Sentence-text grouping is stricter than the unspecified paper split. |

The complete original test suite passed before changes. Expanded tests cover full
statevectors, canonical equivalence, ordering conservation, inactive gates, projection
isolation and the shared-unitary invariant. The pilot's largest independent Qiskit
marginal discrepancy was {max(r["test_oracle_max_error"] for r in rows):.2e}.
The batch evaluator applies actual Qiskit gate matrices to complete complex states;
its speed comes from batching, not an approximation or learned shortcut.

## Complete miniature training

Used original frequency-ranked vocabulary, window 4, two ansatz layers, seed 11,
177 train / 57 test occurrences selected by whole sentence groups. Completed all
50 epochs at the paper's learning rate and penalty using the original Qiskit evaluator.
Wall time: {audit["seconds"]:.2f} s, including diagnostic measurements.

{audit_table}

The constant prior is estimated from training labels only, with add-one smoothing.
Its **96.5% paper metric** is not evidence of language understanding. For 10 independent
fair predicted bits, the expected half-bits success is 62.3047%, while exact matching
is only 0.09766%. Frequency-ranked IDs add substantial bit imbalance on this corpus.
This exposes why the paper metric cannot substantiate a quantum advantage.

## Structural limits checked numerically

- The first encoding RX on each |+> changes global phase only: half the first
  upload's assigned values are unobservable. Window 4 has one upload at 10 qubits.
- Final RZ and CRZ gates cannot affect Z marginals: at least 19/58 parameters are
  inactive for the data term in this two-layer reference circuit. L2 can still move them.
- A finite-difference probability Jacobian for one audit context had 31/58 nonzero
  columns; this single-context count is not a global parameter-rank estimate.
- Reversing the execution order of an uninterrupted CRZ block, retaining each edge's
  angle, changed the state by only {audit["commuting_crz_max_state_error"]:.2e}: those diagonal gates commute.
- A shared ansatz unitary preserves all pairwise full-state fidelities. This limits
  what “learning similar quantum states” can mean; see [experiment 5](05-semantic-bits.md).

**Conclusion:** suitable as a transparent research reference, with the documented
conventions. Predictive performance remains weak in this audit. Do not present weights
moving, falling training BCE, or the half-bits metric as proof of semantic learning.

Artifacts: [audit result](../artifacts/audit/result.json),
[50-epoch history](../artifacts/audit/history.json),
[raw predictions](../artifacts/audit/predictions.npz).
""",
    )

    write(
        "02-word-index-order.md",
        f"""
# Experiment 2 — repeatedly change numerical word indices

**Question:** does the arbitrary numerical dictionary affect QCSE even when target
bitstrings and all training settings stay fixed?

Apply three independently seeded bijections (101, 202, 303) to INPUT word IDs before
constructing C. Each mapping is frozen across every context, epoch and train/test use.
This changes numerical angles, not sentence order. Keep output bits alphabetical.
A separate `joint-id-map` applies map 101 to targets too: it changes the prediction
code and must not be treated as a pure feature-order comparison.

{table(["canonical", "id-map-101", "id-map-202", "id-map-303", "joint-id-map"])}

All three input maps have lower mean test BCE than canonical in this small pilot.
Their means range from 0.7032 to 0.7160 versus 0.7214; no map was selected or reused
because of its score. None beats the canonical prior at 0.6758, and much of the gap
already exists before training. These results establish sensitivity to arbitrary IDs,
not a useful learned semantic ordering. Joint relabeling needs its own prior, stored
under `joint` in [classical controls](../artifacts/pilot/classical.json).

The original frequency dictionary is audited separately because it encodes full-corpus
frequency information; mixing it with these alphabetical-target comparisons would
confound label-bit imbalance and input geometry.

**Decision:** retain multiple fixed ID maps as a robustness check in future work.
Do not conduct a search for a lucky test mapping.
{common}
""",
    )

    layer_delta = values("layer-pairs") - values("fixed-pairs")
    write(
        "03-between-layers.md",
        f"""
# Experiment 3 — layouts and ID maps between upload layers

**Question:** does changing the layout or dictionary between uploads help beyond
using one fixed change everywhere?

- `fixed-pairs`: one seeded qubit permutation reused on every upload; RX/RZ pairs
  stay together, including padded zero pairs.
- `layer-pairs`: independently seeded pair placement per upload (101+layer).
- `reverse-uploads`: reverse complete upload chunks, preserving every padded value.
- `layer-id-maps`: recompute C with a frozen dictionary seeded 101+layer and take
  that layer's original chunk. Compare directly to `id-map-101`, whose first upload
  is identical. This changes numerical features, not just a permutation of C.

All maps remain fixed during optimization and inference, so caching is valid. These
are between-layer changes, not resampling every batch or epoch.

{table(["canonical", "fixed-pairs", "layer-pairs", "reverse-uploads", "id-map-101", "layer-id-maps"])}

Actual upload counts across 234 examples: 64 have one, 143 have two, 24 have three,
and 3 have four. Thus 170/234 examples genuinely exercise multiple uploads; effects
are diluted by the 64 one-upload cases. No extra uploads or features were invented.

Layer-specific pair placement changes BCE by {layer_delta.mean():+.6f} versus fixed
placement (paired SD {layer_delta.std(ddof=1):.6f}); that difference is negligible here.
Reverse uploads average 0.7128, and layer-specific dictionaries 0.7137. Neither beats
the prior. First-upload RX values are invisible, so reversing chunks also changes
which features survive; temporal-memory claims would require separating this effect.

**Decision:** no reason from this pilot to pursue extensive layer-layout searches.
The fixed-vs-layer-specific control is worth keeping once a model learns reliably.
{common}
""",
    )

    write(
        "04-gate-order.md",
        f"""
# Experiment 4 — rotation and entangler execution order

**Question:** do noncommuting gate-order choices materially change the circuit?

`rz-before-rx` executes RZ then RX in BOTH encoding and ansatz, while retaining each
angle on its original axis. It is a joint architectural change; this pilot does not
isolate encoding order from ansatz order. `reverse-cnot` reverses execution along
the same directed edges (q→q+1); it does not reverse control/target roles or add gates.
Parameter count, inputs and number of operations stay fixed.

{table(["canonical", "rz-before-rx", "reverse-cnot"])}

Reverse CNOT order has test BCE 0.6973 versus 0.7214. But its initial mean is already
0.6990, so most of this difference is an initialization/feature-map effect. It remains
worse than the prior at 0.6758 and yields no exact predictions. Reversing RX/RZ is
more variable and worse on average (0.7308); it exposes first-upload RX information
but that alone does not guarantee useful learning.

A control reversing only CRZ gates within each uninterrupted CRZ block agrees in
full state to {audit["commuting_crz_max_state_error"]:.2e}. Diagonal CRZ gates commute; an apparent large
advantage from their order alone would indicate a changed edge-angle assignment,
other intervening gates, a simulator error, or an uncontrolled experiment.

**Decision:** gate order changes behavior as expected, but no evidence here justifies
calling the better initial BCE an improved semantic model. No test-driven gate search.
{common}
""",
    )

    gt = [
        "| Codebook | Qwen nearest-5 mean Hamming | Neighbors within 3 bits | Random-pair Hamming | Unique codes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, g in geometry.items():
        gt.append(
            f"| {name} | {g['nearest5_mean_hamming']:.3f} | {g['nearest5_within3_fraction']:.1%} | "
            f"{g['random_pair_mean_hamming']:.3f} | {g['unique_codes']} |"
        )
    fidelity_error = max(
        r["max_fidelity_change_under_training_unitary"] for r in states["shared_unitary_geometry"]
    )
    write(
        "05-semantic-bits.md",
        f"""
# Experiment 5 — semantic bit families and actual quantum proximity

**Question clarified by the user:** can related words have nearby codes, potentially
with some bits representing semantic attributes?

A simple balanced recursive partition of normalized Qwen word vectors assigns unique
10-bit codes. At each node, two far-apart words define a projection; split its sorted
scores in half, placing the decision in the next high bit. There is no supervised
attribute label, nearest-neighbor optimization, or search over codebooks.

Compare with alphabetical IDs and a random reassignment of the EXACT same semantic
code set. Codes are external pretrained information. For the predictive experiment,
only center-word TARGET codes change; QCSE context input values remain alphabetical.

## Static lexical geometry

{chr(10).join(gt)}

Semantic codes improve locality over shuffled codes, but **do not beat alphabetical
IDs**. Only 31.9% of the five nearest Qwen neighbors land within three bits, below
alphabetical's 34.9%. Alphabetical ordering can retain word-form/morphology structure.
This is a sanity check against the SAME pretrained space used to build the codebook,
not independent semantic validation. No claim that bit 6 means gender/species/quality
is supported: naming bits requires independent labels and held-out attribute probes.

## Context prediction of semantic bits

{table(["canonical", "semantic-codes", "random-codes"])}

Semantic targets yield mean BCE 0.7235 versus shuffled-code 0.7288, with no exact
word predictions. Bit losses across different codebooks have different priors; consult
the codebook-specific [classical controls](../artifacts/pilot/classical.json).
This does not show that QCSE learned useful semantic families.

## Why Hamming-close is not automatically quantum-close

For distinct computational basis bitstrings, fidelity is zero regardless of whether
one bit or ten bits differ. Further, the shared QCSE ansatz U obeys

`|<Uψ | Uφ>|² = |<ψ | φ>|²`.

On all held-out context pairs, the largest change was {fidelity_error:.2e}.
The correlation between target Qwen cosine and full-state fidelity was about −0.0434,
unchanged across codebooks/weights except tiny tie-rounding effects. Marginal-distance
correlations stayed close to zero (roughly −0.027 to +0.050). Shared pair endpoints
are not independent samples; no pairwise p-values are reported.

A separate genuine nonorthogonal representation is

`tensor_product_q Ry((pi/3) * bit_q) |0>`.

Its fidelity is `(3/4)^Hamming`. This formula was checked against Qiskit circuits.
Mean nearest-neighbor fidelity is 0.3392 for alphabetical codes, 0.3240 for semantic,
and 0.2724 for shuffled codes. This construction achieves actual graded proximity,
but is a separate product-state representation, not trained QCSE or quantum advantage.

**Decision:** the semantic-bit idea needs a better code objective and independent
attribute/semantic evaluation. If the goal is trainable full-state geometry, use a
context-dependent trainable encoder or interleaved data uploads and trainable blocks;
changing only this common terminal unitary cannot achieve it.

Artifacts: [code geometry](../artifacts/pilot/code_geometry.json),
[state geometry](../artifacts/pilot/state_geometry.json),
[diagnostic source](../research/state_geometry.py).
{common}
""",
    )

    local_gain = values("shuffled-qwen-layout") - values("qwen-layout")
    no_gain = values("shuffled-layout-no-entanglement") - values("qwen-layout-no-entanglement")
    interaction = local_gain - no_gain
    write(
        "06-qwen-representations.md",
        f"""
# Experiment 6 — local Qwen geometry, layout-only use and direct representations

**Question:** does pretrained semantic information help through placement alone, or
through direct numerical representations?

## Local model and extraction

Used `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` on the local Mac, CPU float32, four
threads, batches of 16. Last nonpadding token with left padding, L2 normalized,
no prompt. 729 isolated words → 729×1024 vectors, maximum four tokens per word.
These are pretrained word vectors, not occurrence-contextual sentence embeddings.

Download/load took 49.53 s; inference 7.91 s; peak process RSS was 1.88 GiB. No remote
inference API was used. Followed the [official Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
for model size and pooling. [Metadata](../artifacts/qwen/metadata.json) records vector
hashes, immutable revision, device and extraction details.

## Layout-only experiment and topology control

Compute original C first, retaining its original linguistic distances. A greedy
nearest-neighbor path through context-word Qwen vectors reorders its rows AND columns
before serialization. Every numerical C entry is preserved. Pretrained vectors decide
placement only. A shuffled word-to-vector assignment uses the same vector distribution.
The paired no-entanglement controls remove all encoding CNOT and ansatz CRZ gates.
They have 40 circuit parameters versus 58, while retaining 58 optimizer slots for
paired initialization. This is a natural topology ablation, not a capacity-matched one.

{table(["canonical", "qwen-layout", "shuffled-qwen-layout", "qwen-layout-no-entanglement", "shuffled-layout-no-entanglement"])}

Define benefit as shuffled BCE minus semantic BCE (positive favors semantics).
The local-chain benefit is {local_gain.mean():+.4f}; without entanglers it is
{no_gain.mean():+.4f}. The paired difference is {interaction.mean():+.4f} with SD
{interaction.std(ddof=1):.4f}. This tiny, three-seed pilot cannot establish a topology mechanism:
semantic layout is nearly tied with shuffled layout on the chain, and both
no-entanglement versions are much worse. The comparison also changes expressivity
and phase-to-amplitude mixing, not only a notion of semantic locality.

## Direct representations and classical comparison

Mean each context's Qwen word vectors. Fit PCA on TRAIN context means only, retain
20 dimensions, divide by training standard deviations and multiply by 0.5 radians.
The quantum encoder consumes these 20 angles in one upload. A classical logistic
predictor receives exactly the same 20 features (400 full-batch Adam steps, rate 0.05,
L2 0.001, train-only standardization, 210 parameters). Quantum has 58 slots.
Budgets and parameter counts differ and are disclosed; no equal-compute claim.

{table(["canonical", "qwen-direct"])}

The classical projected-Qwen head has test BCE **0.6819**, exact-word accuracy
**14.0%** (8/57), versus quantum BCE **0.7061 ± 0.0205**, exact 0%. The constant
bit prior is still slightly better on BCE (0.6758), while predicting no exact words
under alphabetical IDs. These metrics answer different questions.

Direct Qwen is not a clean replacement isolating just semantics: it averages positions,
compresses features, and has one upload versus up to four for native QCSE. Also the
reference first-upload RX gates discard half of these angles as global phase. The
classical control exposes information potentially lost in this quantum map; no
improvement should be attributed automatically to quantum processing.

**Decision:** no positive evidence for Qwen layout-only benefit here. Direct classical
Qwen features are more promising on exact words, but need a larger held-out sample.
Prioritize a small learnability/feature-retention test over another layout sweep.
{common}
""",
    )

    write(
        "07-classical-and-negative-controls.md",
        """
# Experiment 7 — bit-prior controls and shuffled training labels

These were preregistered controls for the other experiments, not a replacement for
the user's semantic-bit request. Canonical output codes and the same held-out split:

| Classical method | Test BCE | Bit accuracy | Exact words | Trainable parameters |
|---|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {name} | {c['test']['bce']:.4f} | {c['test']['bit_accuracy']:.1%} | "
            f"{c['test']['exact_word_accuracy']:.1%} | {count} |"
            for name, count in [
                ("train_bit_prior", 10),
                ("context_bit_mean", 0),
                ("linear_bit_mean", 110),
                ("linear_qwen_projected", 210),
            ]
            for c in [controls["canonical"][name]]
        )
        + f"""

The prior is add-one-smoothed training target frequency per bit, constant across
contexts. Raw context-bit means are untrained and can be exactly 0 or 1; BCE uses
the reference clipping epsilon 1e-10, so confidently wrong bits receive large loss.
Both logistic heads use 400 deterministic full-batch Adam updates at 0.05 and L2
0.001 on weights only. Means/standard deviations and gradients use training data only.
No classical hyperparameters were selected using test data. Parameter counts are not
matched to the quantum model, and no optimization-budget equivalence is claimed.

{table(["canonical", "shuffled-labels"])}

The shuffled-label experiment permutes center targets only within the training set,
keeping contexts, label frequencies and the entire test set unchanged. All final
scores above are against real labels. Its internal training history intentionally
measures shuffled training labels, clearly flagged in the run configuration.

Shuffled-label test BCE (0.7203) is essentially tied with real-label training (0.7214).
This is strong reason not to interpret the pilot's small BCE differences as learned
contextual semantics. The lower constant-prior loss reinforces that limitation.
{common}
""",
    )

    # Compact scientific figure: initial and final means, all fixed canonical-target conditions.
    panels = [
        ("Word IDs", ["canonical", "id-map-101", "id-map-202", "id-map-303"]),
        ("Between layers", ["fixed-pairs", "layer-pairs", "reverse-uploads", "layer-id-maps"]),
        (
            "Gates and learning control",
            ["canonical", "rz-before-rx", "reverse-cnot", "shuffled-labels"],
        ),
        ("Qwen input", ["qwen-layout", "shuffled-qwen-layout", "qwen-direct"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (title, names) in zip(axes.ravel(), panels):
        y = np.arange(len(names))
        means = [values(n).mean() for n in names]
        sd = [values(n).std(ddof=1) for n in names]
        initial = [np.mean([r["initial"]["test"]["bce"] for r in grouped[n]]) for n in names]
        ax.errorbar(means, y, xerr=sd, fmt="o", capsize=4, label="Final mean ± sample SD")
        ax.scatter(initial, y, marker="x", color="darkorange", label="Initial mean")
        ax.axvline(
            controls["canonical"]["train_bit_prior"]["test"]["bce"],
            ls="--",
            color="gray",
            label="Train-bit prior",
        )
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("Held-out bit BCE (lower is better)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("QCSE pilot: 3 paired seeds, one split, 177 train / 57 test occurrences")
    fig.savefig(REPORTS / "pilot-bce.png", dpi=180)
    fig.savefig(REPORTS / "pilot-bce.pdf")
    plt.close(fig)

    write(
        "00-summary.md",
        """
# QCSE audit and bounded experiments — short report

The core implementation follows the paper's equations with documented ambiguities.
It is honest exact simulation, but this pilot does not demonstrate useful quantum
semantic learning. All 57 planned variant runs and the separate 50-epoch audit finished.

- **Metric warning demonstrated:** a constant training-bit prior scored 96.5% on the
  paper's half-bits accuracy in the frequency-ID audit; QCSE had 0% exact-word accuracy.
- **Index/layer/gate order:** numerical behavior changes, often already at initialization.
  Layer-specific shuffles offer negligible benefit; reverse CNOT order lowers BCE but
  still loses to the constant prior. No test-selected winning configuration was adopted.
- **Semantic bits:** Qwen-derived codes beat shuffled codes on neighborhood Hamming
  distance, but alphabetical IDs do slightly better. No named attribute bits were learned.
- **Quantum-state limitation:** a shared terminal unitary cannot change context-state
  fidelities. Nearby binary basis codes are still orthogonal. A separate Ry product
  encoding correctly provides fidelity `(3/4)^Hamming`, without claiming learned advantage.
- **Local Qwen:** all 729 words embedded with the 0.6B model in ~8 s after loading,
  at ~1.9 GiB peak process memory. Layout-only semantics nearly tie shuffled vectors.
  Direct Qwen + classical head achieves 8/57 exact words; every quantum run gets zero.
- **Learning control:** shuffled training labels perform about as well as real labels.
  A larger sweep is premature; first verify reliable context learning and feature retention.

Limits: one tiny sentence-held-out split, three seeds, 26/57 test targets absent from
selected training data, transductive vocabulary, short fixed optimization budget,
external Qwen knowledge, no independent semantic benchmark, and no hardware/noise run.
These are complete pilots, not a reproduction of the paper's accuracy or a convergence claim.

[1. Paper audit](01-paper-audit.md) · [2. Word indices](02-word-index-order.md) ·
[3. Between layers](03-between-layers.md) · [4. Gate order](04-gate-order.md) ·
[5. Semantic bits](05-semantic-bits.md) · [6. Qwen](06-qwen-representations.md) ·
[7. Controls](07-classical-and-negative-controls.md)

![Initial and final pilot BCE with classical prior](pilot-bce.png)

[Detailed PROGRESS.md](../PROGRESS.md) · [Raw results](../artifacts/pilot/results.json)
""",
    )

    progress = Path("PROGRESS.md")
    text = progress.read_text()
    marker = "\n## Completed experiment ledger (generated from saved results)\n"
    text = text.split(marker)[0]
    ledger = [
        marker,
        (f"All {len(rows)} pilot runs completed; total measured model-run time "
        f"{sum(r['seconds'] for r in rows):.2f} s. Separate original-evaluator audit: "
        f"{audit['seconds']:.2f} s. Projection, embedding and reporting time are not included."),
        ("\nThe sample has 177 training and 57 test occurrences. Of 57 test targets, 26 were\n"
        "unseen in selected training labels. No exact context tuple crosses the split.\n"
        "Full protocol and source commit are stored in artifacts/pilot/protocol.json."),
        "\n### Condition overview\n",
        table(list(grouped)),
        "\n### Audit evidence\n",
        audit_table,
        "\n### Decisions and limits\n",
        ("- No quantum condition beats its canonical constant-prior baseline in the input-only comparisons.\n"
        "- Every quantum run has zero threshold exact-word accuracy; valid-code ranking occasionally\n"
        "  recovers a word and is reported separately, never substituted silently.\n"
        "- Much of the apparent order advantage is already present at initialization.\n"
        "- Shuffled labels match real-label test performance: contextual learning remains unproven.\n"
        "- No new codebook or architecture was selected after outcomes.\n"
        "- Semantic lexical geometry uses the same Qwen space that constructed the codes; independent\n"
        "  attribute tests are still needed before labeling individual bits.\n"
        "- The direct quantum Qwen map discards first RX-channel information; this is an architectural\n"
        "  limitation to test explicitly before an enlarged experiment.\n"
        "- Next useful work would be a small controlled learnability task and a context-dependent\n"
        "  trainable encoder if full-state metric learning is desired. It is not another broad sweep."),
        "\n### Complete per-run training record\n",
        ("The following tables come from saved history files, including epoch zero. Test scores\n"
        "were observed for reporting only. They never selected weights or changed the schedule.\n"
        "For shuffled-label runs, training BCE is against shuffled labels; all other training\n"
        "BCE and every test BCE use that run’s actual target codebook. Final real-label training\n"
        "scores for the shuffle control are stored separately in result.json.\n"),
    ]
    for row in rows:
        name = f"{row['name']}-s{row['seed']}"
        history = json.loads((ROOT / name / "history.json").read_text())
        ledger += [
            f"\n#### {name}\n",
            (f"Runtime {row['seconds']:.2f} s; weights L2 change {row['weights_l2_change']:.6f}; "
            f"Qiskit maximum marginal error {row['test_oracle_max_error']:.2e}. "
            f"[Saved run](artifacts/pilot/{name}/result.json).\n"),
            "| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |",
            "|---:|---:|---:|---:|---:|",
        ]
        ledger += [
            f"| {h['epoch']} | {h['train']['bce']:.6f} | {h['test']['bce']:.6f} | "
            f"{h['test']['bit_accuracy']:.4f} | {h['test']['exact_word_accuracy']:.4f} |"
            for h in history
        ]
    ledger += [
        "\n### Completed report index\n",
        *[f"- [{p.name}](reports/{p.name})" for p in sorted(REPORTS.glob("*.md"))],
        ("\nReports preserve negative findings and separate experimental conventions from the paper.\n"
        "All work remains on research/qcse-ordering; no push or remote write was performed.\n"),
    ]
    progress.write_text(text + "\n".join(ledger))


if __name__ == "__main__":
    main()
