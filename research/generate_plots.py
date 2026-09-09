"""Generate publication-grade plots for all QCSE experimental results.

Generates:
1. 01_audit_and_metric_flaws.png / .pdf
2. 02_all_conditions_bce_and_initialization.png / .pdf
3. 03_training_dynamics_epochs.png / .pdf
4. 04_semantic_codes_and_quantum_geometry.png / .pdf
5. 05_classical_vs_quantum_benchmark.png / .pdf
6. 06_executive_dashboard.png / .pdf
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure custom matplotlib config doesn't error
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0f172a",
    "axes.labelcolor": "#1e293b",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

ROOT = Path("artifacts/pilot")
AUDIT_PATH = Path("artifacts/audit/result.json")
PLOTS_DIR = Path("reports/plots")
ARTIFACTS_DIR = Path("/Users/ethan/.gemini/antigravity-ide/brain/1d66f224-c9a7-4c4e-8ab1-8944b22747b6")


def load_data():
    results = json.loads((ROOT / "results.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    classical = json.loads((ROOT / "classical.json").read_text())
    code_geom = json.loads((ROOT / "code_geometry.json").read_text())
    state_geom = json.loads((ROOT / "state_geometry.json").read_text())
    audit = json.loads(AUDIT_PATH.read_text())

    grouped = defaultdict(list)
    for r in results:
        grouped[r["name"]].append(r)

    # Load histories
    histories = {}
    for r in results:
        name = f"{r['name']}-s{r['seed']}"
        hpath = ROOT / name / "history.json"
        if hpath.exists():
            histories[name] = json.loads(hpath.read_text())

    return {
        "results": results,
        "protocol": protocol,
        "classical": classical,
        "code_geom": code_geom,
        "state_geom": state_geom,
        "audit": audit,
        "grouped": grouped,
        "histories": histories,
    }


def save_fig(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / f"{name}.png"
    pdf_path = PLOTS_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path} and {pdf_path}")

    # Copy to brain artifact dir if it exists
    if ARTIFACTS_DIR.exists():
        shutil.copy(png_path, ARTIFACTS_DIR / f"{name}.png")
        shutil.copy(pdf_path, ARTIFACTS_DIR / f"{name}.pdf")
    plt.close(fig)


# ==============================================================================
# Plot 1: Audit & Flawed Metric Comparison
# ==============================================================================
def plot_01_audit(data):
    audit = data["audit"]
    models = [
        ("Fair Bit Null", 0.6931, 0.500, 0.0009765, 0.6230),
        ("QCSE Initial (Ep 0)", audit["initial"]["test"]["bce"], audit["initial"]["test"]["bit_accuracy"], audit["initial"]["test"]["exact_word_accuracy"], audit["initial"]["test"]["paper_similarity_accuracy"]),
        ("QCSE Final (Ep 50)", audit["final"]["test"]["bce"], audit["final"]["test"]["bit_accuracy"], audit["final"]["test"]["exact_word_accuracy"], audit["final"]["test"]["paper_similarity_accuracy"]),
        ("Context Bit Mean", audit["controls"]["context_bit_mean"]["test"]["bce"], audit["controls"]["context_bit_mean"]["test"]["bit_accuracy"], audit["controls"]["context_bit_mean"]["test"]["exact_word_accuracy"], audit["controls"]["context_bit_mean"]["test"]["paper_similarity_accuracy"]),
        ("Linear Bit Mean", audit["controls"]["linear_bit_mean"]["test"]["bce"], audit["controls"]["linear_bit_mean"]["test"]["bit_accuracy"], audit["controls"]["linear_bit_mean"]["test"]["exact_word_accuracy"], audit["controls"]["linear_bit_mean"]["test"]["paper_similarity_accuracy"]),
        ("Train Bit Prior", audit["controls"]["train_bit_prior"]["test"]["bce"], audit["controls"]["train_bit_prior"]["test"]["bit_accuracy"], audit["controls"]["train_bit_prior"]["test"]["exact_word_accuracy"], audit["controls"]["train_bit_prior"]["test"]["paper_similarity_accuracy"]),
    ]

    labels = [m[0] for m in models]
    bce = [m[1] for m in models]
    bit_acc = [m[2] * 100 for m in models]
    exact_acc = [m[3] * 100 for m in models]
    paper_metric = [m[4] * 100 for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Experiment 1: Audit of QCSE Paper Metrics & Non-Contextual Baselines\n(arXiv:2509.05729 Setup on 50 Epochs)", fontsize=14)

    colors = ["#94a3b8", "#3b82f6", "#1d4ed8", "#f59e0b", "#10b981", "#059669"]

    # 1. Paper Half-Bits Similarity Score
    ax = axes[0, 0]
    bars = ax.barh(labels, paper_metric, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.axvline(62.3, color="#64748b", linestyle=":", label="Theoretical Fair-Bit Null (62.3%)")
    ax.set_xlabel("Paper Similarity Metric (%) [≥50% bits match]")
    ax.set_title("Paper's 'Half-Bits-Match' Metric\n(Trivial Prior Achieves 96.5%!)", color="#b91c1c")
    ax.set_xlim(0, 105)
    for b in bars:
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.5)

    # 2. Exact Word Accuracy (Real Language Metric)
    ax = axes[0, 1]
    bars = ax.barh(labels, exact_acc, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.set_xlabel("Exact Word Accuracy (%) [10/10 bits correct]")
    ax.set_title("Real Semantic Metric: Exact Word Accuracy\n(QCSE achieves 0.0%!)", color="#1e3a8a")
    ax.set_xlim(0, 26)
    for b in bars:
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.5)

    # 3. Test BCE
    ax = axes[1, 0]
    bars = ax.barh(labels, bce, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.axvline(np.log(2), color="#64748b", linestyle=":", label="Fair Bit Prior (ln 2 ≈ 0.693)")
    ax.set_xlabel("Held-Out BCE Loss (Lower is Better)")
    ax.set_title("Primary Metric: Held-Out Binary Cross-Entropy", color="#1e3a8a")
    ax.set_xlim(0, 3.0)
    for b in bars:
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2, f"{b.get_width():.3f}", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.5)

    # 4. Bit Accuracy
    ax = axes[1, 1]
    bars = ax.barh(labels, bit_acc, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.axvline(50.0, color="#64748b", linestyle=":", label="Fair Random Guessing (50.0%)")
    ax.set_xlabel("Bit-Level Accuracy (%)")
    ax.set_title("Bit-Level Marginal Accuracy", color="#1e3a8a")
    ax.set_xlim(0, 85)
    for b in bars:
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.5)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "01_audit_and_metric_flaws")


# ==============================================================================
# Plot 2: All 19 Conditions Initial vs Final BCE
# ==============================================================================
def plot_02_all_conditions(data):
    grouped = data["grouped"]
    classical = data["classical"]

    # Order conditions logically
    categories = [
        ("Canonical & ID Bijections", [
            ("canonical", "Canonical (Alphabetical IDs)"),
            ("id-map-101", "Input Permutation (Seed 101)"),
            ("id-map-202", "Input Permutation (Seed 202)"),
            ("id-map-303", "Input Permutation (Seed 303)"),
            ("joint-id-map", "Joint Input & Target Permutation"),
        ]),
        ("Between-Layer Uploads", [
            ("fixed-pairs", "Fixed Pairs Layout"),
            ("layer-pairs", "Layer-Specific Pairs Layout"),
            ("reverse-uploads", "Reverse Upload Order"),
            ("layer-id-maps", "Layer-Specific ID Maps"),
        ]),
        ("Gate Execution Order", [
            ("rz-before-rx", "RZ before RX (Ansatz & Upload)"),
            ("reverse-cnot", "Reverse CNOT Cascade Order"),
        ]),
        ("Semantic vs Random Codes", [
            ("semantic-codes", "Semantic Target Codes (Qwen Bisection)"),
            ("random-codes", "Random Shuffled Semantic Codes"),
        ]),
        ("Qwen Layout & Topology", [
            ("qwen-layout", "Qwen Nearest-Neighbor Layout"),
            ("shuffled-qwen-layout", "Shuffled Vector Layout"),
            ("qwen-layout-no-entanglement", "Qwen Layout (No Entanglement)"),
            ("shuffled-layout-no-entanglement", "Shuffled Layout (No Entanglement)"),
        ]),
        ("Direct Embeddings & Control", [
            ("qwen-direct", "Direct Qwen Angle Map (20 PCA dims)"),
            ("shuffled-labels", "Shuffled Labels (Negative Control)"),
        ]),
    ]

    all_names = []
    display_labels = []
    cat_indices = []
    idx = 0
    for cat_title, conds in categories:
        for cond_key, label in conds:
            all_names.append(cond_key)
            display_labels.append(label)
            idx += 1
        cat_indices.append((cat_title, idx - len(conds), idx - 1))

    fig, ax = plt.subplots(figsize=(14, 11), layout="constrained")
    fig.suptitle("Pilot Study (19 Conditions, 57 Runs): Held-Out Test BCE\nInitial Mean vs. Final Mean ± Sample SD (3 Seeds)", fontsize=14)

    y_pos = np.arange(len(all_names))

    final_means = [np.mean([r["test"]["bce"] for r in grouped[k]]) for k in all_names]
    final_sds = [np.std([r["test"]["bce"] for r in grouped[k]], ddof=1) for k in all_names]
    init_means = [np.mean([r["initial"]["test"]["bce"] for r in grouped[k]]) for k in all_names]

    # Category background alternating bands and category labels on right side
    for i, (cat_title, start_i, end_i) in enumerate(cat_indices):
        if i % 2 == 0:
            ax.axhspan(start_i - 0.45, end_i + 0.45, color="#f8fafc", zorder=0)
        else:
            ax.axhspan(start_i - 0.45, end_i + 0.45, color="#ffffff", zorder=0)
        # Put category title on right margin
        ax.text(1.065, (start_i + end_i) / 2, f"▶ {cat_title}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#334155")

    # Reference baselines
    prior_bce = classical["canonical"]["train_bit_prior"]["test"]["bce"]
    qwen_class_bce = classical["canonical"]["linear_qwen_projected"]["test"]["bce"]
    ax.axvline(prior_bce, color="#059669", linestyle="--", linewidth=1.5, label=f"Train-Bit Prior Baseline ({prior_bce:.4f})", zorder=2)
    ax.axvline(qwen_class_bce, color="#7c3aed", linestyle="-.", linewidth=1.5, label=f"Classical Qwen Logistic Head ({qwen_class_bce:.4f})", zorder=2)
    ax.axvline(np.log(2), color="#94a3b8", linestyle=":", linewidth=1.2, label=f"Uninformed Bit Chance (ln 2 ≈ {np.log(2):.4f})", zorder=2)

    # Plot final means with error bars
    ax.errorbar(final_means, y_pos, xerr=final_sds, fmt="o", color="#2563eb", ecolor="#60a5fa", elinewidth=2, capsize=4, capthick=1.5, markersize=7, label="Final Test BCE (Mean ± SD, Ep 20)", zorder=3)

    # Plot initial means
    ax.scatter(init_means, y_pos, marker="x", color="#ea580c", s=60, linewidth=2, label="Initial Test BCE (Mean, Ep 0)", zorder=4)

    # Annotate key insights
    # Reverse cnot
    cnot_idx = all_names.index("reverse-cnot")
    ax.annotate("Initial mean already 0.6990;\ngate order effect present before training!",
                xy=(init_means[cnot_idx], cnot_idx), xytext=(init_means[cnot_idx] + 0.04, cnot_idx + 0.5),
                arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.2), fontsize=8.5, color="#9a3412", fontweight="bold")

    # Shuffled labels
    shuff_idx = all_names.index("shuffled-labels")
    canon_idx = all_names.index("canonical")
    ax.annotate("Shuffled labels (0.7203) tied with Real labels (0.7214)!\nProves quantum circuit does not learn context semantics.",
                xy=(final_means[shuff_idx], shuff_idx), xytext=(final_means[shuff_idx] + 0.03, shuff_idx + 0.6),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2), fontsize=8.5, color="#b91c1c", fontweight="bold")

    # No entanglement blow-up
    noent_idx = all_names.index("qwen-layout-no-entanglement")
    ax.annotate("Loss blow-up (~1.007) when entanglers removed",
                xy=(final_means[noent_idx], noent_idx), xytext=(final_means[noent_idx] - 0.12, noent_idx + 0.6),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2), fontsize=8.5, color="#b91c1c")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Held-Out Test Binary Cross-Entropy Loss (Lower is Better)")
    ax.set_xlim(0.66, 1.23)
    ax.grid(axis="x", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cbd5e1", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, "02_all_conditions_bce_and_initialization")


# ==============================================================================
# Plot 3: Training Dynamics Across Epochs
# ==============================================================================
def plot_03_dynamics(data):
    histories = data["histories"]
    seeds = [11, 23, 37]

    key_conditions = [
        ("canonical", "Canonical (Alphabetical)", "#2563eb", "-"),
        ("shuffled-labels", "Shuffled Labels (Negative Control)", "#dc2626", "--"),
        ("reverse-cnot", "Reverse CNOT (Best Initial BCE)", "#059669", "-."),
        ("qwen-direct", "Qwen Direct (20 PCA Angles)", "#7c3aed", ":"),
        ("rz-before-rx", "RZ before RX Order", "#ea580c", "-"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Training Dynamics (Epochs 0 to 20): Mean ± 1 SD across Seeds 11, 23, 37", fontsize=14)

    # A: Test BCE Curves
    ax = axes[0, 0]
    epochs = np.arange(21)
    for cond_name, label, color, ls in key_conditions:
        curves = np.array([[h["test"]["bce"] for h in histories[f"{cond_name}-s{s}"]] for s in seeds])
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.12)

    prior_bce = data["classical"]["canonical"]["train_bit_prior"]["test"]["bce"]
    ax.axhline(prior_bce, color="#10b981", linestyle="--", linewidth=1.5, label=f"Train-Bit Prior Baseline ({prior_bce:.4f})")
    ax.set_title("Test BCE across Epochs (Lower is Better)")
    ax.set_ylabel("Held-Out Test BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.67, 0.75)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    # B: Train BCE Curves
    ax = axes[0, 1]
    for cond_name, label, color, ls in key_conditions:
        curves = np.array([[h["train"]["bce"] for h in histories[f"{cond_name}-s{s}"]] for s in seeds])
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.12)

    ax.set_title("Train BCE across Epochs")
    ax.set_ylabel("Train BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.69, 0.74)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    # C: Test Bit Accuracy Curves
    ax = axes[1, 0]
    for cond_name, label, color, ls in key_conditions:
        curves = np.array([[h["test"]["bit_accuracy"] * 100 for h in histories[f"{cond_name}-s{s}"]] for s in seeds])
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.12)

    ax.axhline(50.0, color="#64748b", linestyle=":", label="Chance (50.0%)")
    prior_bit = data["classical"]["canonical"]["train_bit_prior"]["test"]["bit_accuracy"] * 100
    ax.axhline(prior_bit, color="#10b981", linestyle="--", label=f"Train-Bit Prior ({prior_bit:.1f}%)")
    ax.set_title("Test Bit Accuracy (%)")
    ax.set_ylabel("Bit Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylim(46, 60)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")

    # D: Entanglement Ablation Comparison
    ax = axes[1, 1]
    ablation_conds = [
        ("qwen-layout", "Qwen Layout (Entangled)", "#2563eb", "-"),
        ("shuffled-qwen-layout", "Shuffled Layout (Entangled)", "#059669", "--"),
        ("qwen-layout-no-entanglement", "Qwen Layout (NO Entanglement)", "#dc2626", "-"),
        ("shuffled-layout-no-entanglement", "Shuffled Layout (NO Entanglement)", "#ea580c", "--"),
    ]
    for cond_name, label, color, ls in ablation_conds:
        curves = np.array([[h["test"]["bce"] for h in histories[f"{cond_name}-s{s}"]] for s in seeds])
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(epochs, mean, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.1)

    ax.set_title("Entanglement Ablation: Loss Blow-Up Without Entanglers")
    ax.set_ylabel("Held-Out Test BCE")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.70, 1.10)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "03_training_dynamics_epochs")


# ==============================================================================
# Plot 4: Codebook Geometry & Quantum State Fidelity Invariance
# ==============================================================================
def plot_04_geometry(data):
    code_geom = data["code_geom"]
    state_geom = data["state_geom"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), layout="constrained")
    fig.suptitle("Experiment 5: Lexical Codebook Geometry & Quantum State Fidelity Limitations", fontsize=14)

    # 1. Nearest-Neighbor Hamming Distance
    ax = axes[0]
    codebooks = ["alphabetical", "semantic", "shuffled_semantic"]
    labels = ["Alphabetical\nIDs", "Semantic Codes\n(Qwen Bisection)", "Shuffled\nSemantic Codes"]
    mean_hamming = [code_geom[c]["nearest5_mean_hamming"] for c in codebooks]
    within3_frac = [code_geom[c]["nearest5_within3_fraction"] * 100 for c in codebooks]
    random_pair = code_geom["alphabetical"]["random_pair_mean_hamming"]

    colors = ["#2563eb", "#059669", "#dc2626"]
    bars = ax.bar(labels, mean_hamming, color=colors, edgecolor="#1e293b", linewidth=0.8, width=0.55)
    ax.axhline(random_pair, color="#64748b", linestyle="--", label=f"Random Codebook Pairs ({random_pair:.2f})")
    ax.set_ylabel("Mean Hamming Distance to 5 Nearest Qwen Neighbors")
    ax.set_title("Lexical Codebook Locality\n(Alphabetical Beats Unsupervised Semantic!)")
    ax.set_ylim(0, 5.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.3f} bits", ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    # 2. Neighbors Within 3 Bits
    ax = axes[1]
    bars = ax.bar(labels, within3_frac, color=colors, edgecolor="#1e293b", linewidth=0.8, width=0.55)
    ax.set_ylabel("Neighbors within 3 Bits (%)")
    ax.set_title("Proportion of Nearest Neighbors\nwithin 3-Bit Radius")
    ax.set_ylim(0, 45)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)

    # 3. Quantum Fidelity vs Hamming Distance: The Orthogonality Invariant
    ax = axes[2]
    # For computational basis states: fidelity is 0 for any Hamming distance d >= 1
    # For angle-coded product states Ry(pi/3 * bit)|0>: fidelity is (3/4)^d
    d_vals = np.arange(0, 11)
    basis_fidelity = np.where(d_vals == 0, 1.0, 0.0)
    angle_fidelity = (3 / 4) ** d_vals

    ax.plot(d_vals, basis_fidelity, "o-", color="#dc2626", linewidth=2.5, label="Computational Basis QCSE:\nOrthogonal (|⟨ψ|φ⟩|² = 0 for d ≥ 1)")
    ax.plot(d_vals, angle_fidelity, "s--", color="#2563eb", linewidth=2.5, label="Non-Orthogonal Ry Product State:\nGraded Proximity ((3/4)^d)")

    # Mark empirical mean nearest neighbor fidelities
    prod_fids = state_geom["separate_nonorthogonal_product_code_states"]
    ax.scatter([4.24], [prod_fids["alphabetical"]["nearest5_mean_product_state_fidelity"]], color="#1d4ed8", s=80, zorder=5, marker="*", label=f"Alphabetical NN Fidelity ({prod_fids['alphabetical']['nearest5_mean_product_state_fidelity']:.3f})")
    ax.scatter([4.37], [prod_fids["semantic"]["nearest5_mean_product_state_fidelity"]], color="#059669", s=80, zorder=5, marker="^", label=f"Semantic NN Fidelity ({prod_fids['semantic']['nearest5_mean_product_state_fidelity']:.3f})")

    ax.set_xlabel("Hamming Distance (Bits Differing)")
    ax.set_ylabel("Quantum State Fidelity |⟨ψ|φ⟩|²")
    ax.set_title("Quantum Proximity vs. Orthogonality\n(Shared Ansatz Preserves Fidelity: ΔF < 4e-15)")
    ax.set_xticks(d_vals)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "04_semantic_codes_and_quantum_geometry")


# ==============================================================================
# Plot 5: Classical vs Quantum Benchmark (Exact Words & Predictions)
# ==============================================================================
def plot_05_benchmark(data):
    classical = data["classical"]
    grouped = data["grouped"]

    models = [
        ("Classical Qwen (Logistic Head)", 0.6819, 60.2, 14.035, 14.035),
        ("Classical Linear Bit Mean", 0.6957, 59.3, 1.754, 1.754),
        ("Classical Train-Bit Prior", 0.6758, 56.8, 0.0, 0.0),
        ("Quantum Qwen-Direct (20 PCA)", np.mean([r["test"]["bce"] for r in grouped["qwen-direct"]]), np.mean([r["test"]["bit_accuracy"] for r in grouped["qwen-direct"]]) * 100, 0.0, 0.0),
        ("Quantum Reverse-CNOT (Best QCSE)", np.mean([r["test"]["bce"] for r in grouped["reverse-cnot"]]), np.mean([r["test"]["bit_accuracy"] for r in grouped["reverse-cnot"]]) * 100, 0.0, 0.0),
        ("Quantum Canonical QCSE", np.mean([r["test"]["bce"] for r in grouped["canonical"]]), np.mean([r["test"]["bit_accuracy"] for r in grouped["canonical"]]) * 100, 0.0, 0.0),
        ("Quantum ID-Map-202 (Best Valid Code)", np.mean([r["test"]["bce"] for r in grouped["id-map-202"]]), np.mean([r["test"]["bit_accuracy"] for r in grouped["id-map-202"]]) * 100, 0.0, 1.17),
    ]

    labels = [m[0] for m in models]
    bce = [m[1] for m in models]
    bit_acc = [m[2] for m in models]
    exact_words = [m[3] for m in models]
    valid_words = [m[4] for m in models]

    colors = ["#7c3aed", "#10b981", "#059669", "#3b82f6", "#2563eb", "#1d4ed8", "#6366f1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    fig.suptitle("Classical Models vs Quantum QCSE: Head-to-Head Benchmark\n(Exact Words, Valid-Code Words, and Test BCE)", fontsize=14)

    # 1. Exact Word Accuracy (%)
    ax = axes[0]
    bars = ax.barh(labels, exact_words, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.set_xlabel("Exact Word Accuracy (%)")
    ax.set_title("Exact Word Accuracy (%)\n(8/57 for Classical Qwen vs. 0 for Quantum!)", color="#b91c1c")
    ax.set_xlim(0, 18)
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)

    # 2. Valid-Code Word Accuracy (Marginal Likelihood Ranking)
    ax = axes[1]
    bars = ax.barh(labels, valid_words, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.set_xlabel("Valid-Code Word Accuracy (%)")
    ax.set_title("Valid-Code Word Accuracy (%)\n(Ranking Permitted Vocabulary Words)")
    ax.set_xlim(0, 18)
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)

    # 3. Test BCE
    ax = axes[2]
    bars = ax.barh(labels, bce, color=colors, edgecolor="#1e293b", linewidth=0.8)
    ax.set_xlabel("Held-Out Test BCE (Lower is Better)")
    ax.set_title("Binary Cross-Entropy Loss")
    ax.set_xlim(0.65, 0.74)
    ax.axvline(classical["canonical"]["train_bit_prior"]["test"]["bce"], color="#059669", linestyle="--", label="Train-Bit Prior (0.6758)")
    for b in bars:
        ax.text(b.get_width() + 0.001, b.get_y() + b.get_height() / 2, f"{b.get_width():.4f}", va="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.4)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "05_classical_vs_quantum_benchmark")


# ==============================================================================
# Plot 6: Executive Synthesis Dashboard
# ==============================================================================
def plot_06_dashboard(data):
    audit = data["audit"]
    grouped = data["grouped"]
    classical = data["classical"]
    histories = data["histories"]
    seeds = [11, 23, 37]

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    fig.suptitle("Quantum Contextual Semantic Embeddings (QCSE): Empirical Research Synthesis", fontsize=16)

    gs = fig.add_gridspec(2, 3)

    # Subplot A: Paper Similarity Metric vs Real Metrics (Top Left)
    ax_a = fig.add_subplot(gs[0, 0])
    labels_a = ["QCSE\n(Audit)", "Train-Bit\nPrior", "Linear\nBit-Mean", "Fair Bit\nNull"]
    scores_a = [56.1, 96.5, 94.7, 62.3]
    colors_a = ["#2563eb", "#059669", "#10b981", "#94a3b8"]
    bars = ax_a.bar(labels_a, scores_a, color=colors_a, edgecolor="#0f172a", width=0.55)
    ax_a.axhline(62.3, color="#dc2626", linestyle=":", label="Fair Null (62.3%)")
    ax_a.set_ylabel("Paper Half-Bits Score (%)")
    ax_a.set_title("1. Flawed Metric Exposed:\nPrior Gets 96.5% Without Context", color="#b91c1c")
    ax_a.set_ylim(0, 110)
    for b in bars:
        ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_a.legend(loc="lower right", fontsize=8)
    ax_a.grid(axis="y", alpha=0.3)

    # Subplot B: Learning Control: Canonical vs Shuffled Labels (Top Middle)
    ax_b = fig.add_subplot(gs[0, 1])
    epochs = np.arange(21)
    canon_curves = np.array([[h["test"]["bce"] for h in histories[f"canonical-s{s}"]] for s in seeds])
    shuff_curves = np.array([[h["test"]["bce"] for h in histories[f"shuffled-labels-s{s}"]] for s in seeds])

    ax_b.plot(epochs, canon_curves.mean(axis=0), color="#2563eb", linewidth=2.5, label="Canonical (Real Labels)")
    ax_b.fill_between(epochs, canon_curves.mean(axis=0) - canon_curves.std(axis=0, ddof=1),
                      canon_curves.mean(axis=0) + canon_curves.std(axis=0, ddof=1), color="#2563eb", alpha=0.15)

    ax_b.plot(epochs, shuff_curves.mean(axis=0), color="#dc2626", linestyle="--", linewidth=2.5, label="Shuffled Labels (Negative Control)")
    ax_b.fill_between(epochs, shuff_curves.mean(axis=0) - shuff_curves.std(axis=0, ddof=1),
                      shuff_curves.mean(axis=0) + shuff_curves.std(axis=0, ddof=1), color="#dc2626", alpha=0.15)

    ax_b.axhline(classical["canonical"]["train_bit_prior"]["test"]["bce"], color="#059669", linestyle=":", label="Train-Bit Prior (0.6758)")
    ax_b.set_title("2. Negative Control: Shuffled Labels\nTied With Real Training (0.720 vs 0.721)")
    ax_b.set_ylabel("Held-Out Test BCE")
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylim(0.67, 0.74)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8)

    # Subplot C: Exact Words: Classical vs Quantum (Top Right)
    ax_c = fig.add_subplot(gs[0, 2])
    comp_labels = ["Classical\nQwen", "Classical\nLinear", "Quantum\n(All 57 Runs)"]
    comp_words = [14.04, 1.75, 0.0]
    comp_colors = ["#7c3aed", "#10b981", "#2563eb"]
    bars = ax_c.bar(comp_labels, comp_words, color=comp_colors, edgecolor="#0f172a", width=0.5)
    ax_c.set_ylabel("Exact Words Predicted (%)")
    ax_c.set_title("3. Word Recovery Accuracy:\nClassical 14% vs. Quantum 0.0%")
    ax_c.set_ylim(0, 17)
    for b in bars:
        ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.3)

    # Subplot D: Representative Conditions Summary (Bottom Left + Center, 2 cols)
    ax_d = fig.add_subplot(gs[1, 0:2])
    rep_conds = [
        ("canonical", "Canonical (Alphabetical IDs)"),
        ("id-map-202", "Input Word ID Permutation (Seed 202)"),
        ("reverse-uploads", "Reverse Upload Order"),
        ("reverse-cnot", "Reverse CNOT Cascade Execution"),
        ("rz-before-rx", "RZ Before RX Execution"),
        ("semantic-codes", "Semantic Codes (Qwen Bisection)"),
        ("qwen-layout", "Qwen Nearest-Neighbor Layout"),
        ("shuffled-labels", "Shuffled Labels (Negative Control)"),
        ("qwen-direct", "Qwen Direct (20 PCA Angles)"),
        ("qwen-layout-no-entanglement", "No Entanglement Ablation"),
    ]
    rep_keys = [k[0] for k in rep_conds]
    rep_labels = [k[1] for k in rep_conds]
    y_pos = np.arange(len(rep_keys))

    final_means = [np.mean([r["test"]["bce"] for r in grouped[k]]) for k in rep_keys]
    final_sds = [np.std([r["test"]["bce"] for r in grouped[k]], ddof=1) for k in rep_keys]
    init_means = [np.mean([r["initial"]["test"]["bce"] for r in grouped[k]]) for k in rep_keys]

    prior_bce = classical["canonical"]["train_bit_prior"]["test"]["bce"]
    ax_d.axvline(prior_bce, color="#059669", linestyle="--", linewidth=1.5, label=f"Train-Bit Prior Baseline ({prior_bce:.4f})")
    ax_d.axvline(classical["canonical"]["linear_qwen_projected"]["test"]["bce"], color="#7c3aed", linestyle="-.", linewidth=1.5, label="Classical Qwen Head (0.6819)")

    ax_d.errorbar(final_means, y_pos, xerr=final_sds, fmt="o", color="#2563eb", ecolor="#93c5fd", elinewidth=2, capsize=4, markersize=7, label="Final Test BCE (Mean ± SD)")
    ax_d.scatter(init_means, y_pos, marker="x", color="#ea580c", s=50, linewidth=1.8, label="Initial Test BCE (Mean)")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(rep_labels, fontsize=9)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Held-Out Test BCE Loss (Lower is Better)")
    ax_d.set_title("4. Selected Pilot Variants: Test BCE and Initialization Bias")
    ax_d.set_xlim(0.66, 1.05)
    ax_d.legend(loc="lower right", fontsize=8)
    ax_d.grid(axis="x", alpha=0.3)

    # Subplot E: Quantum Fidelity Invariance (Bottom Right)
    ax_e = fig.add_subplot(gs[1, 2])
    dist_vals = np.arange(0, 6)
    basis_f = np.where(dist_vals == 0, 1.0, 0.0)
    angle_f = (3 / 4) ** dist_vals
    ax_e.plot(dist_vals, basis_f, "o-", color="#dc2626", linewidth=2, label="Computational Basis QCSE\n(Orthogonal, F=0)")
    ax_e.plot(dist_vals, angle_f, "s--", color="#2563eb", linewidth=2, label="Ry Angle State ((3/4)^d)")
    ax_e.set_xlabel("Bit Hamming Distance")
    ax_e.set_ylabel("Quantum State Fidelity |⟨ψ|φ⟩|²")
    ax_e.set_title("5. State Fidelity Invariant:\nBasis States Remain Orthogonal", color="#1e3a8a")
    ax_e.set_xticks(dist_vals)
    ax_e.set_ylim(-0.05, 1.05)
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="upper right", fontsize=8)

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "06_executive_dashboard")


def main():
    print("Loading experimental data...")
    data = load_data()
    print("Generating Figure 1: Audit and metric flaws...")
    plot_01_audit(data)
    print("Generating Figure 2: All 19 conditions...")
    plot_02_all_conditions(data)
    print("Generating Figure 3: Training dynamics over epochs...")
    plot_03_dynamics(data)
    print("Generating Figure 4: Codebook and quantum geometry...")
    plot_04_geometry(data)
    print("Generating Figure 5: Classical vs Quantum benchmark...")
    plot_05_benchmark(data)
    print("Generating Figure 6: Executive synthesis dashboard...")
    plot_06_dashboard(data)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
