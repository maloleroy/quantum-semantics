# Research direction and experimental contract

## Objective

Find an application where a quantum predictor, or a hybrid with a demonstrably useful
quantum component, improves on a strong classical approach. The intended advantages
are better results at comparable parameter count or faster useful inference under an
explicit device budget. Lower financial cost is currently a secondary priority.

We have demonstrated Hamiltonian learning and a benefit from retaining coherence
relative to specific retrained controls. We have **not** demonstrated an advantage
over strong classical predictors, practical quantum speedup, or hardware trainability.
Quantified evidence is in [experiments/findings.md](experiments/findings.md).

The planning ceiling is fewer than 20 qubits, fewer than 100 million parameters,
about 10 GB available RAM, and a few hours on 24 physical CPU cores. Completed CPU
timings came from an eight-physical-core Ryzen 7 7840S and do not establish a 24-core
estimate or feasibility near 20 qubits.

## Physical prediction and scoring

```text
context -> state preparation -> trainable quantum dynamics
        -> optional quantum readout rotation -> measurement -> word mapping
```

Learned encoders/decoders are optional. State preparation and measurement are not.
For a word basis of size V, the register needs ceil(log2(V)) qubits; a smaller latent
register followed by a classical word decoder is a different, hybrid architecture.
Classical optimization of quantum parameters does not make the prediction path a
classical predictor, but its gradient/shot cost still belongs to training.

For one preparation, evolution and direct word measurement, the primary score is

$$S=\frac1N\sum_i p_\theta(y_i\mid C_i).$$

This is ideal expected one-shot success in the simulator. It is **not** argmax
accuracy or exp(-mean NLL). The latter is a geometric mean; S is an arithmetic mean.
The current code stores correct-word probabilities directly in NPZ metric arrays.

The CLI defaults to validation selection by S and NLL as a training surrogate;
`--loss success` and `--selection nll` are explicit alternatives. Choose and lock the
objective/selection rule before a new held-out comparison. Historical checkpoints
were selected by NLL; reporting their sampling scores does not retroactively select
them for S. Maximizing only S may favor frequent labels, so keep unigram/count controls
and known-word metrics visible.

Classical predictors can cheaply return argmax. A practical comparison must permit
that strategy and account for the quantum shots/selection rule needed to produce an
alternative answer. Sampling-to-sampling scores are informative but do not exhaust
the available classical strategies. Repeated samples do not reveal which word is
correct without a verifier or a stated selection procedure.

The Born rule already defines a distribution. Temperature adjustment, distribution
reconstruction, postselection and retries are additional operations whose costs
must be specified. Rejected/invalid outcomes count as failures per attempt unless
the full retry procedure is included. A predicted UNK bucket does not recover the
original unseen word. Report known-target scores without renormalizing probabilities.
A single observed continuation is a benchmark target, not a complete measure of
language quality or every valid next word.

## Retained components

All project code is directly in `src/` (the Python package is named `src`).

| Module | Responsibility |
|---|---|
| `data.py` | CSV/text loading, normalization, grouped causal partitions, training-only statistics, NPZ datasets |
| `encoding.py` | Fixed recency/position amplitudes and five QCSE context-angle encodings |
| `quantum.py` | Little-endian gates, RX/RZ/CRZ ansatz, readout rotations, Hermitian evolution, graph operators |
| `models.py` | Hamiltonian/dephased, QCSE, same-input MLP, hybrid/classical-hybrid and count controls |
| `metrics.py` | Sampling success, argmax, NLL, known-target scores and paired group intervals |
| `training.py` | Fresh fits, validation selection, locked checkpoints and separate test scoring |
| `storage.py` | Small JSON metadata, CSV histories, portable NPZ arrays/weights |
| `cli.py` | `prepare`, `train`, `evaluate`, `predict`, `plot` |

The pure word-Hamiltonian predictor uses fixed amplitudes

$$\widetilde\psi_j(C)=\sum_{a=0}^{W-1}\sqrt{0.7^a}\,i^a
\mathbf1[w_{t-1-a}=j],\quad
\psi_C=\widetilde\psi/\|\widetilde\psi\|,$$

then $U_\theta=\exp(-iH_\theta)$ and $p(j|C)=|(U_\theta\psi_C)_j|^2$.
Repeated tokens add before normalization. `--amplitude-decay` now exposes the
fixed .7 default for amplitude-based models; a [decay screen](experiments/decay.md)
tests slower forgetting without learned classical preprocessing. All d² real H
coordinates can learn.
The spectral forward and first-order Fréchet adjoint are exact for Hermitian H and
have a finite coincident-eigenvalue limit. Higher-order gradients are unsupported.

`qcse` retains the original numerical-ID context matrices, plus-state preparation,
RX/RZ/CNOT encoding and RX/RZ/CRZ trainable ansatz. These arbitrary-ID encodings are
baselines, not established semantic geometries. `--encoder` permits cross-comparisons.
`--decoder rotated` learns local RX/RY measurement-basis rotations. The `marginal`
decoder reconstructs an independent-bit distribution classically: it is not the
circuit's joint one-shot output and requires separate preparations to sample bits
independently. Use `born` for the direct word-measurement endpoint.

The new hybrid is deliberately small and modular: a learned classical context
encoder prepares a latent state, a trainable H evolves it, and a stochastic classical
decoder maps a measured latent bitstring to a word. The decoder has no direct context
input. Its averaged probabilities can be evaluated exactly in simulation, while its
inference rule needs one latent measurement and a classical conditional draw.
`classical_hybrid` replaces H with a trainable stochastic latent map. `--freeze-core`
permits an independently trained frozen-core diagnostic. These are reusable components,
not newly established hybrid wins or an exact recreation of the retired latent POC.

## Attribution and representation

Fast classical convergence does not prove the quantum component is unused. A
classical component can depend on quantum features, or bypass them regardless of
its convergence speed. Test the hypothesis with independently retrained controls:

- Freeze the quantum core at initialization and retrain the classical modules.
- Replace it with a tuned classical map and retrain; match information and account
  for all free parameters, embeddings, heads, fixed tables and preprocessing.
- Remove classical shortcuts; separately simplify/freeze encoders or decoders to
  understand capacity, without treating forced quantum learning as an overall win.
- Refit dephased and reduced-entanglement variants for their distinct hypotheses.

Removing a block only after training demonstrates sensitivity/co-adaptation, not
that its computation is irreplaceable. Nonzero gradients alone do not establish a win.

Dephasing in the word basis maps $\rho$ to $\mathrm{diag}(\rho)$ before evolution:

$$p_{\rm coherent}(y)=|\sum_jU_{yj}\psi_j|^2,\qquad
p_{\rm dephased}(y)=\sum_j|U_{yj}|^2|\psi_j|^2.$$

It removes cross terms while preserving input populations; it does not set phases
to zero. The dephased architecture is a restricted control, not all classical models.
A coherence benefit is distinct from an entanglement benefit, hard classical
simulation, and overall classical superiority. No entanglement-specific gain has
been isolated here.

A general shared H plus fixed readout learns an orthonormal measurement basis: word
probabilities are constrained quadratic forms of fixed input amplitudes. A unitary
preserves input inner products. The MLP sees the real and imaginary parts of the
same amplitudes and can learn nonlinear features. More input-independent unitary
layers on the same register still give one unitary; unrestricted H already spans
that family. Input-dependent interleaving or auxiliary registers are different model families.
A validation-only [interleaving screen](experiments/interleaving.md) now tests
fixed context RZ gates between learned dense evolutions. An
[auxiliary-register screen](experiments/ancilla.md) now tests joint Hamiltonian
evolution with direct measurement of only the word register. Neither is a
guaranteed improvement. A [segmented-input screen](experiments/segmented.md)
now prepares recent and distant streams in separate sectors of the joint state,
with a retrained sector-dephased control and same-input MLP.
A [low-rank follow-up](experiments/lowrank.md) now uses learned spectral
subspaces to reduce capacity and simulator cost. `--rank` is capped at the
vocabulary size; `--phase-scale` controls fixed context RZ strength. These are
restricted quantum model families, not a claimed hardware speedup.

The [learned context-gate experiment](experiments/conditional.md) adds
`conditional`, `conditional_fixed` and `conditional_mlp`. Small token-angle tables
pool the last word, preceding recent words and older words into local RZ gates
between joint Hamiltonian blocks. `--recent-window` sets the recent/older split;
`--phase-scale` scales the pooled angles. Word amplitudes still use
`--amplitude-decay`, and direct word measurement ignores the auxiliary register.
`conditional_fixed` freezes the angle table; `conditional --freeze-core` instead
freezes both Hamiltonians while training the angles. The classical control has
the same feature family and an approximately matched trainable parameter budget.
Parameter placement alone does not establish that quantum computation is useful;
the independently trained controls are essential.

## Resource and data discipline

Execution preference updated 29 October 2026: provide training commands to the
user and stop; do not launch or monitor training, including jobs estimated under
ten minutes. Inspect completed artifacts when the user returns. This supersedes
the earlier permission to run short training experiments autonomously.

A register has 2^n amplitudes, but a dense general H has 4^n real coordinates.
At 8/13/19 qubits that is 65,536 / 67,108,864 / 274,877,906,944 parameters.
One complex128 matrix alone needs 1 MiB / 1 GiB / 4 TiB respectively, before
optimizer, gradient and intermediate storage. Small qubit count is not a runtime
or memory guarantee. Generic state preparation and dense-H compilation are not free.

A hardware-oriented candidate needs efficient preparation and a structured,
implementable evolution/readout. Count reset, native gates, depth/connectivity,
shots, readout and classical processing. Compare with a competitive classical task
solver, not merely its statevector simulation. Report training and inference costs
separately; exact CPU gradients are not available free on a device.

Data sources are selected explicitly; do not silently concatenate curated sentences
and their raw Tatoeba duplicates. Normalize before group splitting. Vocabulary,
counts and calibration statistics use only training sentences, independently of
the gradient-example limit. Retain every final partial batch. Preserve the split,
seeds, dataset/source fingerprints and selection lock. Store arrays/weights in NPZ,
histories/tables in CSV, and only small metadata in JSON. No optimizer resume is
implemented; loading selected weights is not continuation of an Adam trajectory.

## Open choices

Before substantial new work, decide the task, advantage axis, actual readout/shot
budget, pure versus hybrid scope, structured/input-dependent H family, objective,
validation criterion, classical controls, and fresh confirmation partition.
The existing test has been inspected and its one-shot endpoint was post-hoc.

Potential directions include aligning training with sampling success, structured
Hamiltonians, context-dependent interleaving, or auxiliary quantum systems.
Screens of these families have produced simulator recipe gains and negative
results, but no overall quantum advantage. A new experiment should have
a plausible route to beating the matched MLP and inexpensive count baselines;
another trainability demonstration alone answers an already established question.

Background: [encoding and expressivity](https://arxiv.org/abs/2008.08605),
[state preparation](https://arxiv.org/abs/1003.5760),
[measurement and dephasing](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/formulations-of-measurements).
