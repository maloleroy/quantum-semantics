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
| `tokenization.py` / `subwords.py` | Training-only byte BPE and lossless WikiText preparation |
| `generation.py` | Autoregressive direct token draws and explicit non-text outcomes |
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

The [byte-BPE screen](experiments/byte-bpe.md) introduces a separately scored
open-vocabulary task. `prepare --tokenizer byte-bpe --vocabulary 512` learns
merge rules on training articles only, retaining all UTF-8 bytes and raw case,
punctuation and numbers. NPZ datasets carry the tokenizer and target byte lengths.
`generate` performs rolling direct draws with no rejection. Word datasets retain
their original default preprocessing and UNK policy. Token-level scores must not
be compared directly with earlier word-level scores; sampled-target bits per byte
has an explicit denominator and is not a full-corpus evaluation. The prepared
screen has completed with zero UNK targets but weak quantum prediction and
generation; coverage alone does not establish prediction quality.

The [product-state comparison](experiments/product.md) adds `product`,
`product_fixed` and `product_mlp`. Learned local RY angles prepare all register
qubits before Hamiltonian evolution; separate pooled RZ angles interleave the
blocks. There is no classical vocabulary decoder. Fixed-controller, frozen-core
and matched classical controls distinguish the relevant attribution hypotheses.
This reduces initial-state preparation to local rotations, but does not establish
efficient Hamiltonian compilation or useful quantum computation. The first comparison improved token NLL to 3.1975 and sampling to 21.79%.
The [expanded training comparison](experiments/product-exposure.md) further reaches
2.9673 / 23.17%, beating an equal-update small-data control and bigram NLL.
The matched MLP improves more, to 2.7966 / 30.84%. Longer context has almost no
benefit and generation remains fragmented. Token-order representation is the next
hypothesis; weighted pooled angles already retain some position information, so
order-sensitive alternatives need a controlled comparison.

The [sequential preparation comparison](experiments/sequence.md) is implemented
and has completed with negative results. `sequence` applies token-ordered local RY/RZ
rotations before the joint Hamiltonians; `sequence_pooled` aggregates the same
angles before rotation, and fixed/frozen controls test learning contributions.
`sequence_mlp` uses a compact, phase-invariant Bloch description of the same
prepared state with an approximately matched parameter budget. Preparation stays
separable and classically simulable. All 58 tests pass, but sequential NLL/sampling regress to 3.1173 / 21.32%,
with no advantage over summed angles and a 3.44-hour fit. The 45–90 minute runtime
estimate was too optimistic. Keep the faster product model as reference.
[Fixed-checkpoint diagnostics](experiments/product-diagnostics.md) show a substantial
non-dominant-target sampling deficit and motivated a [rank/capacity screen](experiments/product-capacity.md), now complete.
Rank 128 improves product NLL to 2.9281 but sampling remains 23.20%, behind the
matched MLP's 31.04%. More rank alone has not closed the deficit;
neither useful long-context generation nor quantum advantage has been established.

The [local measurement-basis screen](experiments/product-readout.md) has completed
without a useful gain. `product` and `product_fixed` now accept `--decoder rotated`,
adding 18 RX/RY angles on the nine token qubits before joint measurement. Auxiliary
rotations are omitted. Frozen-Hamiltonian and fixed-controller controls accompany
ranks 64/128, with audited unrotated/MLP references. All 60 tests pass, but rank-128 NLL is essentially unchanged and sampling
slightly worse, with a 96.2-minute fit. Keep unrotated product as reference. The readout is a physical quantum rotation, not a classical
probability decoder or sampling-temperature adjustment.

The [direct-success diagnostic](experiments/product-success.md) completed using
the existing unrotated product, matched MLP and frozen-Hamiltonian models. It
changes loss and validation selection to S while retaining NLL and generation
quality as separate assessments. A train-modal space predictor already yields
30.05% validation S without context, demonstrating that S improvement alone cannot
satisfy the text-generation goal. The completed product reaches 34.36% S but NLL 8.0966 and repetitive outputs;
the success-trained MLP collapses to spaces. The NLL-trained MLP's available argmax
40.58% still exceeds quantum one-shot success. Future generation experiments
return to declared NLL training/selection while retaining S, argmax, modal-target
buckets and samples. A higher S alone is not a successful language model.

The [context-block streaming screen](experiments/stream.md) completed with negative
results. Four chronological blocks interleaved with learned Hamiltonians reach
3.2984 NLL, worse than full-context pooling (2.9734), short context (3.1750) and
the matched MLP (3.1300). All predictions and reports were independently audited.
Text remains fragmented. Retain the older product reference; do not scale the
streaming recipe. Broader training coverage with equal-update controls is the next
bounded question, without assuming it will close the classical gap. The manual
[coverage comparison](experiments/product-coverage.md) completed with 524,288 unique
positions versus a fourfold repeated old set, matching updates and selection.
Expanded quantum NLL improves to 2.7870 from repeated-set 2.9077, but the equally
exposed MLP reaches 2.4941. Sampling is 24.42% versus 36.66%, and text remains
fragmented. Retain these expanded-data references; additional data has not closed
the classical gap or established useful long-context generation.

The [adaptive-spectrum screen](experiments/adaptive-spectrum.md) completed with
a useful within-quantum gain: NLL 2.7007 versus matched static capacity 2.7558,
with sampling 25.77% versus 24.62%. The matched MLP remains ahead at 2.4393 NLL
and 38.44% sampling, and generated text remains fragmented. A shared linear context table shifts eigenvalues in
each learned Hamiltonian basis. Preparation and direct measurement remain quantum;
no classical vocabulary head is added. Parameter-matched static capacity, same-input
MLP and fixed-basis/base-energy controls accompany the candidate. In this family,
`--freeze-core` leaves spectral shifts trainable, so the resulting Hamiltonian is
not frozen. All locked predictions and reports were independently reproduced.
Keep the adaptive model as the quantum reference; test a declared learning-rate
decay and short-context control before attributing further gains to representation.

The [same-backend cosine comparison](experiments/dgx-cosine.md#completed-results)
completed on the DGX and was audited. Decay from .003 to .0003 over 30 epochs
barely changes adaptive quantum NLL (2.7021 constant, 2.7013 cosine), while the
matched MLP improves to 2.4077. Full 128-token and retrained 16-token adaptive
models are indistinguishable in NLL within the descriptive paired intervals.
Non-space sampling remains only 6.08% versus the MLP's 19.30%; generated text is
fragmented and includes invalid/control characters. Retain constant-rate adaptive
as the quantum reference and cosine MLP as the stronger classical reference.
Changing this schedule has not solved the quantum representation deficit.
All seven locks, selected epochs, metric tables and saved sample renderings were
checked; the remote summary independently reloaded and scored weights on CUDA.
No held-out test was scored. The next candidate should improve conditional
quantum distributions while preserving direct measurement and a meaningful
same-input classical control, rather than merely increase the modal-space score.

The [count-initialized screen](experiments/prior-screen.md) tests that hypothesis
with `prior`: fixed training-bigram amplitudes, learned context-dependent
Hamiltonians and direct word measurement. Zero initial energies reproduce the
count baseline exactly. There is no learned classical encoder/decoder in the
quantum candidate. Unigram-input, fixed-basis, short-context and matched classical
refits test where any gain comes from. Fixed count storage and potentially costly
amplitude preparation are explicitly counted; this is not a hardware speedup claim.
The completed and audited screen improves quantum NLL to **2.6676** from 2.7021,
meeting its declared NLL retention rule. Sampling falls from 25.77% to 25.38%,
while non-space sampling rises from 6.08% to 6.50%; retain both tradeoffs.
Fixed-basis NLL 3.0198 shows that the learned bases matter in this family, but
same-input cosine MLP 2.4726 and the stronger archived MLP 2.4077 remain ahead.
The 16-token refit reaches 2.6730, so useful older context is still unproven.
All three seed-11 quantum diagnostic samples avoid invalid/control characters but remain
fragmented. Retain `prior` as the provisional quantum NLL reference and the older
cosine MLP as the strongest classical reference. The seed follow-up below tests
repeatability; independent-partition confirmation remains needed. No held-out test was scored.

The [completed seed-replication study](experiments/prior-seeds.md#completed-results)
confirms the NLL gain at both new seeds 22/33 and on the three-seed mean:
**2.6678 prior versus 2.7015 adaptive**. It passes the predeclared replication
rule. Mean sampling remains lower (25.51% versus 25.74%); mean argmax is slightly
higher (40.76% versus 40.29%). The stronger MLP remains ahead at 2.4121 NLL and
39.41% sampling. Non-space sampling is 6.54% prior versus 6.07% adaptive and
19.66% MLP. One of the six new prior samples has an invalid UTF-8 sequence; all
remain fragmented. Retain prior as the quantum NLL reference without claiming
universal metric dominance or useful text generation. All 60 official validation
articles already appear in selection data, so this tests seed robustness, not
independent article-level generalization. The held-out test remains unscored.

The [learned-position comparison](experiments/prior-position.md#completed-results)
improves quantum NLL to **2.5951** and sampling to **26.66%**, beating both the
old prior and nearly parameter-matched fixed pooling (2.6685 NLL). Learned lag
weights preserve the state preparation, Hamiltonians and Born output; they are
classical preprocessing, not quantum attention. Fixed bases regress to 2.9962,
while same-feature MLP reaches 2.4213 and the stronger archived MLP remains at
2.4077. The refitted 16-token model reaches 2.5868; longer context still has no
established benefit. Selected spectral weights concentrate on lags 2–4, a post-hoc
diagnostic consistent with improved local resolution, not causal attribution.
Quantum text remains fragmented with an invalid UTF-8 sequence. These initial
single-seed results motivated the replication below; independent confirmation
remains untested, and no held-out test was scored.

The [position seed study](experiments/position-seeds.md#completed-results) replicates
the gain against both prior and matched-capacity pooling at seeds 22/33 and on the
three-seed mean: **2.60325 NLL / 26.57% sampling** full position, versus
**2.66776 / 25.51%** prior and **2.66729 / 25.64%** capacity. The short refit
averages **2.59486**, and the longer-context rule fails; the strong MLP remains
better at **2.41207 / 39.41%**. All six new position samples avoid invalid/control
characters but remain fragmented, and the older invalid seed-11 sample is retained.
Keep learned positions as a seed-replicated quantum reference on reused validation,
without claiming coherent text, useful older context or overall quantum advantage.
No new partition or held-out test scoring was introduced.

The [three-stage depth comparison](experiments/depth-screen.md) and
[seed replication](experiments/depth-seeds.md#completed-results-10-november-2026)
now improve mean quantum NLL to **2.53554** from two-stage content **2.57588** and
matched-width **2.56465**, with **28.02%** sampling. Most extra parameters are in
Hamiltonians; the extra RZ controller remains explicitly counted. The short refit
is still better (**2.51807**) and the strongest MLP leads (**2.41207 / 39.41%**).
Selected full models fit training data better than short models but generalize
worse on reused validation. Preserve the demonstrated depth/interleaving gain
while investigating that generalization gap; it is not independent confirmation,
coherent text or quantum advantage. Invalid UTF-8 samples remain in the evidence.

The [older-context masking screen](experiments/context-dropout.md#completed-results--audited-10-november-2026)
completed with a weak single-seed signal. Training with the older branch hidden for
half of examples reaches **2.52941 NLL / 28.05% sampling**, versus unmasked full
**2.54015**, but the paired NLL interval includes zero. The .25 rate also lacks a
clear gain. Neither beats the independently refitted short model (**2.52258**),
and strong MLP remains ahead (**2.40775 / 39.18%**). Generation stays incoherent;
one .25 sample contains invalid UTF-8. Retain the replicated unmasked recipe as the
established reference, with .50 exploratory only. Default training remains
unmasked. Masking has not established useful older context or solved the classical
gap; further work needs a stronger route to context utility and coherent prediction.

The [broader-coverage comparison](experiments/depth-coverage.md#completed-results--audited-10-november-2026)
completed with a positive single-seed NLL result at unchanged 30,720 updates.
The larger 2,097,152-position pool improves full quantum NLL from **2.54015 to
2.46253** and beats the equally exposed short refit (**2.47673**), with both
prespecified interval/non-space criteria passing. Sampling remains slightly lower
for full than short; seed replication and independent confirmation are still needed.
The equally exposed MLP improves more, reaching **2.22207 NLL / 39.42% sampling**,
so the quantum/classical NLL gap widens. Quantum generation remains incoherent,
including an invalid UTF-8 sample. Retain expanded coverage as a promising quantum
candidate and the new MLP as the stronger reference, without claiming useful
long-form generation or quantum advantage.

## Resource and data discipline

The [content-controller investigation](experiments/content-controller.md) adds a
small classical query/key selector while retaining Hamiltonian refinement and Born
output. The [completed seed replication](experiments/content-seeds.md#completed-results-9-november-2026)
confirms improvement over position references: mean NLL **2.57588** versus
**2.60325**, with sampling **27.29% versus 26.57%**. The short refit remains better
at **2.56249**, and strongest MLP at **2.41207**. This remains reused-validation
seed evidence, not independent confirmation or coherent generation.

[Locked-context interventions](experiments/content-diagnostics.md#completed-results)
show that removing, reversing or replacing older tokens worsens full-model NLL.
The older branch is used, but sensitivity under corruption does not overturn the
short refit's advantage. Full-context generalization and quantum expressivity
remain separate hypotheses requiring controlled experiments.

Execution preference updated 3 November 2026: the new research goal authorizes
experiments estimated at ten minutes or less on the laptop and longer experiments
on the DGX. Stop if DGX connectivity fails. This replaces the earlier manual-run
handoff preference. Intensive DGX work must use verified GPU tensors, forward
passes and gradients, not merely a GPU allocation; CPU work is limited to setup,
data handling, light tests and explicitly requested bounded comparison benchmarks.
Use the CUDA dependency extra and explicit CUDA devices in custom code. Long runs
are started once and the goal is then paused, without model polling, until the run
finishes and the user resumes the goal again. As requested on 9 November 2026,
use the laptop `dgx start` notifier after starting a Slurm job; do not use
`codex-long-jobs` for future work. `dgx` is an interactive Bash function backed by
`~/.local/lib/dgx-watchdog.sh`; `bash -ic 'dgx start'` loads it in a
non-interactive command environment. Queue-empty notifications require subsequent
Slurm accounting and artifact verification before claiming success.

Resource preference updated 5 November 2026: default to `prod10` with one 10 GB
A100 MIG slice for future submissions. Reserve `prod80` and a full 80 GB A100 for
measured memory needs or a documented compute/time requirement. Check memory and
throughput with the declared workload on `prod10` before escalating; preserve the
scientific batch/update budget. See the [DGX guide](dgx.md) for submission overrides.

CUDA execution and simulator throughput are verified in the [DGX guide](dgx.md).
The [DGX cosine study](experiments/dgx-cosine.md) carries the pending optimization
comparison forward with fresh constant-rate CUDA controls, preserving the original
batch/update budget rather than confounding the rate comparison with backend changes.

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


The [content-dependent predictive comparison](experiments/content-screen.md) and
its [seed replication](experiments/content-seeds.md) are completed and audited.
The default 10 GB MIG slice suffices for these configurations. Future Slurm work
uses `dgx start` on the laptop after submission, as described above.
