# Negative, inconclusive and unretained experiments

Companion to the [explainer](../research-summary.md) and [successful steps](successful.md). Evidence
through 15 November 2026. These are rejected recipes or hypotheses, not proofs that an entire model
family cannot work. A small or metric-specific gain is preserved where it occurred. Related pilots,
controls and seed replications are grouped by research question; report reanalyses are not counted
as new training experiments.

NLL is lower-is-better; S means expected one-draw success. Later WikiText results are
validation-only. Historical exact-bit accuracy, cosine retrieval, top-five accuracy, word NLL and
byte-token NLL are different endpoints and must not be pooled.

**Mathematical conventions.** See the [explainer](../research-summary.md) for $p_\theta(y\mid C)$,
the training loss and the direct Born readout, and the [configuration table](successful.md) for
vocabulary sizes, context lengths, qubits and Hamiltonian ranks. In older QCSE experiments,
“layers” counts RX/RZ/controlled-RZ ansatz repetitions; in the Hamiltonian studies, it counts
evolution blocks $e^{-iH_\ell(C)}$. These are different logical depth measures; neither is an
established native hardware gate count. Later byte-BPE quantum screens use ten qubits unless
otherwise stated.

## Historical QCSE, encoder/decoder and attention branches

### 1. Original QCSE production sweep

- **Tried:** context-decay, learning-rate, window, depth and batch configurations; **44 of 45**
  configurations had complete exports.
- **Result:** selected exact-word test accuracy was only **1.068%**, using thresholded agreement of
  all 14 output bits.
- **Decision:** no useful generator. Batch changes also changed update counts; the apparent 64-layer
  winner does not establish a clean depth benefit.
- **Evidence:** historical source H1 below; [retained endpoint
  cautions](findings.md#3-earlier-encoders-readouts-and-graph-models). The incomplete configuration
  supplies no final result.

### 2. Focused local QCSE/attention sweeps

- **Tried:** revised causal controls and earlier small encoding/attention comparisons.
- **Result:** the **16-fit causal study** had zero exact-word accuracy almost everywhere. An earlier
  14-fit snapshot favored learned classical encoding; its attention result used only **133 synthetic
  test examples**.
- **Decision:** useful pipeline/representation evidence, insufficient to select a useful quantum
  next-token model. These studies had different sampling budgets and cannot be combined into one
  benchmark.
- **Evidence:** H2; the later eight 1,024-draw causal runs completed, while eight 4,096-draw runs
  and `full-cv-50` were interrupted and have no final quality conclusion.

### 3. Semantic encoder/decoder prototype and pathway ablations

- **Tried:** learned embeddings, classical encoder/decoder and a small quantum circuit; compare
  trained, frozen, identity and restricted classical pathways.
- **Result:** on the capped phrase task, trained and frozen circuits both give **14.7% cosine
  top-one**; validation CE favors frozen (**4.703 versus 4.754**). Decoder-only training gives
  **2.1%**.
- **Decision:** learned representations matter, but these controls did not establish a benefit from
  training the circuit. No retained quantum-model improvement.
- **Evidence:** H3. These retrieval metrics differ from direct quantum measurement.

### 4. Semantic hyperparameters and source-mixture matrix

- **Tried:** learning rate, initialization scale, layers, batch size, and phrase/cleaned/both capped
  datasets; generally one seed and 25–30 epochs.
- **Result:** larger LR improves some retrieval scores, but extra depth does not reliably help. On
  the combined-source matrix, no circuit gives **17.4% top-one** versus **16.8%** trained circuit.
- **Decision:** no robust quantum contribution. Source-specific caps change the test population, so
  this did not isolate a beneficial mixture on fixed evaluation data.
- **Evidence:** H3. Scripts for Qwen references were prepared; their existence is not a completed
  baseline measurement.

### 5. Quantum attention with fixed versus learned input encoding

- **Tried:** one/two quantum-attention layers, classical versus QCSE encoding, circuit removal, and
  LR/initialization variations on a 5,000-sentence cap.
- **Result:** at the matched original LR, one quantum layer improves test CE **4.6504 → 4.6084**,
  but cosine top-one falls **15.04% → 9.94%** versus no circuit. Learned inputs decisively beat
  fixed QCSE inputs.
- **Decision:** mixed endpoint effects, no demonstrated quantum retrieval advantage; the no-circuit
  arm was not equally retuned throughout. No adoption in the current Hamiltonian model.
- **Evidence:** H4. Simulated overlaps/postselection do not establish efficient finite-shot
  attention.

### 6. Broad robust hybrid circuit screens and depth expansion

- **Tried:** 90 shorter CV fits, then a 156-record screen spanning encoders, qubit counts, depths,
  observables and circuit controls.
- **Result:** hybrid and identity controls are close; exact fixed-word QCSE stays near frequency
  references. Depth eight costs more without a clear gain.
- **Decision:** no stable advantage from adding generic circuit depth or enlarging the quantum
  feature register. The later studies supersede provisional screen rankings.
- **Evidence:** H5. The 156 records include parameter-free controls; these are not 156 independent
  trained architectures.

### 7. Robust follow-up, common-target and gate-family controls

- **Tried:** tuned follow-up (**258 CV fits + 30 refits**) and **40 distinct common-target/control
  fits**, covering causal/bidirectional contexts, identity/local/frozen/CRZ/XX/ZZ circuits and
  classical MLP/BERT references.
- **Result:** final causal MLP NLL/top-one **3.24873 / 35.656%**, versus CRZ **3.46637 / 33.316%**
  and identity **3.47083 / 33.357%**. Small circuit NLL/top-five gains occur, but primary top-one
  gains are inconclusive.
- **Decision:** no quantum advantage. MLP beats the tuned BERT reference too; model capacities
  differ. CRZ/ZZ are equivalent function families here, so their ranking is not an expressivity
  discovery.
- **Evidence:** H5 and H6; [retained historical
  synthesis](findings.md#3-earlier-encoders-readouts-and-graph-models). Matching both training and
  evaluation positions removed earlier context-composition confounds.

### 8. Statistical/bigram-angle QCSE encoding

- **Tried:** replace arbitrary numerical-word encodings with training-statistic-based angles.
- **Result:** exact-word accuracy rises **4.92% → 16.59%**, but the supplied classical bit-prior
  already scores **16.77%**.
- **Decision:** a useful statistical input does not establish an additional quantum contribution.
  This representation was not retained as the quantum improvement.
- **Evidence:** [historical encoding
  findings](findings.md#3-earlier-encoders-readouts-and-graph-models), H7. The older vocabulary was
  transductive, unlike today's train-only protocol.

### 9. Learned local measurement axes in the original QCSE model

- **Tried:** learn the local measurement basis on 1,246 held-out targets.
- **Result:** binary loss improves **.5708 → .5679** and top-five **10.46% → 12.07%**, while
  threshold-word accuracy declines **4.04% → 3.69%**.
- **Decision:** endpoint-dependent effects, no useful exact-word improvement. This does not justify
  the later product-model readout either; that was tested independently.
- **Evidence:** [historical readout
  findings](findings.md#3-earlier-encoders-readouts-and-graph-models), H7.

## Earlier Hamiltonian branches

### 10. Fixed graph/global-time quantum word walk

- **Tried:** build a training-count graph, encode recency counts, and learn global
  walk/context-phase parameters.
- **Result:** word-walk NLL/top-one **4.2964 / 8.72%**, versus classical undirected diffusion
  **4.0350 / 17.26%** and directed propagation **3.4615 / 26.83%**.
- **Decision:** graph structure helps relative to weak QCSE, but quantum propagation adds no
  demonstrated advantage over the classical graph alternatives.
- **Evidence:** [retained graph results](findings.md#3-earlier-encoders-readouts-and-graph-models),
  H8.

### 11. Predictive amplitudes plus a quantum residual walk

- **Tried:** start from a useful count-based predictive distribution, then add quantum-walk
  refinement.
- **Result:** test NLL **2.966171**, versus calibrated classical bigram **2.966157**; the paired
  difference interval includes zero.
- **Decision:** essentially a tie on likelihood; the tiny top-one difference (**31.70% versus
  31.63%**) does not establish useful quantum refinement.
- **Evidence:** [retained predictive-walk
  results](findings.md#3-earlier-encoders-readouts-and-graph-models), H8.

### 12. Fixed Hamiltonian encoder followed by QCSE

- **Tried:** use graph evolution as the encoder before the existing trainable QCSE ansatz and
  marginal-bit decoder.
- **Result:** top-one improves **7.47% → 9.66%**, but NLL worsens **5.1826 → 5.2162**.
  Hamiltonian-only encoding gives **5.07% / 5.3477**.
- **Decision:** a ranking gain rather than a likelihood improvement; no adoption as the best
  generator. Test had already been used, and no independent confirmation followed.
- **Evidence:** [retained encoder
  findings](findings.md#3-earlier-encoders-readouts-and-graph-models), H8.

### 13. Directed source/target Hamiltonian dilation

- **Tried:** double the state space into source and target roles to represent directed transitions.
- **Result:** selection chooses **epoch zero**; validation NLL **2.9328**, essentially
  supplied-bigram performance. Mean target-role success is only **0.000619**.
- **Decision:** conditional word scores hide the very low probability of reaching the required role.
  Nonzero gradients did not translate into useful learned improvement.
- **Evidence:** [retained directional
  findings](findings.md#3-earlier-encoders-readouts-and-graph-models), H8.

### 14. Latent hybrid context correction

- **Tried:** learned classical preparation/controller and vocabulary head around an
  eight-dimensional quantum latent state, with a matched classical replacement.
- **Result:** the initial quantum recipe improves the bigram, but equal-budget tuning reverses its
  apparent lead: three-seed NLL **2.7940 quantum versus 2.7816 classical**; top-one **36.17% versus
  36.03%**.
- **Decision:** context correction works; no additional quantum likelihood benefit survives the
  stronger control. Classical modules also prevent attributing the whole gain to quantum dynamics.
- **Evidence:** [retained hybrid
  findings](findings.md#3-earlier-encoders-readouts-and-graph-models), H8. This is different from
  direct token-register measurement.

## WikiText word and byte-token experiments

### 15. Fixed context-phase interleaving

- **Tried:** fixed, nondecaying context phases between two dense Hamiltonians; later low-rank phase
  variants.
- **Result:** dense interleaving worsens NLL **4.0110 → 4.2596**, S **11.25% → 10.40%**. The later
  low-rank version gives **3.8830 versus 3.8744** static, with no clear NLL gain.
- **Decision:** reject these fixed-feature recipes. Learned/contextual interleaving subsequently
  succeeds, so this is not a rejection of interleaving as a family.
- **Evidence:** [dense screen](interleaving.md) · [low-rank follow-up](lowrank.md).

### 16. Slower forgetting in the fixed amplitude encoder

- **Tried:** decay .9/.97 instead of .7, plus independently retrained four-word controls.
- **Result:** .97 worsens coherent NLL **3.6379 → 3.7215** and MLP **3.3003 → 3.4949**. Dephased
  aggregate improvement is driven by unknown targets, with worse known-word prediction.
- **Decision:** retain .7 for that coherent reference. A dephased model benefits from 32 versus four
  words, but that is not a quantum-coherence improvement.
- **Evidence:** [decay and context controls](decay.md).

### 17. Coherently separate recent and distant input streams

- **Tried:** put recent and older amplitudes in separate auxiliary sectors; compare coherence with a
  sector mixture, first at small scale then 12 fits across three seeds.
- **Result:** expanded-study mean NLL **3.5655 coherent**, **3.5547 previous auxiliary**, **3.4884
  sector mixture**, **3.1641 MLP**. Coherent known-word S is also below the mixture.
- **Decision:** reject the proposed cross-sector interference gain. Removing that interference helps
  NLL, while all quantum variants remain behind the classical control.
- **Evidence:** [pilot](segmented.md) · [larger study](segmented-large.md).

### 18. Calibrate recent/distant sector allocation

- **Tried:** fit one training-only recent-sector probability per seed, with selected quantum weights
  fixed.
- **Result:** weights remain near **.515**; mean NLL changes only **3.488359 → 3.487537**, while S
  falls **12.2600% → 12.2408%**.
- **Decision:** a measurable but negligible likelihood gain, too small to justify another long run
  or explain the MLP gap. Not adopted into the current recipe.
- **Evidence:** [sector calibration](sector-calibration.md).

### 19. First byte-BPE model: fixed-amplitude preparation with learned gates

- **Tried:** carry the word-level amplitude/context-gate model onto the open-vocabulary 128-token
  task, including static/fixed/frozen/short controls.
- **Result:** **4.2773 token NLL / 5.68% S**, versus unigram **4.2945 / 9.30%**, bigram **3.0593 /
  23.16%**, MLP **3.4424 / 24.14%**.
- **Decision:** keep the tokenizer, replace the model preparation. Coverage succeeded, but
  prediction and generation did not; the 16-token refit is also better on NLL (**4.2459**).
- **Evidence:** [byte-BPE screen](byte-bpe.md). No comparison of these token NLLs with older word
  NLLs is valid.

### 20. Sequential token-ordered local preparation

- **Tried:** apply RY/RZ rotations in token order rather than summing angles, with pooled, frozen,
  fixed and matched classical controls: ten qubits, $W=128$, then two rank-64 Hamiltonian stages.
- **Result:** **3.1173 NLL / 21.32% S**, worse than product **2.9673 / 23.17%**, and essentially
  tied with summed angles (**3.1150 / 21.34%**).
- **Decision:** reject this recipe. Longer context worsens NLL, outputs remain fragmented, and a
  learned fit takes **3.44 hours**.
- **Evidence:** [sequential preparation](sequence.md).

### 21. Rotate the product model's measurement basis

- **Tried:** add 18 trainable local RX/RY readout angles at ranks 64/128, with controller/core
  controls.
- **Result:** rank-128 NLL is essentially unchanged, **2.92811 → 2.92786**; S falls **23.20% →
  23.11%**. Rank 64 gives no useful gain either.
- **Decision:** retain unrotated direct measurement. The longest fit takes **96.2 minutes**, with no
  generation improvement.
- **Evidence:** [local readout screen](product-readout.md).

### 22. Train and select directly for one-draw success

- **Tried:** replace NLL training/selection with expected-success training/selection, retaining
  modal-space and classical controls: minimize $-S=-N^{-1}\sum_t p_\theta(y_t\mid C_t)$ on the
  ten-qubit, two-stage/rank-128 product model.
- **Result:** product S rises **23.20% → 34.36%**, but NLL collapses **2.9281 → 8.0966**, with
  repetitive outputs. The success-trained MLP collapses to spaces.
- **Decision:** reject as a language-model improvement; return to NLL training/selection. A constant
  space predictor already gets **30.05% S**, and the NLL-trained MLP can use **40.58% argmax**.
- **Evidence:** [success-objective diagnostic](product-success.md). This is a score gain with a
  worse generator, not a failed implementation.

### 23. Stream ordered context blocks through successive Hamiltonians

- **Tried:** four chronological blocks with quantum evolution between blocks, versus pooled, short,
  frozen and matched MLP controls: ten qubits, $W=128$, four rank-32 Hamiltonian stages.
- **Result:** streaming NLL **3.2984**, versus pooled **2.9734**, short **3.1750**, and matched MLP
  **3.1300**; S **20.79%**.
- **Decision:** reject this streaming recipe. It provides neither improved prediction nor meaningful
  prose.
- **Evidence:** [streaming experiment](stream.md).

### 24. Cosine learning-rate decay for the quantum model

- **Tried:** decay learning rate from .003 to .0003, with fresh same-backend constant-rate controls
  on the DGX.
- **Result:** adaptive quantum NLL is **2.7021 constant versus 2.7013 cosine**, essentially
  unchanged. The MLP improves to **2.4077**, so its stronger schedule is retained as a control.
- **Decision:** retain constant-rate quantum training. Later unnormalized 90-epoch cosine also fails
  to establish a clear gain (**2.42985 versus 2.43218**).
- **Evidence:** [DGX cosine](dgx-cosine.md) · [later schedule controls](factor-norm.md). The earlier
  [CPU cosine protocol](adaptive-cosine.md) was superseded, not a second completed comparison.

### 25. Older-context dropout during training

- **Tried:** hide the older branch for .25/.50 of training examples while keeping intact-context
  validation and the same update budget.
- **Result:** seed-11 NLL **2.53538 / 2.52941**, versus full **2.54015**; both improvement intervals
  include zero. Neither passes the declared older-context criterion against short **2.52258**.
- **Decision:** retain unmasked quantum training. The .50 recipe is exploratory only; it establishes
  neither a regularization benefit nor useful older context.
- **Evidence:** [context-dropout screen](context-dropout.md). Samples remain incoherent, including
  two replacement characters in one .25 continuation.

### 26. A fourth stage specifically, rather than more capacity

- **Tried:** four-stage/rank-128 versus three-stage/rank-164 at almost equal parameters, on the
  expanded pool and 90-epoch budget.
- **Result:** NLL **2.40905 versus 2.41051**; depth-minus-width interval **[−.01247, +.01049]**
  crosses zero, and depth has slightly worse non-space NLL.
- **Decision:** the declared depth/interleaving rule fails. Both larger models beat the smaller
  reference, so extra capacity is a positive candidate; attributing that gain specifically to stage
  four is unsupported.
- **Boundary:** four stages improve S by .213 percentage points over width, a separate positive
  metric result. Neither model is seed-replicated, and no new four-stage short refit exists.
- **Evidence:** [capacity study](depth-four.md); see [positive capacity
  result](successful.md#18-increase-overall-quantum-capacity).

### 27. Normalize raw Hamiltonian factors after Adam steps

- **Tried:** keep each complex factor column at norm √2 before training and after every Adam step,
  leaving Adam moments unchanged; compare constant and cosine schedules separately.
- **Result:** constant NLL worsens **2.43218 → 2.66367**; cosine **2.42985 → 2.50069**. Both
  penalties have paired intervals entirely above zero; S, argmax, non-space and training losses
  worsen too.
- **Decision:** reject both tested normalized recipes; keep normalization disabled. Norms were
  controlled correctly, so this is a scientific negative result.
- **Boundary:** inference is invariant to positive factor scaling, but Adam's learning trajectory is
  not. This rejects the tested projection/moment recipe, not every possible normalization method.
- **Evidence:** [completed normalization study](factor-norm.md).

## Findings that should not be mistaken for failed experiments

- **Longer context:** most earlier 128-token refits lost or tied against 16 tokens, even after
  position/content/depth gains. Expanded coverage finally gave a replicated **NLL** benefit; S and
  argmax still favored short context. At 90 epochs, NLL evidence is inconclusive again. See
  [coverage replication](coverage-seeds.md) and [budget screen](budget-screen.md).
- **Gradient/plateau probes:** active NLP gradients and 9,216 initialization-gradient evaluations
  did not establish a barren plateau. Some final diagonal gates are exactly invisible to the
  measurement. This is a diagnostic outcome, not proof of a universal training limit. See [retained
  structural lessons](findings.md#4-plateaus-and-structural-lessons), H6.
- **Locked product diagnostics:** excess probability on unused targets was too small to explain the
  sampling gap; limited active-subspace mass motivated the rank test, but did not prove rank was the
  cause. See [product diagnostics](product-diagnostics.md).
- **Locked older-context interventions:** masking, reversing or replacing older tokens worsened
  full-model NLL. The branch is used, although retrained short models then generalized better.
  Sensitivity does not establish the benefit of retaining a branch during training. See [content
  diagnostics](content-diagnostics.md).
- **Locked Hamiltonian diagnostics:** dynamics and auxiliary entanglement were active. Raw factor
  norms grew 17–21×, and rescaling preserved predictions; that motivated normalization, but
  predicted no guaranteed training gain. See [Hamiltonian diagnostics](hamiltonian-diagnostics.md).
- **Resource, parity and audit checks:** CPU/CUDA comparisons, memory probes, microbatch timing,
  cleanup reproduction and report audits validate execution; they are not new predictive trials.
  Interrupted or merely proposed runs have no final success/failure score.

## Recovering historical evidence

The compact current [findings](findings.md) preserve the main historical conclusions. Older detailed
reports were intentionally removed during cleanup and remain in Git. The following source keys make
the earlier entries recoverable without reviving retired runners:

| Key | Commit | Report paths at that commit |
| --- | --- | --- |
| H1 | `fb71a4a` | `results/cluster-sweep/short-report.md` |
| H2 | `fb71a4a` | `results/focused-causal-1024/short-report.md`; `results/focused-interim-14/short-report.md`; `docs/runs.md` |
| H3 | `fb71a4a` | `docs/archive/SEMANTIC_ABLATIONS.md`; `SEMANTIC_GRID.md`; `SEMANTIC_HYPERPARAMS.md`; `SEMANTIC_MATRIX.md` in the same directory |
| H4 | `fb71a4a` | `results/attention-cv-25-ablation/report.md`; `results/attention-hyper-25/report.md` |
| H5 | `fb71a4a` | `docs/runs.md`; `docs/archive/PROJECT_RESEARCH_REVIEW.md`; the `results/robust-*` report directories listed there |
| H6 | `fde4934` | `docs/findings.md` (full retained robust benchmark, gradient evidence and recovery references) |
| H7 | `fde4934` | `experiments/encoding_study/REPORT.md`; `experiments/measurement_basis/REPORT.md` |
| H8 | `fde4934` | `docs/findings.md`; `experiments/hamiltonian_study/REPORT.md`; `CAUSAL_COMPARISON.md`; `CONTEXTUAL_POC.md` in the same directory |

For example: `git show fb71a4a:results/attention-cv-25-ablation/report.md`. Historical test scores
are explicitly separate from today's unscored WikiText test; no new training or held-out scoring was
performed to prepare this summary.
