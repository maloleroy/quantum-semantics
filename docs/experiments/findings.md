# Findings retained for future experiments

26 October 2026. These are historical measurements, not a new model comparison
performed during repository cleanup. Old runners are retired; new code uses a
common interface and defaults to sampling-success selection. Do not silently
reinterpret historical NLL selection or argmax accuracy as device performance.

## 1. Hamiltonian-only next-word prediction

Tatoeba: 246,072 sentences, 207,510 normalized groups. Train-only vocabulary:
255 words + UNK, eight qubits, four preceding tokens, fixed position/recency
amplitudes, 65,536 trainable H coordinates, direct Born readout. No learned
classical encoder/head or supplied prior is in this quantum predictor.

New grouped split seed 20261026; 32,768 optimization / 4,096 validation / 8,192
test targets. Test has 7,225 sentence groups and 16.98% UNK targets. Full count
references use 1,159,392 training transitions; the subset bigram sees only the
optimization examples. Every gradient epoch visits the fixed subset once.

| Predictor | One sample correct | Argmax accuracy | NLL |
|---|---:|---:|---:|
| Initial coherent H | 1.91% | 2.57% | 6.5861 |
| Trained coherent H | **6.76%** | **24.21%** | **3.5974** |
| Retrained dephased H | 5.71% | 19.51% | 3.6683 |
| Same-input classical MLP, 64,852 parameters | **24.77%** | **38.29%** | **2.6205** |
| Subset-only bigram | 18.95% | 34.09% | 2.8837 |
| All-training bigram | 19.90% | 34.46% | 2.6877 |
| Calibrated all-training bigram | 19.87% | 34.46% | 2.6867 |
| Directed classical context propagation | 12.15% | 30.60% | 3.1069 |
| Unigram | 4.55% | 16.98% | 4.2408 |

One-shot values are ideal simulated expectations, recovered post-hoc by averaging
exp(-individual NLL), not exp(-mean NLL). Hardware shots/noise were not tested.
The classical sampling column does not prohibit a classical argmax strategy.
Known-target one-shot success is 6.76% for H and 22.36% for MLP, without renormalization.

Three confirmation seeds 61/62/63 followed LR screening at seed 11. Adam, 30 epochs,
batch 512, no decay, clipping 5, float64/complex128. LR grid .001/.003/.01; selected
H/dephased .001, MLP .003, calibrated bigram .01. Checkpoints minimize full validation
NLL including epoch zero. All recipes/checkpoints were locked before test.

| Paired coherent-minus-control quantity | Difference | Sentence-group 95% interval |
|---|---:|---:|
| One-shot success vs dephasing | +1.050 pp | [+0.988, +1.111] pp |
| NLL vs dephasing | −0.07090 | [−0.08301, −0.05937] |
| NLL vs frozen initialization | −2.98867 | [−3.03383, −2.94038] |
| NLL vs matched MLP | +0.97690 | [+0.94433, +1.00890] |

Intervals condition on these fits/split, average seed differences, and resample
sentence groups. They do not include new-corpus or model-selection uncertainty.
H chose epoch 30 in every seed and the lowest tested LR; a hard plateau/convergence
ceiling was not established. Training accuracy was about 25%, versus 24% held out.

**Conclusion:** H training and a restricted coherence benefit transfer to language.
Both sampling and argmax still favor strong classical models. No practical quantum
advantage, entanglement-specific gain or hardware scalability has been established.

Artifacts: `output/retained/language/` has the original NPZ dataset/test arrays,
confirmed weights bundled as NPZ, CSV fit/history tables, figure and small summaries.
These generated files are local and ignored; transfer them explicitly for a handoff.
`checkpoints.npz` keys are `fit/stage/parameter` (e.g.
`confirm-coherent-s61/state/coefficients`). They are historical arrays, not new CLI
run directories. `provenance.json` preserves original hashes/settings. Original
runner source is in Git commit `fde49341d541988a7909127edec8bf9493abe218`.
The 24-fit pipeline took 260.493 seconds on eight one-thread Ryzen 7 7840S workers;
observed aggregate worker memory was about 4.5 GB. These are simulator timings.

## 2. Controlled Hamiltonian learning

| Task | H parameters / register | Initial argmax | Trained argmax | Retrained dephased |
|---|---|---:|---:|---:|
| 16-symbol modular composition | 256 / four qubits | 11.67% | 100.00% | 8.33% |
| UCI handwritten digits | 4,096 / six qubits | 5.84% | 91.96% | 61.88% |

These are argmax results, not new one-shot estimates. Five seeds 47–51 followed
validation screening; 80 fits took 27.664 seconds, excluding pilot/download/checks.

For symbols, target (a+b) mod 16 is computed by U† D_b U applied to |a>. A Fourier
solution exists; complete dephasing before D_b erases b, bounding uniform-population
accuracy at 1/16. Classical modular addition solves the task exactly, so this proves
a mechanism/architecture separation rather than superiority over classical algorithms.

Digits use fixed normalized pixel amplitudes and four measured qubits, with the
other two traced out. All 16 outcomes remain; unused digit labels count as errors.
The coherent model's NLL 1.0475 is below the post-hoc **1.3301** lower bound for every
completely dephased input followed by any unitary and the same rank-four readout.
The bound follows because a rank-four projector captures at most the four largest
eigenvalues of a diagonal input state. It is not an accuracy or general-classical
bound. A similar-size classical MLP reaches **95.43%**, NLL **0.1821**.

Arrays, confirmed weights, CSV histories and provenance are under
`output/retained/controlled/`. The digit bound does not transfer as a strong bound
for the language experiment's different measurement/input structure.

## 3. Earlier encoders, readouts and graph models

These studies used different vocabularies, training budgets and sometimes metrics.
Their numbers must not be pooled into one architecture ranking.

| Study | Quantified observation | Retained interpretation |
|---|---|---|
| Fixed-graph/global-time word walk, 512-word task | H walk NLL 4.2964 / top-1 8.72%; classical undirected diffusion 4.0350 / 17.26%; directed diffusion 3.4615 / 26.83% | Corpus structure helps, but classical propagation was stronger |
| Predictive amplitudes plus quantum walk | Test NLL 2.966171 vs classical calibrated bigram 2.966157; top-1 31.70% vs 31.63% | Small difference did not establish an additional quantum contribution |
| Fixed Hamiltonian encoder plus QCSE | Top-1 9.66% vs original QCSE 7.47%; NLL 5.2162 vs 5.1826 | Better argmax did not mean better likelihood |
| Directed source/target dilation screen | Selected epoch zero; validation NLL 2.9328; mean target-role success 0.000619 | Mostly implements a supplied bigram prior; conditional word scores hide rare-role cost |
| Latent hybrid context correction | Tuned validation NLL H 2.7940 vs classical 2.7816; argmax 36.17% vs 36.03%, three fresh seeds | Learned classical encoder/controller/head confound attribution; classical replacement has better NLL |
| QCSE encoding study, 1,246 held-out targets | Bigram-angle exact-word accuracy 16.59% vs original 4.92%; supplied classical bit prior already 16.77% | Useful statistics, not demonstrated quantum improvement; older vocabulary was transductive |
| Learned local measurement axes, 1,246 targets | BCE .5708→.5679; top-5 10.46%→12.07%; threshold word accuracy 4.04%→3.69% | Measurement basis matters, but improvement depends on the endpoint |
| Historical causal benchmark | MLP NLL 3.24873 / top-1 35.656%; q8 CRZ 3.46637 / 33.316%; circuit identity 3.47083 / 33.357% | MLP led; parameter counts/tuning differed, so no universal architecture claim |

The latent POC used an eight-dimensional quantum feature space followed by a
classical vocabulary head. That three-qubit latent state was not one basis state per
word. Its free weights were in the learned classical modules; this fact alone does
not prove the quantum feature map unused, but the controls did not establish its gain.

## 4. Plateaus and structural lessons

- Flat exact-word accuracy can coexist with improving probabilities/NLL. Historical
  NLP gradients stayed active (reported active-layer RMS at least .0161); initialization
  probes through q12 did not establish a vanishing-gradient explanation.
- Final diagonal gates are invisible under computational-basis probabilities unless
  followed by mixing/readout rotation. Their zero gradients have an exact structural cause.
- Independent-bit reconstruction discards joint correlations and is not joint Born
  measurement. Vocabulary-label equivariance and meaningful sentence order are separate.
- A shared unitary preserves fidelity; extra input-independent depth cannot create a
  new semantic geometry. A real-symmetric fixed walk is also reciprocal in basis transitions.
- Some historical “100 epochs” meant two expected passes of replacement draws, not
  100 corpus passes. Exposure, optimization, readout and task definitions must be separated.

No single cause of the original NLP plateau was established. The historical broad
benchmark snapshot is `fb71a4a`; current cleanup retains conclusions rather than its
execution infrastructure. Historical code/results needed for forensic work remain
recoverable from Git and the external pre-cleanup backup, not as active runners.

## 5. Cleanup verification

The new loader reproduced every retained language dataset array exactly. Loading
the confirmed historical parameters into the moved predictors reproduced all four
model families' mean sampling scores to within 1e-12. The five QCSE encoders and
RX/RZ/CRZ ansatz matched the old independent Qiskit reference within 3e-16.
The reduced suite has 17 passing scientific checks; strict Pyright reports zero
errors/warnings and Ruff passes. A short all-model CLI fit/evaluate/sample/plot run
was exercised. These verify the refactor; they are not new competitive benchmarks.

## 6. WikiText phase-interleaving screen (28 October 2026)

A validation-only nine-qubit screen tested two learned dense Hamiltonians with
fixed, nondecaying context RZ features between them, using a 32-word window,
512-word vocabulary, 8,192 gradient examples and 12 epochs. No learned classical
encoder/decoder or count prior is present in the quantum prediction path.
Interleaving worsened validation NLL (4.2596 vs 4.0110 for an independently
retrained, equally parameterized static control) and one-shot success (10.40%
vs 11.25%). The full-training bigram reached NLL 3.0300 and success 22.68%.
Better interleaving training NLL alongside worse validation NLL is consistent
with overfitting; shuffling older-word order did not worsen validation NLL.
Test was not scored. This rejects promotion of this particular recipe, not the
broader input-dependent architecture hypothesis. See the
[protocol, complete scores and limitations](interleaving.md).

## 7. Low-rank quantum evolution and larger training subset (28 October 2026)

Implemented exact action of `H = Q diag(e) Q†` with learned orthonormal complex
subspace Q and real energies e, avoiding dense exponentiation. Two rank-128
blocks on the 512-word register use 262,400 real trainable coordinates. With
32,768 training targets, window 32 and 30 epochs, static validation NLL reaches
**3.8744**, expected one-shot success **11.58%**, known-target success **2.04%**;
fit time **213.4 s**. The earlier two-dense-block static recipe gave NLL 4.0110
on only 8,192 training examples and 12 epochs; this is a recipe improvement,
not an isolated rank-effect claim. On 8,192 examples the new rank-128 static
recipe gave NLL 3.9902 in 65.7 s.

Fixed context-phase interleaving at scale .25 gives NLL 3.8830 on the expanded
subset, without an established NLL advantage over static (paired validation
interval includes zero). MLP on the same expanded subset gives NLL **3.3003**,
success **21.92%**; full-training bigram remains NLL **3.0300**, success **22.68%**.
Quantum inference is still direct word measurement with no learned classical
encoder/decoder. The QR/spectral method is a simulator improvement, not an
established efficient quantum compilation. Test remains untouched; rank/phase/
subset screens are adaptive validation exploration. See
[complete protocol and results](lowrank.md). Strict Pyright/Ruff pass and all
25 tests pass, including dense-reference gradients for the new exact evolution.

## 8. Auxiliary-register word measurement (28 October 2026)

Added joint low-rank Hamiltonian evolution on a nine-qubit word register plus
one auxiliary qubit, initialized to zero. Measuring only the word register keeps
all outcomes and uses no learned classical encoder/decoder. Two rank-64 blocks
use 262,272 real trainable coordinates. With the same 32,768 WikiText training
targets, 32-word window and 30 epochs, coherent validation NLL reaches **3.6379**,
sampling success **14.01%**, versus **3.8744 / 11.58%** for the earlier nearly
parameter-matched word-only model. Fits take 257–287 seconds.

The independently retrained dephased control is better on NLL (**3.5954**) and
known-target sampling (**2.45%** vs coherent **2.24%**), while coherent aggregate
sampling is higher (14.01% vs 13.65%) because of UNK. MLP (3.3003 / 21.92%) and
all-training bigram (3.0300 / 22.68%) remain stronger. The auxiliary register is
active and joint states are entangled, but no entanglement-specific advantage
was tested. Post-fit truncation from 32 to four context words worsens coherent
NLL to 3.7195; keeping 16 yields 3.6427. This is sensitivity, not a retrained
long-context comparison. Validation selection is adaptive, one seed, and test
remains untouched. See [complete methods and CSV evidence](ancilla.md).
All 30 tests, strict Pyright and Ruff pass, including density-matrix probability
and gradient checks and the single-auxiliary-state equivalence control.

## 9. Slower decay and independently retrained context controls (28 October 2026)

Exposed fixed amplitude decay through model/CLI/checkpoint configuration and
screened .9/.97 against .7 on the same 512-word WikiText inputs. On 32,768
training targets, decay .97 worsens coherent NLL **3.6379 → 3.7215** and MLP NLL
**3.3003 → 3.4949**. Dephased aggregate NLL improves **3.5954 → 3.5643**, but
known-target NLL/sampling worsen; the aggregate gain is driven by UNK and its
paired NLL interval includes zero. Retain .7 as the coherent reference.

Independently retrained four-word controls at decay .97 give coherent NLL 3.7730
and dephased NLL 3.7826 versus 32-word scores 3.7215/3.5643. The dephased
32-word-minus-four NLL difference is −.21836, descriptive article-group interval
[−.28222, −.16360], and known-target NLL also improves. Coherent aggregate NLL
interval includes zero and its known-target metrics worsen. Thus this provides
a context-length benefit for a dephased population model, not a demonstrated
quantum coherence or classical-computation advantage. All models still trail
stronger classical references. These are adaptive, single-seed validation
experiments, with test untouched. See [protocol and full CSV results](decay.md).
All new fits finished within ten minutes; 31 tests, strict Pyright and Ruff pass.

## 10. Separating recent and distant quantum input streams (28 October 2026)

Prepared recent-four (.7 decay) and older-word (.97 decay) amplitudes in separate
sectors of the existing auxiliary qubit, retaining 262,272 Hamiltonian parameters.
On 8,192 training targets, coherent validation NLL improves to **3.7312** from
3.7892 for the earlier auxiliary recipe, but sampling falls **13.96% → 12.34%**.
Independently removing only interference between sectors gives better NLL
**3.6785**; a same-input, approximately parameter-matched MLP reaches **3.4824**
and sampling **19.39%**. Known-word sampling is 1.97%, 2.11%, 3.13% respectively.
There is no demonstrated cross-sector interference gain or classical advantage.

Both quantum fits select epoch 30 and have a large train/validation NLL gap.
A 131,072-target dataset is prepared with verified identical validation and
statistics, and a checksum-guarded twelve-fit, three-seed Bash runner is ready at
`output/segmented-large/run.sh`. Individual larger fits exceed ten minutes, so
that experiment was handed to the user. As of 29 October, all twelve fits
are complete and verified; see the [full audit](segmented-large.md). Test is untouched.
See [methods, screen evidence and handoff](segmented.md). All 36 tests, strict
Pyright and Ruff pass; scientific tests verify sector density matrices and gradients.


## 11. Completed larger study (29 October 2026)

All twelve fits in `output/segmented-large/batch-IAnOGm` are verified, including
recovered sector-mixture seeds 22 and 33. Source/data/weight locks passed, and
reloaded validation predictions reproduced saved scores within 1e-12 tolerances.
No held-out test was scored. Three-seed mean validation NLL/sampling: earlier
auxiliary **3.5547 / 14.04%**, coherent segmentation **3.5655 / 12.37%**, sector
mixture **3.4884 / 12.26%**, same-input MLP **3.1641 / 22.09%**. Mixture NLL seed
SD is .00277; the recovered fits are consistent with the first seed.

Coherent minus mixture NLL is +.07710, descriptive article-group interval
[+.06409, +.08985]; known-target NLL is also worse. Coherent aggregate sampling
is slightly higher, but known-word sampling is lower (2.32% vs 2.36%). The
mixture retains within-stream coherence. MLP leads on all reported aggregate
and known-target metrics. No useful cross-sector interference gain or general
quantum advantage is demonstrated. See [complete results and limitations](segmented-large.md).
The completed recovery took 32–33 minutes per fit with two workers; no recovery
remains. Scaling coherent segmentation unchanged is not supported by this study.

## 12. Frozen-core input allocation diagnostic (29 October 2026)

Fit one recent-sector probability on all 131,072 training targets for each
locked mixture seed, then evaluate validation. The fitted probabilities stay
near .515; three-seed mean NLL improves only **3.488359 → 3.487537**, while
aggregate sampling decreases **12.2600% → 12.2408%**. Known-target sampling
slightly improves **2.3647% → 2.3812%**. This changes randomized quantum input
preparation, with no learned classical word decoder. All three fits and scoring
took approximately 105 seconds; test remains untouched. The equal-allocation
reference reproduces saved predictions within 1e-12 tolerances. See
[protocol and CSV evidence](sector-calibration.md). This small frozen-core gain
does not explain the MLP gap or justify another long run just for this scalar.

## 13. Learned context-gate pilot (29 October 2026)

Added learned pooled token angles controlling local RZ gates between two
auxiliary-register Hamiltonian blocks, with direct word measurement and no
classical vocabulary decoder. At 8,192 WikiText training targets, the selected
LR .003 recipe gives validation NLL **3.6796**, sampling **13.54%**, known-target
sampling **2.75%**. Static auxiliary at LR .001 gives **3.7892 / 13.96% / 2.07%**;
the improvement is in NLL and known words, with aggregate sampling lower.
The learned recipe still trails the classical feature control (**3.4744 NLL,
19.49% sampling**). Frozen quantum dynamics gives NLL 4.7796; the small angle
table alone does not reproduce the jointly trained recipe at this budget.

All six pilot fits were verified; all 42 tests, strict Pyright and Ruff passed.
The expanded 32,768-target screen fixes LR .003 for all controls and independently
retrains a four-word-context model. All six fits are now complete and verified
after the user-run recovery; see section 14. Training is not launched or monitored
by the assistant under the user's updated execution preference. Test remains
untouched. See [protocol and complete evidence](conditional.md).


## 14. Expanded context-gate controls (30 October 2026)

All six fits completed and passed source/data/weight checks, validation prediction
recomputation and table/paired-interval audits. At 32,768 training targets with
LR .003 for every model, learned context gates improve quantum validation NLL
**3.6360 → 3.5518** and sampling **14.09% → 14.36%** relative to static auxiliary
evolution. Fixed gates give **3.6230 / 14.12%**; frozen quantum dynamics gives
**4.7747 / 10.17%**. The gain survives the LR control, with an extra 15,360 learned
angle parameters. This supports the recipe's dependence on trained dynamics,
not an advantage over classical computation.

An independently trained four-word conditional model does better on the point
estimates (**3.5274 NLL / 14.84% sampling / 2.99% known-target sampling**) than
32 words (**3.5518 / 14.36% / 2.71%**). The paired NLL interval includes zero,
while both sampling intervals favor the shorter model. Useful older-context
computation is not established. The classical feature control remains stronger
(**3.2210 / 21.92% / 5.49%**). These are single-seed, adaptively selected validation
results, with test untouched. Full results, intervals and resource timings are
in the [expanded report](conditional.md#completed-expanded-results).

Do not scale the unchanged long-context recipe on these results. Learned gates
are a better quantum reference at this budget, but competitive generation remains
unresolved. The 41.80% validation UNK rate and loss of punctuation/numbers/case
also obstruct meaningful text generation; an open-vocabulary task would require
explicit new tokenization and scoring, without treating its per-token scores as
comparable to the current per-word numbers.


## 15. Open-vocabulary preparation and generation (30 October 2026)

Implemented deterministic training-only byte BPE, retaining ID 0 as UNK and
covering all 256 bytes before learned merges in a 512-outcome register. Case,
punctuation, numbers and within-article whitespace are preserved. NPZ datasets
carry merge tables and target byte lengths; scoring adds sampled-target bits per
byte with an explicit denominator. New autoregressive generation retains every
sampled outcome, including UNK/unused tokens, and renders joined byte fragments.
Word datasets retain their existing defaults. These are representation/software
changes, not measured improvements over word-level models.

All 45 tests, strict Pyright and Ruff pass, including round-trip/leakage/metric/
checkpoint tests. A guarded six-fit manual comparison is prepared at
`output/byte-bpe-screen/run.sh`, with 128-token and retrained 16-token contexts,
static/fixed/frozen quantum controls, a matched classical feature control, count
references and fixed-prompt sample diagnostics. The user subsequently completed all six fits; results are in section 16.
No test was scored. See the [declared protocol,
scoring limits and command](byte-bpe.md). Training remains manual.


## 16. Byte-BPE results and direct generation (30 October 2026)

All six fits, count references, paired intervals and saved sample renderings
were verified. The tokenizer learned 255 merges in 41 seconds of preparation,
with zero UNK targets and exact source-byte coverage. That representation gain
has not yielded competitive predictions: learned gates give **4.2773 token NLL /
5.68% sampling / 3.8501 target bits per byte**, versus MLP **3.4424 / 24.14% /
3.0986** and all-training bigram **3.0593 / 23.16% / 2.7537**. The bigram uses
6.86 million training transitions; gradient models use 32,768 fixed targets.

Learned gates improve over static quantum NLL 4.4888 and sampling 5.26%, but
barely improve unigram NLL 4.2945 and trail unigram sampling 9.30%. The 16-token
conditional control has better NLL (4.2459); sampling is similar (5.66%). The
128-token context has not shown a useful sampling benefit. Fixed-prompt samples
contain fragments/repetition, with replacement/control characters in some quantum
outputs, despite no sampled UNK IDs. This is not meaningful long-text generation.
Single-seed adaptive validation and sampled byte-denominator limitations remain;
test was untouched. See [complete protocol and evidence](byte-bpe.md#completed-results).

The next controlled hypothesis changes initial state preparation to learned local
rotations while retaining Hamiltonian computation and direct measurement. More
training data is another unresolved axis; neither is a proven remedy.


## 17. Product-state preparation implementation (30 October 2026)

Added `product`, `product_fixed` and `product_mlp`: learned local RY preparation
of the ten-qubit joint register, followed by the existing Hamiltonians, learned
context RZ interleaving and direct token measurement. The quantum model has
293,012 parameters (262,272 in dynamics); the classical feature control has
292,695. There is no learned classical vocabulary decoder. Frozen-controller
and frozen-core controls separately test dependence on learned preparation and
trained dynamics; none establishes advantage without results.

All 51 tests, strict Pyright and Ruff pass, including dense probability/gradient
references and initial-state separability. A five-fit manual screen reuses the
exact full/short byte datasets and a pinned, previously audited amplitude-model
reference with its matching archived source. No tokenizer refit or duplicate
reference training is required. The user subsequently completed the training/summary pipeline; results are in
section 18 and test remains untouched. See the [declared protocol and command](product.md).


## 18. Product-state results (30 October 2026)

All five locked fits and their saved predictions were verified, together with
count references, the prior amplitude reference, 21 paired intervals and 21
saved sample renderings. Product preparation improves validation NLL from
4.2773 to **3.1975** and sampling success from 5.68% to **21.79%**. Trained
Hamiltonians improve over the frozen-core control (4.3635 NLL / 16.82% sampling).
The matched MLP reaches 3.2306 NLL / 26.48% sampling: a small single-seed product
NLL lead coexists with a substantial sampling deficit. Bigram still leads both
NLL and sampling, with much more training-transition exposure.

The 16-token product control gives 3.1859 NLL / 21.88% sampling; longer context
still has no demonstrated benefit. Samples remain fragmented, despite no invalid
text outcomes in the three learned-product draws. These are adaptive validation
results, not confirmation of quantum advantage or meaningful generation. See the
[complete evidence and limitations](product.md#completed-results).

Next test: increase unique training targets with the tokenizer and held-out
partitions fixed, retaining equal-update small-data and matched-classical controls.


## 19. Nested training-exposure comparison prepared (30 October 2026)

Prepared a manual six-fit [exposure comparison](product-exposure.md): expand the
fixed gradient set from 32,768 to 131,072 targets, keeping the byte tokenizer,
validation/test arrays and all-training count statistics unchanged. The expanded
set contains every original training target. Product, short-context, frozen-core
and matched-MLP fits use 30 epochs; original-set product/MLP controls use 120,
equalizing optimizer updates at 7,680. Validation checkpoint counts differ with
epoch count and are declared, so this is not an equal-selection-budget claim.

Small causal-context/nested-sampling checks and generated-tool Ruff/Pyright pass.
Full preparation and training subsequently completed; see section 20. The command uses the local long-job
controller with event-only delivery; execution remains manual. No source-model
change or test scoring was made for this next screen.


## 20. Expanded product-state exposure results (30 October 2026)

Six completed fits, nine metric rows, 30 paired comparisons and 24 saved sample
renderings were audited. Product validation reaches **2.9673 NLL / 23.17% sampling**,
versus **3.1793 / 21.91%** when repeating the original small dataset for equal
updates. More distinct training targets help. The matched expanded-data MLP,
however, reaches **2.7966 / 30.84%**, eliminating the earlier product NLL lead.
Trained quantum dynamics still improve over frozen dynamics (4.2567 / 18.00%).

Product beats bigram NLL (3.0593) and argmax, but sampling is indistinguishable
from bigram's 23.16% under the descriptive interval. The 16-token product yields
2.9667 NLL / 23.11% sampling; older context adds only .060 sampling percentage
points and no NLL benefit. Samples remain fragmented and can contain invalid
UTF-8/control outcomes. Test remains untouched. See the
[complete evidence](product-exposure.md#completed-results).

Next investigate a sequential local-rotation encoding of token order, retaining
trained Hamiltonians and direct measurement, with independently retrained pooled,
frozen and matched-classical controls. This is an encoding hypothesis, not an
established cause of the current deficit or an entanglement advantage claim.


## 21. Sequential local-rotation preparation implemented (31 October 2026)

Added `sequence`, `sequence_pooled`, `sequence_fixed` and `sequence_mlp`.
Token-conditioned RY/RZ gates compose oldest-to-newest with fixed .97 age decay,
then the existing Hamiltonians and pooled RZ interleaving produce direct token
measurement. The quantum model has 287,902 parameters (262,272 in dynamics).
The matched MLP has 287,711 and receives an equivalent compact Bloch description
of the separable prepared state, without unobservable phase information.

All 58 tests, Ruff and strict Pyright pass, including explicit gate/dense-gradient
references, separability, order/padding checks and Bloch-state reconstruction.
No predictive gain is established. A six-fit manual comparison reuses the exact
expanded full/short datasets and pinned product/product-MLP predictions with their
matching archived source. Training subsequently completed; see section 22. Test is untouched. See the
[declared protocol and command](sequence.md).


## 22. Sequential preparation regresses (31 October 2026)

All six fits, ten metric rows, 27 paired intervals and 24 saved sample renderings
were verified. Sequential preparation gives **3.1173 NLL / 21.32% sampling**,
worse than the previous product model's **2.9673 / 23.17%**, and essentially tied
with the summed-angle control (3.1150 / 21.34%). Its matched MLP gives
3.0381 / 27.08%. Longer context worsens NLL despite a small sampling gain; text
remains fragmented. No test scoring occurred. The learned fit took **3.44 hours**,
exceeding the 45–90 minute estimate. Do not scale this recipe. See
[results and timing limitations](sequence.md#completed-results).

Read-only [product diagnostics](product-diagnostics.md), under matching archived
source, reproduce both best product/MLP checkpoints. Beyond the most common target,
product sampling is 4.19% versus MLP 10.85%. Zero-training-target outcomes consume
1.08% product probability, too little to explain most of the overall gap. Incoming
mass in the two learned rank-64 subspaces averages 26.6% and 25.5%; this motivates
but does not prove a rank-capacity hypothesis. Return to the faster product family
with matched classical capacity controls before further scaling.


## 23. Product rank-capacity comparison prepared (31 October 2026)

Prepared four manual fits: product and matched product-feature MLP at ranks
32 and 128, retaining the audited rank-64 results. All use the same 131,072-target
training set and 30-epoch NLL-selected protocol. Parameter budgets are 161,876 /
161,200 at rank 32 and 555,284 / 554,138 at rank 128 (quantum / classical).
Four workers request two threads each; timing is not directly comparable with
the old one-thread runs. Source/data/reference locks and generated-tool checks
pass; no model source changed and no new training or test scoring occurred.
See the [declared limitations and manual command](product-capacity.md).


## 24. Rank improves NLL but leaves the sampling gap (31 October 2026)

Four new fits, eight metric rows, 30 paired comparisons and 18 saved sample
renderings were verified. Product NLL improves across ranks 32/64/128:
3.0327 / 2.9673 / **2.9281**. Rank-128 sampling is **23.20%**, effectively unchanged
from rank 64's 23.17% and bigram's 23.16% under the descriptive intervals. Its
matched MLP reaches **2.8303 NLL / 31.04% sampling**; even the small MLP beats the
larger quantum model. Text remains fragmented. No test was scored.

The longest new fit took 20.7 minutes with two threads, below the broad 1–3-hour
estimate. Thread/concurrency differences prevent a rank-only timing conclusion.
Do not escalate rank solely on this modest NLL gain. See the
[complete evidence](product-capacity.md#completed-results). The next bounded
hypothesis changes the local quantum measurement basis with 18 additional angles,
retaining direct measurement and attribution controls.


## 25. Local quantum measurement basis implemented (31 October 2026)

Product models now support `--decoder rotated`: RX/RY on the nine token qubits,
with 18 extra trainable parameters and direct measurement. Default Born models
retain their old parameter structure. No rotations are allocated to the discarded
auxiliary. All 60 tests, Ruff and strict Pyright pass, including dense probability/
gradient references, initial identity, auxiliary-rotation invariance and checkpoint
restoration. No improvement is measured yet.

Prepared four manual fits: rotated ranks 64/128, fixed-controller rank 128 and
frozen-Hamiltonian rank 128 (readout still trainable). The exact expanded dataset
and four pinned unrotated/MLP references are reused. Archive locks and generated
tool checks pass; training is unrun and test is untouched. See the
[declared protocol and command](product-readout.md).


## 26. Local measurement basis adds no useful gain (1 November 2026)

Four new fits, ten metric rows, 33 paired comparisons and 18 saved sample renderings
were audited. Rank-128 rotated NLL is **2.92786** versus unrotated **2.92811**;
sampling falls slightly to **23.11%** from **23.20%**. Rank-64 effects are also
negligible. Both remain below the matched MLP, and text remains fragmented.
Trained-Hamiltonian controls still improve over frozen Hamiltonians, but this does
not make the added readout useful. No test scoring occurred.

The longest fit took 96.2 minutes, slightly exceeding the 30–90 minute estimate
and much slower than the previous unrotated run. This is not a controlled speed
benchmark. Keep unrotated product as reference; do not scale this readout recipe.
See the [complete evidence](product-readout.md#completed-results). Next separate
direct expected-success training from the NLL surrogate, with a dominant-token
baseline and explicit language-quality/NLL safeguards in interpretation.


## 27. Direct one-shot objective diagnostic prepared (1 November 2026)

Prepared a three-fit manual [success-objective diagnostic](product-success.md):
unrotated rank-128 product, matched MLP and frozen Hamiltonians, all trained and
selected by expected one-shot success. Exact data and other budgets remain fixed;
the audited NLL-trained references are retained. This changes loss and selection
together and is not an isolated loss-effect claim.

A training-chosen constant modal-token baseline (space, ID 33) scores 30.05%
validation sampling/argmax with infinite NLL on other targets. It makes explicit
that beating the old 23.20% quantum sampling score alone need not mean useful
context modeling. Separate non-modal-target scores, NLL and unfiltered samples
will expose collapse. No candidate is promoted for generation solely on S.
Generated-tool/source/data/reference checks pass; training is unrun and test is
untouched. No model code changed.


## 28. Direct success improves the metric but damages generation (1 November 2026)

Three fits, eight metric rows, 23 paired comparisons, 16 target buckets and 15
saved sample renderings were independently verified. Product sampling rises to
**34.36%**, above both its NLL reference (23.20%) and the modal baseline (30.05%),
but NLL degrades to **8.0966** and argmax declines. Its outputs repeat only five
token IDs per saved continuation, half spaces. The success-trained MLP collapses
to spaces, while the NLL-trained MLP's available argmax remains **40.58%**, above
quantum one-shot success. This is not a generator improvement or practical quantum
advantage. No test scoring occurred; the longest fit took 20.7 minutes.

Keep the NLL-trained generator reference. Future generation experiments again
use declared NLL training/selection and retain sampling, argmax, modal/non-modal
buckets and unfiltered samples to prevent score-only promotion. See the
[complete evidence](product-success.md#completed-results). Next investigate quantum
evolution between ordered context blocks, with matched pooled/classical controls,
instead of further rank/readout changes or pure-success optimization.


## 29. Context-block evolution prepared (1 November 2026)

Implemented `stream`, `stream_pooled` and `stream_mlp`. The quantum model processes
four chronological context blocks with learned Hamiltonians between them, using
local RY preparation then local RZ inputs and direct token measurement. All 66 tests,
Ruff and strict Pyright pass, including dense probability/gradient references.

Prepared five manual NLL-trained/selected fits with pooled, short-context,
frozen-dynamics and matched MLP controls. Source/data/reference locks and generated
tool checks pass. Four archived NLL references, count/modal baselines, target buckets
and unfiltered samples remain visible. No training or test scoring has occurred;
there is no measured gain. See the [protocol and command](stream.md).


## 30. Ordered streaming regresses (1 November 2026)

All five fits and their locked predictions were independently reproduced. Ordered
streaming reaches **3.2984 NLL / 20.79% sampling**, worse than repeated full-context
pooling (**2.9734 / 23.49%**), short context and its matched MLP. Twelve metric rows,
32 paired intervals, 24 target buckets and 18 saved sample renderings were checked.
No test scoring occurred. The longest fit took 35.9 minutes.

The recipe does not improve longer-context prediction or produce meaningful prose.
Keep the earlier product reference. Next test broader training coverage against
repeating the current sample for equal updates, with classical comparisons; only
1.91% of eligible training transitions currently receive gradient training.
See the [completed evidence](stream.md#completed-results).


## 31. Broader training coverage prepared (1 November 2026)

Prepared four manual fits of the retained rank-128 product and matched MLP:
524,288 unique training positions versus four copies of the existing 131,072.
Both datasets have equal row counts, 30 epochs, 30,720 optimizer updates and the
same validation checkpoint schedule. Tokenizer/count statistics and validation/test
partitions stay fixed; no new source data are introduced.

Nested-position and causal-materialization helper checks, generated-tool typing/
linting, shell syntax and pinned reference/source checks pass. The launcher performs
full dataset preparation/verification before training. Neither preparation nor
training has run. Four fits use two threads each; the unmeasured planning estimate
is 1–3 hours. See the [protocol and command](product-coverage.md).


## 32. Coverage helps, while the classical gap persists (1 November 2026)

All four fits, dataset invariants, locked predictions and reports were audited.
Expanded product reaches **2.7870 NLL / 24.42% sampling**, versus **2.9077 / 23.79%**
for equal-update repetition. The expanded MLP reaches **2.4941 / 36.66%**, improving
more than quantum. Eleven metric rows, 32 paired intervals, 22 target buckets and
18 saved sample renderings were checked. No held-out test was scored.

Quantum samples remain fragmented and include invalid/control characters. The
small train/validation NLL gap does not diagnose a cause, but additional coverage
alone has not solved representation or optimization limits. Keep the expanded-data
fits as references. Next investigate context-dependent spectral phases with static
capacity and classical controls. The longest fit took 100.9 minutes. See the
[completed evidence](product-coverage.md#completed-results).


## 33. Context-dependent spectra implemented (1 November 2026)

Added `adaptive` and `adaptive_mlp`: linear pooled context shifts the eigenvalues
inside each learned rank-128 Hamiltonian basis. Zero-initialized shifts preserve
the original product predictions at initialization. All 71 tests, Ruff and strict
Pyright pass, including batched dense probability/gradient comparisons.

Prepared four manual fits on the exact 524,288-position dataset: adaptive, matched
rank-176 static capacity, fixed bases/base energies, and matched MLP. The fixed-basis
control still trains context-dependent spectral shifts; it is not a frozen H.
Both audited expanded-data references remain pinned. Generated-tool and reference
checks pass. Training is unrun; planning estimate 2–5 hours. See the
[protocol and command](adaptive-spectrum.md).


## 34. Adaptive spectra improve the quantum recipe (2 November 2026)

All four fits and locked predictions were independently reproduced. Adaptive spectra
reach **2.7007 NLL / 25.77% sampling**, beating matched static capacity (**2.7558 /
24.62%**) but trailing the matched MLP (**2.4393 / 38.44%**). Nine metric rows,
23 paired intervals, 18 target buckets and 18 saved sample renderings were checked.
No test was scored. The longest fit took 130.6 minutes.

Text remains fragmented, including an UNK and invalid/control characters. This is
a within-quantum recipe improvement, not quantum superiority or meaningful long
text. Retain adaptive as the quantum reference. Next test fixed cosine learning-rate
decay with static/classical controls and short-context retraining. See the
[completed evidence](adaptive-spectrum.md#completed-results).


## 35. Learning-rate decay and context control prepared (2 November 2026)

Implemented opt-in epoch-wise cosine decay, retaining constant updates by default.
All 73 tests, Ruff and strict Pyright pass, including schedule endpoints, applied
update effects, constant-equivalent weights and locked selection/metadata history.

Prepared full/short adaptive, matched static and matched MLP fits, with a fixed
.003-to-.0003 schedule over 30 epochs on unchanged training targets. Three audited
constant-rate references remain pinned. Generated-tool checks and small masking
checks pass. Full short-dataset preparation and research training are unrun;
planning estimate 2–5 hours. See the [protocol and command](adaptive-cosine.md).

## 36. CUDA execution and simulator throughput verified (2 November 2026)

All 33 model kinds have matching CPU/CUDA probabilities and gradients, with
portable NPZ weights and locked training/evaluation. Final verification on
3 November passes 109 tests, Ruff, strict Pyright and all experiment CLI stages.
The laptop retains CPU dependencies; DGX Slurm jobs require CUDA and use the optional CUDA build.

Three-repeat, same-data/precision benchmarks on a full A100 give **4.14×** faster
adaptive quantum fits and **3.30×** faster matched MLP fits than the faster of
four/sixteen-thread CPU medians. GPU kernels were profiled and selected weights
agree within tolerance. This is simulator throughput, not a quantum hardware
advantage or a new predictive result; benchmark test partitions were not scored.
See the [DGX setup, timings and scope](../dgx.md).


## 37. Controlled cosine comparison moved to CUDA (3 November 2026)

The renewed goal authorizes short local experiments and longer DGX GPU work,
replacing the earlier manual launch preference. DGX access and its CUDA dependency
build are available. Prepared seven same-backend fits: constant/cosine adaptive,
static and MLP, plus cosine short-context adaptive. Fresh CUDA constant controls
avoid attributing backend changes to the learning-rate schedule. Original data,
batch 512, update count, seed and NLL selection remain fixed.

The isolated runner requires actual CUDA tensors/forward/gradients and CUDA parity
tests before fitting, then verifies locked predictions and retrieves artifacts.
No new predictive evidence is yet available. See the [declared DGX protocol](dgx-cosine.md).


## 38. Cosine decay helps the MLP, not the quantum reference (4 November 2026)

The seven-fit DGX comparison completed in 2h08m. A lost SSH connection caused
the local failure notification; Slurm completed successfully, and all artifacts
were recovered and checked without restarting training. GPU checkpoint reloads
reproduced validation arrays. The local audit checked all locks, selected epochs,
ten metric rows, 20 target buckets, 32 paired intervals and 24 sample renderings.
No held-out test was scored.

Adaptive quantum NLL barely changes (**2.7021 constant → 2.7013 cosine**), while
the matched MLP improves (**2.4393 → 2.4077**). Retrained 16-token adaptive NLL
is **2.7010**: the study provides no evidence that the older 112 tokens help.
Non-space sampling remains **6.08% quantum versus 19.30% MLP**, and quantum text
is fragmented with invalid/control characters. Retain constant adaptive and
the stronger cosine MLP as references. The next representation experiment should
target conditional non-space distributions and quantum attribution, not another
schedule-only change. See the [audited results](dgx-cosine.md#completed-results).


## 39. Count-initialized quantum refinement prepared (4 November 2026)

Added `prior`, `prior_unigram` and `prior_mlp`. Fixed training-count amplitudes
provide the initial distribution; contextual Hamiltonians learn refinements and
direct Born measurement produces tokens. The matched classical logit correction
has the same input features and starts at the same count distribution. Quantum
and MLP budgets are 736,512 and 735,830 trainable parameters, plus the declared
fixed count cache. No learned classical vocabulary head is added to the quantum
candidate, and no preparation/compilation speedup is assumed.

All 82 CPU tests, Ruff and strict Pyright pass; 37 CUDA checks await DGX execution.
Tests cover exact count initialization, padding/train-only statistics, dense
Hamiltonian probabilities and gradients, learning away from identity, explicit
fixed-basis behavior and locked training/checkpoints. The six-fit protocol also
includes unigram input, a short-context refit and constant/cosine MLP controls.
Prepared-data masks and archived reference locks pass local checks. Scientific
training was unrun at declaration. Subsequently, Slurm job 10876 passed all 119
tests on the DGX and the full-size CUDA learning gate, then entered the first fit.
No predictive result is yet available; see the [declared comparison](prior-screen.md).


## 40. Count initialization improves quantum NLL with a sampling tradeoff (5 November 2026)

All six DGX fits completed in 1h40m; local observer failures were network failures.
Recovered artifacts and input hashes passed. GPU checkpoint reloads and the
independent local audit verified every selected prediction/report and all 21
saved sample renderings. No held-out test was scored.

Quantum NLL improves **2.7021 → 2.6676**, meeting the declared retention criterion,
while total sampling falls **25.77% → 25.38%**. Non-space sampling improves
**6.08% → 6.50%**. Fixed-basis NLL **3.0198** and unigram-input **2.7178** show
that learned Hamiltonian bases and the bigram prior both contribute in this family.
The short-context refit reaches **2.6730**; useful older context remains unproven.

Same-input cosine MLP reaches **2.4726**, and the strongest archived MLP remains
better at **2.4077**. The three new quantum samples avoid invalid/control bytes
but remain fragmented. Retain `prior` as the provisional quantum NLL reference,
preserve its sampling regression, and keep the archived cosine MLP as the strong
classical comparison. These single-seed reused-validation results do not establish
quantum advantage or coherent long text. See the [completed audit](prior-screen.md#completed-results).


## 41. Seed replication declared with validation reuse explicit (5 November 2026)

The count-initialized NLL gain is still based on seed 11. Prepared six fresh fits:
prior, adaptive quantum and the strongest cosine adaptive MLP at seeds 22 and 33.
Keep all original data, recipes, update budgets and selection rules; include locked
seed-11 references in the three-seed report. No new architecture is introduced.

All 60 official validation articles already appear in the existing 4,096-target
selection set. This is a seed-robustness study, not independent article-level
confirmation. Test remains untouched. Require an NLL gain at both new seeds and
on the three-seed mean, report every fit and preserve sampling/argmax tradeoffs.
Input/reference checks, strict generated-tool typing/linting and two reporting
checks pass; they distinguish mean scores from ensemble predictions and reject
averaging away a failing new seed. Slurm job 10895 subsequently passed all 119
DGX tests and the six full-size CUDA learning checks, then started `prior_s22`.
Results remain pending. See the [declared protocol](prior-seeds.md).


## 42. Prior NLL improvement replicates across seeds (5 November 2026)

Slurm 10895 completed all six new fits in 1h32m. Inputs/artifacts, GPU checkpoint
reloads, every report table, the predeclared decision and all 18 saved sample
renderings passed audit. The held-out test was not scored.

Prior beats adaptive at both new seeds: NLL differences **−0.02858** and
**−0.03801**. Three-seed mean NLL is **2.66776 prior versus 2.70146 adaptive**,
so the declared replication rule passes. Mean sampling remains slightly lower
(**25.51% versus 25.74%**), while mean argmax rises (**40.76% versus 40.29%**).
The stronger MLP remains ahead at **2.41207 NLL / 39.41% sampling**. Prior's
non-space sampling reaches 6.54%, versus 6.07% adaptive and 19.66% MLP.

New prior samples remain fragmented, with one invalid UTF-8 sequence among the
six continuations. Retain the prior quantum NLL reference and the stronger MLP
recipe. The gain is replicated across these seeds on reused validation; all 60
validation articles were already exposed, and this is not independent-partition
confirmation, useful long text or quantum advantage. See the
[completed comparison](prior-seeds.md#completed-results).


## 43. Feature-specific position weighting prepared (5 November 2026)

Added `prior_position` and `prior_position_mlp`: learn separate normalized lag
weights for every spectral-shift and RZ feature, initialized to the old fixed
decays. Quantum prediction still uses count amplitudes, contextual Hamiltonians
and direct Born measurement. The candidate has 754,038 trainable parameters;
the fixed-pooling capacity control has 753,414 and the matched MLP 753,356.

All 91 CPU tests, Ruff and strict Pyright pass; 39 CUDA checks await the DGX.
Prepared five seed-11 fits: full context, fixed-pooling capacity, refitted 16-token
context, fixed bases and cosine MLP. Data, target partitions, training budget and
NLL selection remain fixed. Keep the stronger archived MLP and prior visible.
This is exploratory reuse of validation; test remains unscored. No measured
improvement is yet available. See the [declared protocol](prior-position.md).

Slurm 10898 subsequently passed all 130 DGX tests and all five full-size CUDA
learning checks, then started the candidate fit with 89–90% observed GPU utilization.
Source/data/reference manifests remain pinned. Results are pending; goal work pauses
after this verified startup while the experiment and completion delivery continue.


## 44. Learned positions improve local prediction, not demonstrated long context (5 November 2026)

Slurm 10898 completed all five fits in 1h32m despite a failed SSH observer. Recovery
verified all source/data/artifact hashes without resubmission. GPU checkpoint
reloads and the independent local audit reproduced all reports, decisions and
18 sample renderings. No test scoring occurred.

Full-context quantum NLL reaches **2.59512**, versus **2.66761** prior and
**2.66849** matched-capacity pooling. Sampling rises **25.38% → 26.66%** and
non-space sampling **6.50% → 7.36%**. The declared retention rule passes. The
single-seed gain requires replication; it does not inherit the old recipe's
three-seed evidence.

The refitted 16-token model reaches **2.58684**; full minus short NLL is +.00828,
article interval [−.00433, +.02002]. The older-context criterion fails. Learned
recent spectral weights put 96.09% of their mass on lags 2–4 (66.01% initially),
but this descriptive profile is not a causal attribution. Fixed bases give NLL
2.99617, supporting learned-basis utility within the family. Same-feature MLP
(2.42135) and strongest archived MLP (2.40775) remain better. Quantum continuations
remain fragmented, with one invalid UTF-8 sequence. Keep the new candidate
provisional and preserve the classical gap and long-context limitation. See the
[completed comparison](prior-position.md#completed-results).


## 45. Learned-position seed replication declared (5 November 2026)

Prepared six unchanged-recipe fits: full learned positions, matched-capacity fixed
pooling and refitted short context, each at seeds 22/33. Nine audited archived fits
complete three-seed comparisons with the prior and strongest MLP. No model source,
data statistics, partition, update budget or selection rule changes are introduced.

Require position NLL to beat both quantum references at each new seed and on the
three-seed mean. Keep older-context usefulness as a separate criterion, and report
all sampling/argmax tradeoffs. This remains validation reuse, not independent
confirmation. Input/reference checks, generated-tool typing/linting and reporting
checks pass. Research fits are unrun; see the [declared protocol](position-seeds.md).

Slurm 10906 subsequently passed all 130 DGX tests and six full-size CUDA learning
checks, then started `position_s22` with 90% observed GPU utilization. Inputs remain
pinned. Goal work pauses after verified startup while the experiment and completion
delivery continue; no new predictive result is available yet.


## 46. Position gain replicates; older-context and classical gaps remain (5 November 2026)

Slurm 10906 completed six fits in 2h09m despite an SSH observer failure. Recovery
verified submitted inputs, artifacts and locked reports without restarting training.
The independent audit reproduced 18 metric rows, 36 buckets, 60 intervals, five
seed summaries, both decision rules and all 12 saved sample renderings. No test
was scored. The current prod10 default is preserved; this job used its earlier
pinned prod80 submission.

Position beats both prior and matched-capacity pooling at each new seed and on
the three-seed mean, passing the declared rule. Mean NLL/sampling are
**2.60325 / 26.57%** position, **2.66776 / 25.51%** prior and **2.66729 / 25.64%**
capacity. Position minus prior mean NLL is −.06451, conditional article interval
[−.07600, −.05207]. Sampling improves by 1.066 percentage points.

The short refit averages **2.59486 NLL / 26.64% sampling**. Full minus short NLL
is +.00839, interval [−.00017, +.01714]; the older-context rule fails. Strong MLP
still leads at **2.41207 / 39.41%**, with non-space sampling **19.66%** versus
position **7.36%**. All six new position samples avoid invalid/control characters
but remain fragmented; the older invalid seed-11 sample remains evidence.

Retain learned positions as a seed-replicated quantum reference on reused
validation, not a demonstration of long-context benefit, coherent generation or
quantum advantage. Independent confirmation remains needed. See the
[completed comparison](position-seeds.md#completed-results).


## 47. Content-dependent controls implemented; MIG probe prepared (5 November 2026)

Added `prior_content` and `prior_content_mlp`. An ordered last-four-token query
scores older token keys and modifies the learned position weights. This is a small
classical selector, not quantum attention; the quantum output still comes from
Hamiltonian evolution and direct Born measurement without a vocabulary head.
Zero keys preserve initial pooling and initial spectra preserve the count prior.

All 101 CPU tests, Ruff and strict Pyright pass; 41 CUDA checks await the DGX.
Prepared a bounded prod10 probe of full, capacity, short, fixed-basis and MLP
configurations with full resident training data and unchanged batch 512. It measures
actual CUDA updates and memory before choosing resources for substantial fitting.
No validation/test scoring or predictive experiment has run. See the
[implementation and probe protocol](content-controller.md).


## 48. Content configurations fit on the default MIG slice (5 November 2026)

Slurm 10910 completed its bounded resource probe in 37 seconds on `prod10`.
All 142 tests and five real CUDA learning checks passed; retrieved artifacts,
hashes, accounting and timing calculations verify. Peak reserved allocator memory
was 0.955–1.018 GiB on the 10 GB MIG slice with all training rows resident.
Projected optimizer time is 22–50 minutes per fit, 2.44 hours across the five
configurations, excluding validation, final scoring and I/O. Keep `prod10` and
unchanged training budgets. No validation or test prediction was scored and no
scientific weights were retained. See the [measurements](content-controller.md#completed-resource-probe).


## 49. Content-dependent prediction comparison declared (5 November 2026)

Prepared five fresh seed-11 fits on the measured prod10 MIG slice: full content
quantum, matched-capacity position quantum, refitted short context, fixed bases
and same-feature cosine MLP. Preserve archived position and strong MLP references,
identical data/target partitions, batch 512, 30,720 updates and NLL selection.
Require a gain over both quantum references to retain the candidate; useful older
context remains a separate full/short criterion. No held-out test is scored.

Generated tools pass Ruff/strict Pyright and input/reference checks. Query/key
selection is classical preprocessing; output still uses Hamiltonian evolution and
Born measurement. This is reused-validation exploration, with seed and independent
confirmation still required. See the [declared comparison](content-screen.md).


Slurm 10911 subsequently passed all 142 tests and five full-size CUDA learning
gates on the prod10 MIG slice, then began the full-context content fit. Its Python
process occupied 1,144 MiB GPU memory and all pinned inputs remained intact.
Goal work pauses while the experiment and completion delivery continue; predictive
results are pending.


## 50. Content selection improves the single-seed score; short context still wins (9 November 2026)

Slurm 10911 completed five fits in 2h29m on prod10. The SSH observer failed,
but recovery verified all original inputs and artifacts without restarting training.
The independent audit reproduced all locks, tables, intervals, decisions and
18 saved sample renderings. Test remains unscored.

Full-content quantum NLL is **2.57984**, versus position **2.59512** and matched
capacity **2.59624**. Sampling rises **26.66% → 27.30%**. The declared candidate
rule passes, with a small, as-yet-unreplicated gain on reused validation.
The short refit scores **2.56048**: full minus short NLL +.01936, conditional
article interval [+.00388, +.03315]. Older-context usefulness is not demonstrated.

The strongest MLP retains **2.40775 NLL / 39.18% sampling**, versus quantum
**2.57984 / 27.30%**; non-space sampling is **19.30% versus 7.67%**. Learned
bases outperform fixed bases, without establishing quantum advantage. Full-content
samples avoid invalid/control characters but remain incoherent. Preserve these
limitations and replicate seeds before accepting a robust new quantum reference.
See the [audited comparison](content-screen.md#completed-results-audited-9-november-2026).


## 51. Content-gain seed replication declared (9 November 2026)

Prepared six unchanged-recipe fits: full content quantum, matched-capacity position
quantum and refitted short content, each at seeds 22/33. Nine pinned archived fits
complete three-seed comparisons with the position model and strongest MLP.
Require the content gain against both quantum references at each new seed and
on the mean. Useful older context remains a separate full/short criterion.

Keep data/partitions, batch 512, 30,720 updates and NLL selection unchanged; no
model-source changes or test scoring. Prior measured MIG fit times project 3h20m
for the six fits, so keep prod10 with a six-hour ceiling. Local setup and reporting
checks pass. This tests seed robustness on reused validation, not independent
confirmation. See the [declared replication](content-seeds.md).


## 52. Content gain replicates; refitted short context still performs better (9 November 2026)

Slurm 10980 completed six fits in 3h22m on prod10 despite an SSH observer failure.
Recovery verified all 230 original inputs, artifact hashes and independent report
audits without restarting training. All 142 DGX tests and six CUDA learning gates
passed. No held-out test was scored.

Content beats both position references at each new seed and on the three-seed
mean, passing the declared replication rule. Mean NLL/sampling are **2.57588 /
27.29%** content, **2.60325 / 26.57%** position and **2.60124 / 26.58%** matched
capacity. Mean content minus position NLL is −.02737, conditional article interval
[−.03741, −.01762]. Sampling improves by .715 percentage points.

The short refit averages **2.56249 / 27.48%**. Full minus short NLL is positive
at all three seeds, mean +.01339, interval [+.00360, +.02277]; the older-context
rule fails. Strong MLP retains **2.41207 / 39.41%**, with non-space sampling
**19.66% versus content 7.67%**. All six new content samples are valid text bytes
but remain incoherent. Retain the content recipe as a seed-replicated quantum
reference on reused validation, not independent confirmation, useful long context
or quantum advantage. See the [completed replication](content-seeds.md#completed-results-9-november-2026).


## 53. Locked full models use older content, despite losing to short refits (9 November 2026)

A bounded 16-second laptop diagnostic reproduced the three selected full-content
models' original CUDA scores and then changed only positions older than 16 tokens.
No weights changed and no training, selection or test scoring occurred. Independent
checks verify interventions, donors, all tables and 36 paired intervals.

Mean NLL **2.57588** worsens to **2.61147** when older context is masked,
**2.58712** when its valid positions are reversed, and **2.60210** when replaced
with same-length context from another article. All three changes worsen NLL at
every seed and their mean article intervals lie above zero. Non-space NLL worsens
too. The full model is sensitive to older tokens and their arrangement; an unused
older branch cannot explain these results.

The independently refitted short model still wins at **2.56249**. Context corruption
and removal measure sensitivity/co-adaptation under an input shift, not the benefit
of training with a longer window. They do not identify the cause of the remaining
generalization deficit or support quantum advantage. Investigate full-context
generalization with declared regularization or quantum-expressivity controls.
See the [completed diagnostic](content-diagnostics.md#completed-results).


## 54. Quantum depth versus width comparison declared (9 November 2026)

Prepared a three-stage content quantum comparison with a near-matched two-stage
rank-176 width control, independently refitted short context, fixed bases and
same-feature cosine MLP. Most added coordinates are in the Hamiltonians; no
classical vocabulary head or larger query/key selector is added. The extra RZ
controller is explicitly counted, so this is a depth/interleaving recipe test.

Keep data, updates, NLL selection, archived two-stage content/strong MLP references
and held-out-test exclusion fixed. Existing three-stage dense output/gradient checks
and generated setup checks pass. Before fitting on prod10, require all CUDA tests,
real learning and a bounded resource gate. Use the laptop `dgx start` notifier after
submission. Results are not yet available. See the [declared comparison](depth-screen.md).
