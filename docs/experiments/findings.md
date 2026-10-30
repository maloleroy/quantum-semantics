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
references and fixed-prompt sample diagnostics. Full-corpus tokenizer fitting
and training are **unrun**. No test was scored. See the [declared protocol,
scoring limits and command](byte-bpe.md). Training remains manual.
