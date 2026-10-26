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
