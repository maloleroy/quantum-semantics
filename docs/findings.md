# Findings to keep

Updated 2026-10-23. This is the retained synthesis of the merged robust benchmark
and its plateau investigation. Historical report exports and one-off cohort
scripts have been removed from the working tree; exact recovery references are
below. These are existing measurements, with no new training in this cleanup.

Subsequent research is reported separately in the
[fixed-encoding/Born experiment](../experiments/hamiltonian_study/BORN_EVIDENCE.md)
(26 October 2026). It establishes learning in a Hamiltonian without a learned
classical encoder/head, on symbolic composition and UCI digits. It also derives a
likelihood separation from completely dephased models with the same readout.
This does not resolve the historical NLP plateau or establish superiority over
the strong classical digit controls.

The [natural-language follow-up](../experiments/hamiltonian_study/LANGUAGE_BORN.md)
then tested the same training/readout principle on real Tatoeba next-word targets:
24.21% trained-H accuracy versus 2.57% initially and 19.51% after retrained
dephasing; the matched MLP reaches 38.29%. This is new evidence of Hamiltonian-only
learning and a coherence contribution in this small language architecture. It is
separate from diagnosing the old fixed-graph/global-time plateau.

## What explains the jumps and plateaus?

**The experiments did not establish a single cause of the observed accuracy
plateau, and did not demonstrate an NLP barren plateau.** They did identify
readout restrictions, structurally inactive parameters, limited exposure and
task-composition confounds that matter for interpreting the curves.

First distinguish the plotted quantity. Top-1 accuracy is the fraction of correct
word predictions; NLL is the negative log probability of the correct word.
Accuracy can stay constant while probabilities and NLL improve: a prediction
changes only when the winning word changes. Hard bit-threshold decoding adds
another discontinuity: every target bit must cross to the correct side of 0.5
for an exact-word success. This explains how jumps followed by flat accuracy
can occur; it does not identify the cause of a particular historical jump.

| Finding | Evidence | What it establishes |
| --- | --- | --- |
| Training was still changing the circuit | Latest q8 controls had nonzero active-layer gradient RMS, minimum 0.0161; zero clipping. Between sampled epochs 50 and 100, monitor NLL improved 0.0325–0.0719 and top-5 improved 0.537–2.148 percentage points. | These runs were still improving on their fixed 2,048-target monitors. They are not full-validation convergence tests. |
| Initialization gradients did not collapse | 288 conditions / 9,216 circuit-gradient evaluations: two tasks, three batches, three encoder/head seeds, q2/4/8/12, depth 2/8, std 0.05/1 and 32 circuit seeds. | Nonzero gradients through q12. Each batch had only eight targets, heads changed size with q, and no trained checkpoints were probed. Later/larger-register plateaus remain possible. |
| Small variance did not mean a small mean | Causal q12/std 0.05 gradient RMS was 0.008517 at depth 2 and 0.008552 at depth 8; mean-gradient RMS was 0.008434/0.008260. | Near-common initializations can have low variance around a useful nonzero gradient. Variance alone is insufficient. |
| Some zero gradients have an exact cause | Final RZ/CRZ/ZZ gates commute with computational-basis measurement. The original 14-qubit, depth-2 QCSE probe excluded 27 such coordinates; its other layer RMS values were 0.0188/0.0356. | Structural unobservability, independent of a statistical barren plateau. XYZ or rotated readout changes which phases are visible. |
| The word decoder restricts the model | Independent-bit valid-word logits equal a constant plus the word's binary code dotted with marginal log-odds. | It is a fixed binary-code head. Marginals discard joint correlations; this is a proven restriction, not proof that it caused the plateau. |
| Exposure was smaller than the epoch label suggests | 100 × 20,000 replacement draws over 1,014,085 eligible targets: about 1.97 expected passes and 86.09% expected unique coverage. Seed replay gave 86.082%/86.107%. | One hundred sampled epochs are not one hundred corpus passes. Saved seeds were replayed; omitted DGX draw streams were not independently checked. |

The earlier toy global-observable probe showed decreasing variance (about
266-fold from q2 to q8 at depth 8/std 1), while local-observable variance stayed
similar. It was not an NLP loss. It must not be substituted for the actual NLP
gradient evidence above. All these simulations were exact statevectors without
hardware or shot noise.

The remaining diagnostic is focused: examine trained checkpoints immediately
before/after a jump and during a flat interval, using the same target set.
Compare NLL, top-1/top-5 and per-target rank/bit transitions, alongside circuit
gradient mean/RMS/variance and prediction concentration. Existing top-five
predictions support rank transitions; confidence, calibration and full MRR need
checkpoint inference. Control head dimensions and use optimizer-sized batches
before interpreting qubit scaling. This investigation has not yet been run.

## Robust benchmark results worth retaining

The final comparison used seeds 47/48 and the same 32,348 validation sentence
groups / 252,120 positive-position targets. L4 has four preceding slots; L2R2
has two preceding and two following slots. Each trainable arm used 100 × 20,000
draws, Adam batch 64, simulation microbatch 8, LR 0.003 and final epoch-100
weights. Circuit arms use q8/depth 2. Accuracy columns are percentages.

| Model | L4 NLL | L4 top-1 | L4 top-5 | L2R2 NLL | L2R2 top-1 | L2R2 top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-32 | 3.248730 | 35.6560 | 61.4632 | 2.206363 | 54.8931 | 77.4518 |
| BERT-32 | 3.330437 | 34.0838 | 59.5464 | 2.307967 | 51.7222 | 75.5529 |
| q8 CRZ | 3.466374 | 33.3163 | 58.1190 | 2.468397 | 50.2348 | 73.5967 |
| q8 identity | 3.470825 | 33.3567 | 57.9419 | 2.477890 | 50.0319 | 73.5261 |
| q8 local | 3.471120 | 33.1927 | 57.9474 | 2.481080 | 49.9931 | 73.4495 |
| q8 frozen CRZ | 3.474374 | 33.5174 | 57.8990 | 2.469915 | 50.1011 | 73.6336 |
| q8 XX | 3.465515 | 33.3782 | 58.1176 | 2.485275 | 50.0442 | 73.4787 |
| q8 ZZ | 3.468542 | 33.2594 | 58.3042 | 2.457248 | 50.3782 | 73.8728 |

MLP led both tasks and all three metrics in each seed. It had 368,880 parameters
versus 186,190 for CRZ; BERT had 385,394. This was a fixed-recipe comparison,
with different capacities and no equal tuning claim. It established no quantum
advantage. Left-input BERT still used full attention and an appended MASK;
it was not a causal GPT decoder.

The useful circuit findings are small and metric-dependent:

- Causal trainable CRZ versus frozen CRZ: NLL −0.00800, top-5 +0.2199pp,
  top-1 −0.2011pp. The top-1 interval included zero. Freezing only the circuit
  still leaves its encoder and head trainable.
- ZZ versus identity: top-5 +0.3623pp causal, +0.3467pp bidirectional. Nominal
  paired 95% intervals were [+0.2353, +0.4952] and [−0.0052, +0.6947]pp.
  Causal NLL changed sign across seeds and under sentence-macro weighting.
- CRZ and ZZ chains with free local RZ fields represent the same functions:
  `CRZ(c,t,g) = RZ_t(g/2) RZZ_ct(-g/2)`. The state/input-gradient/mapped-gradient
  checks had maximum error 3.55e−15. Their independent initializations and Adam
  coordinates differ; their ranking is not evidence of different expressivity.

Intervals used 10,000 paired seed/sentence bootstrap draws, with no correction
for selection or multiple comparisons. Two seeds and a reused validation fold
limit inference. The historical outer test had already been inspected.

Earlier native CBOW included first-word targets, unlike causal prediction.
Removing those targets only at evaluation left different training populations.
The final controls matched both training and evaluation targets. This is why
the final table supersedes earlier cross-direction comparisons.

The robust vocabulary used both complete curated sources before splitting:
202,172 deduplicated sentences and 10,864 words. That transductive convention
differs from the Hamiltonian study's training-only vocabulary. Historical
attention cosine retrieval, threshold-bit accuracy, paper half-bit similarity,
and word-softmax accuracy are distinct metrics.

## What carries into the Hamiltonian work

The current implementations are in
[the Hamiltonian study](../experiments/hamiltonian_study/REPORT.md):

- The alternating model uses a training-only, symmetrically normalized count
  graph H, square-root recency-count amplitudes, then layers of context phase
  `exp(-i gamma[l] diag(c))` followed by graph mixing `exp(-i beta[l] H)`.
  Its 16-layer version has 33 parameters including unigram-mixture smoothing.
- The later [matched causal comparison](../experiments/hamiltonian_study/CAUSAL_COMPARISON.md)
  freezes `exp(-i t H)` at validation-selected t=16 and optionally adds the
  existing 16-layer RX/RZ/CRZ QCSE ansatz. It uses the same marginal-bit word
  decoder in all three arms, without unigram mixing.

In the latter comparison, baseline QCSE / Hamiltonian alone / Hamiltonian plus
ansatz achieved top-1 **7.47% / 5.07% / 9.66%**, top-5 **22.18% / 15.59% /
26.42%**, and NLL **5.182640 / 5.347671 / 5.216199**. The combined model improved
ranking over baseline QCSE while worsening NLL. This directly illustrates why
all three metrics should be retained. Three seeds shared an already-used test
set; this was not a fresh confirmation.

In the main, differently decoded graph study, the walk reached NLL 4.2964 versus
QCSE 5.1826; matched classical diffusion reached 4.0350. A predictive quantum
residual tied a classical bigram/unigram mixture near NLL 2.9662: paired
difference +0.000014, 95% interval [−0.000851, +0.000880]. Graph structure and
vocabulary relabeling symmetry are useful design principles; quantum evolution
has not established an additional predictive benefit.

Keep the causal task, training-only graph statistics, explicit order/recency,
matched identity and classical controls, and separate marginal/joint-Born
readouts. The Hamiltonian study already includes joint-Born controls; their
existence does not settle the robust hybrid's different readout question.
Do not transfer numerical scores between these corpora, vocabularies and budgets.

## Directional Hamiltonian: preliminary result and training diagnosis

The role-doubled directional model has a one-seed screen, with 2,048 training
and validation targets, 16 layers and 15 epochs. Its best checkpoint was epoch
zero: validation NLL 2.932844, top-1 32.1777%, top-5 60.7910%. The classical
bigram on the same validation targets scored 2.933470 / 32.2266% / 60.7422%.
The directional model's mean target-role success was 0.00061855. These are
conditional validation scores, with no held-out confirmation of this variant.

There are concrete reasons to investigate model structure before interpreting
the absence of a training gain as a gradient plateau:

- The fixed transition table uses all 1,162,743 training transitions. The
  trainable correction sees only 2,048 targets and has 33 global scalars:
  16 walk times, 16 context-phase strengths and a unigram-mixture logit.
  It has no learned word embeddings or context-to-interaction network.
- For the pure short-time directional walk, target amplitudes are proportional
  to `t sqrt(P(next|last))`. Dividing by target-role success cancels the leading
  time factor, recovering P. Learning a time can predominantly change success
  probability while barely changing conditional predictions. Context phases
  perturb this result; the full model is not exactly time-invariant.
- Earlier context enters through a recency-weighted word-count diagonal.
  Only 7.7148% of validation targets occur in their context. The potential
  directly phases context words; indirect mixing can still affect other words,
  but this is a restricted path for learning new multiword prediction rules.
- Adam LR 0.03 is comparable to the initialized per-layer times (mean 0.02043).
  This motivates a smaller-LR check; it does not by itself prove overshooting.

A fresh mechanism check loaded the saved epoch-zero parameters on 32 fixed
validation examples. Halving all walk times changed mean target-role success
from 0.00061402 to 0.00015529, about fourfold. Mean total-variation distance
between the raw conditional prediction and the bigram was only 0.003756 at
the saved times and 0.001326 at half time; doubling time gave 0.015315.
Walk-time and context-phase gradient norms were nonzero on this subset.
These are local mechanism measurements, not full-validation quality estimates
or a new training run. They support a near-bigram, restricted-correction
hypothesis; they do not establish a single cause of the training outcome.

A proposed next model would retain the bigram as a baseline and learn a
correction from ordered context. Learned word embeddings would determine
weights of several Hermitian directional interaction generators, with a
normalized word readout using the available state rather than conditioning
on a rare role outcome. A residual word head can start at zero correction.
This changes what the model can learn; a held-out gain is a hypothesis.
Compare baseline-only, classical context correction and Hamiltonian context
correction using the same head, data and training budget. Merely learning a
neural correction would not establish an additional quantum benefit.

The subsequent [contextual proof of concept](../experiments/hamiltonian_study/CONTEXTUAL_POC.md)
implements a compact latent-state version of this idea. On the existing Tatoeba
validation split, three-seed mean NLL improves from 2.9493 to 2.8050, top-1 from
32.91% to 35.56%, and top-5 from 60.38% to 63.69%. It uses 32,768 training targets,
17,208 parameters and an eight-dimensional latent state. The matched classical
context model reaches 2.8288 NLL / 35.25% top-1; last-word-only training retains
the original bigram. All three start at that exact baseline. This demonstrates
a trainable context correction under the declared recipe; it does not establish
a quantum benefit or validate the original fixed vocabulary-graph model.

An equal-budget 30-fit tuning follow-up changes the initial comparison. Each
model received nine 60-epoch LR/decay recipes, then its top two were repeated
with fresh seeds 47/48/49. NLL selection chooses LR 0.01 / decay 0.01 for both.
Hamiltonian / classical means are **2.7940 / 2.7816 NLL**, **36.17% / 36.03%
top-1**, **63.53% / 63.88% top-5**. The classical control has the better primary
loss; accuracy remains close. Both prefer checkpoints well before epoch 60.
The best recipe hits the upper grid boundaries, so a global optimum is not
established. All results still use the reused validation subset, without test
scoring. The report records the full grid, selected epochs and audit scope.

## Evidence and recovery

The complete benchmark snapshot is commit
`fb71a4a1e96ff99d1c421fd7a0a1fce8481c8427`, merged by `36dd09a`.
Executed frozen training source was `483bea6`; copied extensions have their
own source hashes. Hamiltonian source/results were restored by `7aa6a35`.

| Historical evidence at fb71a4a | Outcome to retain |
| --- | --- |
| `results/robust-timed-cv/` | 90 short/early-stopped CV fits; not the later full-budget protocol. |
| `results/robust-screen-main/` | 156 records including parameter-free controls; deeper QCSE costly, identity competitive. |
| `results/robust-followup/` | 258 CV + 30 refits; MLP-32 led, circuit primary top-1 gains inconclusive. |
| `results/robust-followup-prediction-review/` | Corrected float64 reductions and exposed target-composition confounds; a reanalysis, not new fits. |
| `results/gradient-replicated-10324/` | 288 conditions / 9,216 initialization gradients; no demonstrated NLP barren plateau. |
| `results/robust-target-controls/` and `results/robust-causal-controls/` | 40 distinct fits, including 32 core records; the final paired table above. |
| `results/cluster-sweep/`, `results/attention-*`, `results/focused-*`, `docs/archive/` | Historical capped/short protocols and development logs; not additional comparable replications. |

Recover a report without restoring its clutter:

```bash
git show fb71a4a:results/robust-causal-controls/README.md
git show fb71a4a:results/gradient-replicated-10324/README.md
git show fb71a4a:docs/archive/TRAINABILITY.md
```

For old runners use a separate historical checkout. Fresh benchmark manifests
hash the retained source set; this cleanup must not bypass old source guards.
Original local ZIPs, prediction arrays, checkpoints and ignored outputs remain
on disk. Compact reports never contained all underlying tensors or draw streams.
The causal archive SHA-256 is
`1e2d55fd2cad3c07e4efcda739c5111192cfca88f9918c5a3f1b38edb432c1b1`;
the replicated-gradient archive SHA-256 is
`dbb4e08704d3b2fd1085b739766941dd62402c55f06865a7fcd53fde9e7403cd`.
