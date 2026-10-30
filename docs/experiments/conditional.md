# Learned context gates between Hamiltonian blocks

Update 30 October 2026: all six expanded fits have completed through the user-run
recovery. The [completed expanded results](#completed-expanded-results) below
supersede the historical interruption/handoff status. No recovery is needed.

29 October 2026. Validation-only follow-up to the [larger segmented study](segmented-large.md).
Static Hamiltonians and fixed input streams still trail the MLP. This experiment
lets context change the evolution itself, while retaining direct word-register
measurement and avoiding a learned classical vocabulary decoder.

## Model and controls

Prepare the existing normalized word amplitudes with decay .7, append a zero
auxiliary qubit, apply rank-64 joint Hamiltonian evolution, apply ten local RZ
gates, apply a second rank-64 joint evolution, and measure only the nine-qubit
word register. All measurement outcomes are retained. The two Hamiltonians use
262,272 real trainable coordinates, as in the earlier auxiliary reference.

Each RZ angle is the sum of three learned token-table pools: the last word,
preceding three words (fixed .7 decay), and remaining older words (fixed .97
decay). The latter two are normalized weighted means over valid positions.
Padding and absent streams contribute zero. The table is initialized with
independent Gaussian entries of standard deviation .1 and has 15,360 real
parameters at the present dimensions. The complete predictor has 277,632
trainable coordinates, approximately 94.5% in the quantum dynamics. Table
lookups and pooling are classical preparation/control costs, not free quantum
operations. There is no embedding-to-vocabulary classical head in this model.

For greater depth there is a separate angle table at each interleaving stage.
The current screen uses two Hamiltonian blocks and one stage only. The
context-dependent gates mean the evolution is no longer one shared unitary
for all inputs. This motivates the experiment but does not establish a
classical-computation or hardware advantage. Arbitrary amplitude preparation
and low-rank Hamiltonian compilation costs remain unresolved.

Independently trained controls:

- `ancilla`: the original static auxiliary model, with 262,272 parameters.
- `conditional_fixed`: identical initialized context gates held fixed while
  training the quantum core; 262,272 trainable parameters plus 15,360 stored
  random control entries.
- `conditional --freeze-core`: hold the random quantum core fixed and train
  only the 15,360 angle parameters. This tests dependence on trained quantum
  dynamics; its smaller trainable budget is explicit.
- `conditional_mlp`: the same amplitude coordinates and pooled-angle feature
  family, with independently learned angle tables and a tanh MLP replacing
  the quantum computation. Total 277,315 parameters. The angle inputs and
  available raw context information match; fitted tables need not match.

## Declared exploratory protocol

Use the existing `output/interleaved-screen/dataset.npz`: 8,192 fixed training
targets, all 4,096 validation targets, 512-entry vocabulary including UNK,
32-word contexts, official WikiText article partitions and train-only statistics.
No test scoring. Fit seed 11, 30 epochs, batch 512, NLL training and validation
NLL checkpoint selection, including epoch zero. Adam with clipping 5, float64/
complex128. Screen learned quantum angles at LR .001 and .003; fixed-gate and
static controls use .001, frozen-core and classical controls use .003. Recipe
selection among the two learned-angle LRs uses validation NLL and is adaptive.
This is not fresh held-out confirmation or equal hyperparameter search.

A six-update pilot measured .11460 s/update for the conditional quantum model
and .02339 s/update for the MLP after the first update. With 480 training updates,
this projects 55.0 and 11.2 seconds respectively before validation, checkpointing
and contention. Run six one-thread workers; expected full screen is under five
minutes, below the user's ten-minute boundary. Exact timings will be recorded.
The generated pilot/runner and artifacts live in `output/conditional-screen/`.

Scientific verification compares forward probabilities and gradients with
explicit dense context Hamiltonians, checks stream order/padding, and recovers
the static model at zero phase scale. Existing tests cover normalization and
portable weights. The test suite has 42 passing cases; strict Pyright and Ruff
are required before committing. These are implementation checks, not evidence
of prediction quality.

## Completed 8,192-target screen

All six fits completed in 29–108 seconds and passed source/data/weight locks and
prediction recomputation (1e-12 tolerances). Test remains unscored. Exact
[validation metrics](conditional-validation.csv) and
[paired differences](conditional-paired.csv) are versioned.

| Recipe | Validation NLL | Sampling | Known-target sampling | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| Learned gates, LR .001 | 3.7318 | 13.42% | 2.34% | 30 |
| Learned gates, LR .003 | 3.6796 | 13.54% | 2.75% | 30 |
| Fixed gates, LR .001 | 3.7750 | 13.81% | 2.12% | 30 |
| Frozen quantum core, LR .003 | 4.7796 | 10.13% | 1.40% | 30 |
| Static auxiliary, LR .001 | 3.7892 | 13.96% | 2.07% | 30 |
| Classical feature control, LR .003 | **3.4744** | **19.49%** | **3.29%** | 8 |

Choose LR .003 for the next learned-gate screen by declared validation NLL.
Relative to static, NLL improves .10959 and known-target sampling increases
.689 percentage points, but aggregate sampling decreases .422 points.
Against the MLP, NLL is still worse by .20515, descriptive paired article-group
interval [+.16828, +.24609]. The frozen-core result shows the recipe depends on
trained dynamics at these optimization budgets; it is not a quantum advantage
or a matched-parameter classical replacement test. All interval estimates
condition on this seed and reused validation set, with selection uncertainty
excluded. The learned .003 versus fixed/static comparisons also change LR.

Selected training NLL for the .003 conditional model is 2.5282 versus validation
3.6796, leaving a substantial generalization gap. More data and controls with
the same LR are justified before interpreting the gain as an architecture effect.

## Declared expanded screen

Before expanded scoring: reuse the existing 32,768-target dataset at
`output/dataset-selection/dataset-20261026T211719Z-c96984fa/dataset.npz`.
Verify that validation contexts/targets/groups, vocabulary and training count
statistics match the smaller screen. Run conditional, fixed-gate, frozen-core,
static auxiliary and MLP models at the same LR .003, seed 11, 30 epochs, batch
512, NLL loss and validation NLL selection. No further LR search.

Also independently retrain the conditional model with only the final four
context words: mask all older positions to padding in every partition while
preserving targets, article groups, vocabulary and statistics. This tests
access to older context under the same allocated architecture; unused older
angle entries remain allocated but receive zero gradient. It is different from
post-fit masking. It does not isolate word order or establish a quantum-specific
long-context benefit.

The six-worker small screen's slowest fit took 107.7 seconds. Multiplying its
entire duration by four gives a conservative 431-second projection for the
expanded fits (validation size is unchanged). Expected screen duration is
roughly 5–8 minutes, below the ten-minute handoff boundary. Generated preparation,
runner, masked data and artifacts live under `output/conditional-expanded/`.

## Historical expanded-screen interruption and recovery handoff

Status checked after the user's completion message: only the MLP has a locked
selection, saved weights and validation arrays. Its validation NLL is 3.220979,
sampling success 21.9190%, selected epoch 7. Reloaded predictions reproduce the
saved arrays and scalar metrics within 1e-12 tolerances; source/data/weight
checks pass. The five quantum fits have only run metadata and datasets, no
saved checkpoints, and no surviving workers. Their logs are empty. The cause
of termination is not established. Do not treat this as a completed comparison.

The user's execution preference now supersedes the earlier ten-minute rule:
provide training commands and stop, regardless of estimated duration. Do not
launch or monitor training on the user's behalf.

`output/conditional-expanded/recover.sh` restarts only missing learned, short,
static, fixed-gate and frozen-core fits in the existing `batch-M1UX59`. It preserves
the MLP, incomplete directories and original logs. Each restart has a fresh run
directory and log; no optimizer resume is available. A process check, lock and
source/data/script checksum manifest guard recovery. The preflight verifies
completed fit settings, locks and recomputed validation predictions. Re-running
the recovery command skips any additional completed fits. After all five workers
finish, it audits all six models and writes the expanded validation and paired
CSV tables. Worker failure prevents aggregation and preserves other results.

The five fits run concurrently with one Torch thread each. The estimate remains
5–8 minutes, based on the completed smaller screen, not a measurement of these
unfinished fits. Syntax, checksums and the completed-MLP preflight were checked;
the recovery training itself has not been launched by the assistant.

```bash
bash output/conditional-expanded/recover.sh
```


## Completed expanded results

Audited 30 October 2026. The user-run recovery completed the five missing fits
without replacing the original MLP. All source/data/script manifest checks pass.
The six selections have the declared seed, architecture, training budgets and
LR; the full/short dataset hashes match their preparation records. Reloaded
weights reproduce all saved validation arrays and scalar scores within 1e-12
tolerances. The six trained-model table rows and all twenty paired differences
and bootstrap intervals were also independently checked against the artifacts.
Test remains unscored. No training was launched or monitored during this audit.

| Recipe, all LR .003 | Validation NLL | Sampling | Known-target sampling | Argmax |
| --- | ---: | ---: | ---: | ---: |
| Learned gates, 32 words | 3.5518 | 14.36% | 2.71% | 36.38% |
| Learned gates, four words | 3.5274 | 14.84% | 2.99% | 35.35% |
| Fixed gates, 32 words | 3.6230 | 14.12% | 2.32% | 36.18% |
| Frozen quantum core, 32 words | 4.7747 | 10.17% | 1.40% | 16.50% |
| Static auxiliary, 32 words | 3.6360 | 14.09% | 2.24% | 35.86% |
| Classical feature control, 32 words | **3.2210** | **21.92%** | **5.49%** | **42.07%** |

See [exact fit records](conditional-expanded-validation.csv) and
[paired differences](conditional-expanded-paired.csv). These are one-seed,
validation-selected fits; they must not be compared as three-seed estimates or
fresh held-out confirmation. The vocabulary and targets match within this
expanded study. Its 32,768 training targets differ from the larger segmented
study's 131,072, so their rows are not a controlled architecture comparison.

With LR now held equal, learned gates improve NLL over static by .08425,
descriptive article-group 95% interval [−.11138, −.05951] for learned minus
static. Sampling improves .268 percentage points, interval [+.099, +.433],
and known-target sampling improves .461 points. Compared with fixed gates,
NLL improves .07125 and sampling .239 points. Thus the gain survives the
learning-rate control at this training budget. The angle table adds 15,360
trainable parameters over these controls; this is not an exact parameter-count
isolation. The frozen-core comparison supports dependence on trained dynamics
for this recipe, not the superiority of quantum computation over classical maps.

Older context does not show a benefit: 32-word minus four-word NLL is +.02441,
interval [−.00597, +.05140]. This interval includes zero, so the NLL result does
not establish a general degradation. However, 32-word sampling is lower by
.476 percentage points, interval [−.731, −.245], and known-target sampling is
lower by .283 points, interval [−.378, −.180]. Training NLL is lower with 32
words (2.8039 versus 2.8593) despite worse validation point estimates, consistent
with overfitting the additional input. This does not identify the sole cause
or prove that older context is intrinsically useless.

The MLP still leads: 32-word conditional minus MLP NLL is +.33079, interval
[+.29862, +.36280], and sampling is lower by 7.559 percentage points, interval
[−8.196, −6.948]. Known-target NLL and sampling also favor MLP. Even the stronger
four-word quantum variant does not close that gap. All-training bigram remains
NLL 3.0300/sampling 22.68%, but sees more unique training transitions than these
gradient models. The unigram's NLL 3.4851 and sampling 18.37% illustrate how
strongly aggregate scores are influenced by the 41.80% UNK rate; its known-target
sampling is only 1.91%. Known-target probabilities are never renormalized.

Recovered fit times were 252 seconds for frozen dynamics and 338–377 seconds
for the four jointly/core-trained quantum fits, with five one-thread workers.
The original MLP fit took 94 seconds under the original contention conditions.
The recovery ran within the estimate; these are simulator timings, not hardware
costs. The 42-test scientific suite, strict Pyright and Ruff passed for the
unchanged model source in commit `99a7bb3`; this audit changes results/docs only.

## Research decision

Retain learned gates as a better quantum reference at this optimization budget,
but do not scale the unchanged 32-word recipe merely on the basis of its static
control improvement. It has not demonstrated useful older-context computation,
competitive sampling or a classical advantage. All confidence intervals retain
single-seed and adaptive-validation limitations.

Meaningful text generation also remains limited by word-level UNK replacement
and preprocessing that discards punctuation, numbers and case. An open-vocabulary
representation is a distinct next experiment, not a numerical improvement to be
claimed by comparing its token NLL with these word scores. Any such change must
retain train-only fitting, official article partitions, frozen test selection,
and explicit scoring/tokenization units. Model controls should remain direct
quantum word/token measurement, retrained frozen/core replacements, and a
competitive classical predictor with the same information and exposure.
