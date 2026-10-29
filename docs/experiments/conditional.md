# Learned context gates between Hamiltonian blocks

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
