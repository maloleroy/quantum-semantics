# Context-dependent phase interleaving screen

28 October 2026. Exploratory validation-only experiment on WikiText-2 Raw.
No held-out test is scored. This screen is not a confirmation comparison.

## Hypothesis and declared protocol

A shared unitary acting on recency-weighted amplitudes cannot change their input
fidelities and gives distant words little influence. Insert fixed local RZ gates
between two independently learned dense Hamiltonian evolutions:

`amplitudes(context) -> exp(-i H0) -> RZ(context) -> exp(-i H1) -> word measurement`.

For each qubit, sum fixed Gaussian token features multiplied by independent fixed
position signs, then divide by the square root of the number of valid tokens.
Every position contributes at equal scale; padding contributes zero. Random
features use seed 20261028 independently of corpus statistics. Token features
have shape V by log2(V); position signs have shape (layers-1) by W by log2(V).
The fixed tables are included in checkpoints and count toward memory costs.
There are no learned classical encoder or decoder parameters and no count prior.
Local RZ operations alone do not create entanglement; the dense evolutions may.
No entanglement advantage is implied.

This tests an input-dependent extension of the shared-unitary family, inspired
by [data re-uploading](https://arxiv.org/abs/1907.02085). The paper does not
establish that these particular random language features are useful.

`interleaved_static` is independently retrained with the same number of dense
Hamiltonians and no intervening RZ. It tests whether the extra context signal
helps beyond the same optimization parameterization. Its two unitaries still
compose into one unitary. The baseline MLP has fewer parameters and only sees
recency amplitudes; it is a competitive reference, not an information-matched
or parameter-matched attribution control for the new features.

Declared before training: NLL loss and validation NLL checkpoint selection,
including epoch zero; Adam LR .001, clipping 5, 12 epochs, batches of 512,
seed 11, two evolutions, float64/complex128, one Torch thread per fit.
8,192 training examples are sampled without replacement using NumPy seed
20261028 from the existing 32,768-example screening dataset. Vocabulary and
count statistics still use all training articles. All models see the identical
subset and all 4,096 existing validation examples. Count baselines use the
full-training statistics. The official article partitions are preserved.
512 vocabulary entries including UNK, 32 preceding tokens, nine qubits.
Each quantum model has 524,288 trainable parameters. No hyperparameter search
or held-out comparison is part of this screen.

Generated recipe, dataset, locked selections, weights and validation arrays live
under `output/interleaved-screen/`. The run records fingerprint dataset and model
source; fit histories are CSV, arrays/weights NPZ. These files are ignored and
must be explicitly transferred for handoff. Source dataset:
`output/dataset-selection/dataset-20261026T211719Z-c96984fa/dataset.npz`.

Pilot: four Adam steps averaged 0.5222 s/step, projecting 100.3 seconds of updates
per fit. Validation, checkpointing, contention and thermal effects add overhead.
Two quantum fits run concurrently, with the smaller classical references on a
third worker. This is CPU statevector simulation, not device training. Dense
Hamiltonian compilation and arbitrary amplitude preparation remain scalability
obstacles; the new local phase gates do not remove those costs.

## Completed results

| Predictor | Selected epoch | Validation NLL | One-shot success | Known-target success | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interleaved | 12 | 4.2596 | 10.40% | 0.96% | 35.28% |
| Static control | 11 | 4.0110 | 11.25% | 1.79% | 35.52% |
| MLP | 12 | 3.4969 | 18.34% | 2.30% | 41.80% |
| All-training bigram | 0 | 3.0300 | 22.68% | 5.94% | 43.16% |
| Unigram | 0 | 3.4851 | 18.37% | 1.91% | 41.80% |

[Exact scores, fit paths and timings](interleaving-validation.csv).
Known-target probabilities are not renormalized. UNK accounts for 41.80% of
validation targets. MLP has 261,802 trainable parameters and predicts UNK for
every validation argmax here; it is undertrained, not a strong neural reference.
Count models have no trained optimizer parameters but store corpus-derived
probability tables (512² bigram entries), which are not free information.

Interleaving minus static control: NLL +0.2486, article-group bootstrap 95%
interval [+0.2018, +0.2924]; success −0.849 percentage points, interval
[−1.014, −0.652]. Known-target success difference is −0.825 percentage points.
[Paired descriptive intervals](interleaving-paired.csv) condition on this single
seed, these selected fits and the validation partition used for selection.
They are not independent confirmation or model-selection-adjusted intervals.

Selected training NLL is 2.7919 for interleaving versus 2.9297 for static;
validation is worse despite the improved training fit. That pattern is
consistent with overfitting the random contextual features at this sample size.
It does not prove a convergence limit or identify the sole source of failure.
Do not promote this fixed-phase recipe to a larger confirmation run.

A post-fit [older-order perturbation](interleaving-context.csv) restricts scoring
to 4,046 full windows, shuffles positions older than the last four within each
window, and preserves the last four words. Interleaving NLL changes
4.2566 → 4.2424; static NLL changes 4.0070 → 3.9799. Neither deterioration nor
a useful distant-order benefit is observed on NLL. This is sensitivity only:
no shuffled-context retraining or confidence interval was performed.

Measured fit times: interleaving 181.6 s, static 172.9 s, MLP 7.2 s, counts
under 0.1 s each, with concurrent execution. Test remains unscored. Full checks:
21 pytest tests pass; strict Pyright and Ruff pass. A scientific check compares
the phase layer to explicit per-qubit RZ gates and checks both Hamiltonian
blocks' gradients, including a finite difference. Existing normalization and
portable checkpoint tests cover both new model kinds.

Next research decision: investigate lower-capacity structured or shared
Hamiltonians and controllable phase strength before increasing data/compute.
Long-context phase access alone has not improved generalization. A competitive
classical context baseline, independently retrained context ablations, and a
fresh confirmation design are still needed before any advantage claim.
