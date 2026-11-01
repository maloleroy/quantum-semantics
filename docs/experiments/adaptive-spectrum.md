# Context-dependent Hamiltonian eigenvalues

1 November 2026. Implemented and checked; training is unrun. The expanded-data
product reaches 2.7870 NLL while its equally exposed MLP reaches 2.4941. More
coverage helps but does not close the gap. This screen tests a different placement
of context dependence, with explicit parameter and classical controls.

## Prediction path

`adaptive` retains the product model's learned local RY preparation and local RZ
interleaving. Each Hamiltonian now has context-dependent eigenvalues:

\[
H_l(C)=Q_l\,\operatorname{diag}(e_l+s(C))\,Q_l^\dagger,
\qquad U_l(C)=\exp[-iH_l(C)].
\]

The columns of each learned complex Q are orthonormal. The rank-128 spectral
subspaces and base energies are separate for each of the two stages. A single
shared table of shape (1, 3, 512, 128) produces 128 real shifts by the existing
linear last-token/recent/older weighted pooling. The shifts are shared across
stages, but act in different learned bases. `--phase-scale` scales these shifts
and the existing input gates. Empty pools contribute zero.

The shift table starts at zero, so with the same seed the initial quantum
predictions exactly match the unrotated rank-128 product. The table receives
nonzero gradients at initialization. Input preparation and readout are unchanged;
there is no nonlinear classical encoder, classical word head, temperature,
postselection or rejected outcome. Final measurement uses the nine token bits and
ignores the auxiliary bit. Each H(C) is Hermitian and each evolution is unitary.

This adds a trainable linear context-to-phase map (196,608 parameters). It changes
both accessible context features and how they enter dynamics. It is not proof of
an exclusively quantum benefit; the matched MLP receives those same features.
Arbitrary learned spectral bases still lack an efficient hardware compilation
claim. Simulator gradients and exact probabilities are not free on a device.

## Four fits

| Fit | Model | Trainable parameters |
|---|---|---:|
| adaptive | Rank-128 context-dependent spectrum | 751,892 |
| static | Ordinary product with rank 176 | 751,988 |
| fixed_basis | Adaptive model with fixed bases and base energies | 227,348 |
| mlp | Matched adaptive-input MLP | 750,460 |

The rank-176 static model allocates almost the same total parameters to larger
Hamiltonian subspaces instead of spectral input tables. It tests capacity allocation,
not an isolated same-rank phase effect. The audited rank-128 product remains a
separate reference. The MLP independently learns the same preparation, local-control
and spectral-control feature families, then maps 1,162 features through 312 tanh
hidden units to token probabilities. Its network budget matches the learned
quantum-core budget to rounding.

For `adaptive --freeze-core`, Q and base energies are fixed at initialization,
but **spectral shifts remain trainable** along with the original input controls.
Thus H(C) still changes during training; this is a fixed-basis/base-spectrum
control, not a frozen Hamiltonian. A benefit over it would support learning the
spectral bases/base energies within the recipe, not quantum necessity or superiority.

## Locked data, training and reporting

Use exactly `output/product-coverage/batch-RofZOb/data/full.npz`:
524,288 unique training positions, 4,096 validation targets, 512-outcome train-only
byte BPE, 128-token contexts and the original article groups. Training counts,
byte lengths, UNK policy, validation and test are unchanged. No test is scored.

All fits start fresh with seed 11, two stages, auxiliary size 2, recent window 16,
phase scale 1, 30 epochs, batch size 512, Adam learning rate .003, NLL loss and
NLL checkpoint selection. Amplitude decay .9 is recorded but unused. There are
30,720 optimizer updates and the same checkpoint schedule for every fit.
Four concurrent fits use two Torch threads each on the eight-physical-core laptop.

Archive source for the completed expanded product and MLP, verify their data and
weight locks and reuse their already reproduced validation arrays. Current source
checks are not bypassed to load old weights. The stronger expanded MLP is retained
even if a new matched MLP underperforms it.

Report nine metric rows, 18 modal/non-modal target buckets and 23 paired intervals:
adaptive versus each of the other eight fits/references/count/modal baselines,
for NLL, sampling and argmax, excluding infinite-NLL inference against the modal
predictor. Intervals are descriptive article-bootstrap summaries for one seed and
reused validation; they exclude selection uncertainty.

Save 18 unfiltered 64-token continuations for the four new models and bigram/unigram,
using the same three fixed prompts and seeds 20261030–20261032 as earlier screens.
Store raw token IDs in NPZ, text/tables in CSV and only small metadata in JSON.
No new reference samples or held-out test scoring. NLL, non-space prediction,
invalid outcomes and text quality must accompany S; a score gain alone is not
meaningful text generation. Long-context benefit and practical quantum advantage
remain unproven and would need additional controlled confirmation.

## Manual command

```bash
bash output/adaptive-spectrum/launch.sh
```

Planning estimate: **2–5 hours**, unmeasured for these models. Larger static QR
factorizations and spectral feature pooling may add cost; the preceding rank-128
product fit took 100.9 minutes. The controller records actual fit times. The
codex-long-jobs launcher uses event-only delivery and no viewer; the agent has not
started training. It checks source, scripts, dependencies, data and references
before and after fitting and preserves completed fits on retry:

```bash
bash output/adaptive-spectrum/launch.sh output/adaptive-spectrum/batch-XXXXXX
```

The final marker is `ADAPTIVE_SPECTRUM_COMPLETE`, printed after verification and
reporting. All generated tools, snapshots and fit artifacts live under ignored
`output/adaptive-spectrum/`. All 71 tests, Ruff and strict Pyright pass. Tests
compare batched evolution and gradients against explicit dense exponentials,
check normalization, zero-shift equivalence, padding, checkpoint restoration,
context removal, trainable-parameter budgets and fixed-basis gradients. Generated
tool imports, shell syntax and archived-reference checks also pass. No predictive
gain has yet been measured for this architecture.
