# Low-rank Hamiltonian screen

28 October 2026. Validation-only follow-up to the fixed-phase interleaving screen.
The previous turn made progress by identifying worse validation performance despite
better training fit; the next hypothesis is that reducing quantum capacity helps.

## Model and protocol

Each block uses `H = Q diag(e) Q†`, where Q has r orthonormal complex columns
obtained by reduced QR of a learned real/imaginary factor array, and e is a learned
real vector. The exact evolution is

`U = I + Q diag(exp(-i e) - 1) Q†`.

Applying this expression to states avoids constructing a dense V by V operator.
The orthogonal complement is unchanged. This is a restricted unitary family, not
an approximation to a trained dense Hamiltonian. At V=512, r=32, two blocks
have 2(2Vr+r)=65,600 real trainable coordinates, versus 524,288 for the two dense
blocks. Redundant factor coordinates count toward this total. Fixed input remains
the recency/position amplitude state. Fixed random context RZ gates between the
blocks use all 32 positions, with phase multiplier 1 or .25. `lowrank_static`
omits those gates and is separately retrained. No learned classical modules,
word prior, probability reconstruction, or postselection are used.

The representation is a simulator capacity/runtime experiment. Arbitrary learned
Q may still require expensive preparation/compilation on a device; low rank does
not by itself establish shallow circuits, local interactions or a quantum speedup.
The simulator remains classically tractable. QR differentiability requires
independent columns, as documented by
[PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html).
The implementation rejects a near-zero QR diagonal rather than silently using
undefined gradients. Dense-reference tests check the action and derivatives,
including all-zero energies, and norm preservation.

Declared before fits: ranks fixed at 32; phases {1, .25} and the static control;
seed 11, Adam LR .001, 30 epochs, batch 512, gradient clipping 5, NLL training and
validation-NLL checkpoint selection including epoch zero. Identical 8,192 gradient
examples and 4,096 validation targets from `output/interleaved-screen/dataset.npz`.
All vocabulary/count statistics remain training-only. Official article partitions
and 512-word vocabulary are unchanged. No test evaluation.

A stronger MLP reference uses the same data and 30 epochs at LR .003 (historically
useful in the Tatoeba study), 261,802 parameters; it is not parameter/information
matched to the new model. The preceding dense fits used only 12 epochs; any
comparison is a recipe comparison, not an isolated rank effect. All-training
bigram and unigram scores on these exact validation examples remain the count
references. This screen changes capacity, parameterization, initialization and
training duration, and also screens phase scale; conclusions must reflect that.

Six pilot Adam steps averaged .05086 s, projecting 24.4 s of updates per quantum
fit, excluding validation/I/O and concurrency. Three quantum workers and one MLP
worker each use one CPU thread. Files are under `output/lowrank-screen/`, including
the generated recipe, per-fit source/data fingerprints, locked selection, NPZ
weights/validation arrays and CSV histories. Generated tools/artifacts are ignored.

## Adaptive rank follow-up

The initial screen finished in 33–37 seconds per quantum fit. At rank 32,
validation NLL was 4.2036 for phase 1, 4.1323 for phase .25, and 4.1011 for static;
all selected epoch 30. The static recipe still trailed the earlier dense static
NLL of 4.0110. Before inspecting additional fits, selected a rank-128 follow-up
for phase .25 and static, keeping the same 30 epochs, LR, seed and subset. This
is adaptive validation exploration, not a preregistered confirmation; rank,
phase and checkpoint choices all consume validation information.

## Adaptive training-set-size follow-up

Rank 128 completed in 66–68 seconds: static NLL 3.9902 versus phase-.25 NLL
4.0269, with selected training NLL 2.9150 and 2.8790 respectively. This large
training/validation gap motivates using the full prepared 32,768-example subset.
Before fitting it, fixed rank 128, phase .25 versus static, and the same 30-epoch
budgets/LRs/seeds; MLP reference remains LR .003. Only the training subset expands;
validation, vocabulary, count statistics and official partitions are identical.
These fits have four times the example exposure per epoch and remain validation
exploration. Based on the completed timings, each quantum fit projects to less
than five minutes including validation. Test remains untouched.

## Results on 8,192 training examples

| Model | Parameters | Selected epoch | Validation NLL | One-shot success | Known-target success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank 32, phase 1 | 65,600 | 30 | 4.2036 | 10.39% | 1.37% |
| Rank 32, phase .25 | 65,600 | 30 | 4.1323 | 10.91% | 1.64% |
| Rank 32, static | 65,600 | 30 | 4.1011 | 10.93% | 1.72% |
| Rank 128, phase .25 | 262,400 | 28 | 4.0269 | 11.17% | 1.75% |
| Rank 128, static | 262,400 | 29 | 3.9902 | 11.34% | 1.93% |
| MLP, LR .003 | 261,802 | 8 | 3.4862 | 19.13% | 3.08% |
| All-training bigram (previous screen) | 0 optimized | 0 | 3.0300 | 22.68% | 5.94% |

UNK rate is 41.80%; known-word probabilities are not renormalized. Bigram retains
512² corpus-derived table entries despite having no optimizer parameters. MLP
still trails the all-training bigram. Training subset versus full-training count
exposure differs and must not be hidden by parameter comparisons.

[Exact measurements](lowrank-validation.csv) include argmax, known-word NLL,
selected training NLL, fit times and artifact paths. The rank-128 static recipe
improves slightly on the earlier dense static validation NLL (3.9902 vs 4.0110),
with half as many parameters and a 65.7 s fit rather than 172.9 s. The recipes
have different initialization, parameterization, epoch budgets and concurrent
workloads; this is not an isolated speed or rank-effect benchmark.

[Paired article-group bootstrap intervals](lowrank-paired.csv) compare phase .25
minus static. Rank-32 NLL difference +.0312 has 95% interval [−.0046, +.0660];
rank-128 difference +.0368 has interval [+.0124, +.0606]. Rank-128 one-shot
success difference is −.166 percentage points, interval [−.238, −.100]. These
are descriptive intervals on validation-selected checkpoints, conditional on one
seed and reused validation articles, not independent confirmation or a correction
for the adaptive search. There is no demonstrated benefit from these context phases.

Implementation verification: all 25 scientific/roundtrip/training tests pass,
strict Pyright has no errors or warnings, and Ruff passes. New tests compare
low-rank actions and derivatives to dense exponentials at random and zero
energies. Both model variants use the existing normalization and portable-weight
checks. No held-out tests have been scored for this screen.

## Results on 32,768 training examples

| Model | Selected epoch | Validation NLL | One-shot success | Known-target success | Argmax | Fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank 128, phase .25 | 30 | 3.8830 | 11.41% | 2.07% | 33.86% | 225.6 s |
| Rank 128, static | 15 | 3.8744 | 11.58% | 2.04% | 33.20% | 213.4 s |
| MLP, LR .003 | 7 | 3.3003 | 21.92% | 4.46% | 42.36% | 57.2 s |

A direct array comparison verified identical vocabulary, validation contexts,
targets/groups and training-only count statistics between the small and expanded
training datasets. The low-rank static NLL improves 3.9902 → 3.8744; phase-.25
improves 4.0269 → 3.8830. Thus more unique training examples improve this recipe,
though the experiment also increases update count fourfold. It does not isolate
unique-example coverage from optimization exposure. These are fresh Adam fits.

Phase minus static NLL is +.00864, descriptive article-group 95% interval
[−.01647, +.03540]. One-shot success difference is −.164 percentage points,
interval [−.221, −.103]. Phase has slightly better known-word sampling and
argmax, despite worse aggregate success and NLL. None of this establishes a
useful long-context advantage; checkpoint/recipe selection reused validation.

The improved quantum recipe still trails both classical references. Its
32-word amplitude preparation remains strongly recency-weighted, and fixed
context RZ gates have not established a predictive gain. The next architectural
hypothesis should address the restrictive shared-unitary word readout, for
example with an auxiliary register whose outcomes are marginalized in the word
measurement, with explicit preparation/gate costs and retrained controls.
This is a proposed experiment, not demonstrated performance. Generalization,
long-form generation, broad vocabulary coverage and hardware advantage remain
unresolved; these runs do not complete the research objective.

To rerun the expanded static recipe independently with the installed CLI:

```bash
uv run qcse train \
  --dataset output/dataset-selection/dataset-20261026T211719Z-c96984fa/dataset.npz \
  --models lowrank_static --rank 128 --layers 2 --seeds 11 \
  --epochs 30 --batch-size 512 --lr .001 --loss nll --selection nll --threads 1 \
  --output output/lowrank-reproduction
```

For the phase variant use `--models lowrank --phase-scale .25`. The CLI locks
selected weights but does not score test. No long job was delegated to the user:
all measured fits completed within ten minutes.
