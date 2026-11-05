# Learned position weights for quantum count refinement

5 November 2026. Declared before fitting; this is an exploratory validation study.

## Hypothesis and prediction path

The replicated prior model improves NLL but still trails the MLP, especially on
non-space targets, and its 128-token context has not established an advantage over
16 tokens. Its context features all use the same fixed .7 recent/.97 older decay.
That retains some order, but cannot independently select different lags for
different Hamiltonian eigenvalues or local rotation angles. Test whether allowing
each feature to learn its position weights improves conditional token prediction.
This is a hypothesis about a representation restriction, not a diagnosed cause.

`prior_position` retains the fixed training-bigram amplitudes, two context-dependent
rank-128 Hamiltonians on a 512-outcome word register and one auxiliary qubit, local
RZ interleaving, and direct Born token measurement. It adds no classical vocabulary
head. For each stage/feature and each of the recent/older streams, normalize
`decay**age * exp(position_logit)` over valid context positions, then pool the same
learned token table with these weights. The last token remains separate. Logits
start at zero, reproducing the old controller; zero initial Hamiltonian energies
and shifts still reproduce the supplied count distribution. Padding contributes
zero even for an empty stream. The position logits are independent of token content;
this is a learned classical linear controller, not quantum attention.

The new controller has 17,526 extra coordinates (127 positions × (128 spectral
features + 10 rotation features)), 2.38% above the prior model's 736,512. All fixed
count-cache costs remain: 262,656 float64 entries, with arbitrary amplitude
preparation and general learned bases. No hardware speedup is claimed. Softmax over
position weights is classical preprocessing; it is not a replacement for the final
Born output distribution. There is one word measurement, no retries or rejection.

## Fixed comparison

| Label | Recipe | Rank | Trainable parameters | Rate |
|---|---|---:|---:|---|
| quantum | Learned position weights, full 128-token context | 128 | 754,038 | Constant .003 |
| pooled_capacity | Original fixed pooling, additional Hamiltonian capacity | 131 | 753,414 | Constant .003 |
| short | Learned position weights, independently refit with only last 16 tokens | 128 | 754,038 | Constant .003 |
| fixed_basis | Learned position weights, fixed quantum bases and zero base energies | 128 | 229,494 | Constant .003 |
| mlp_cosine | Same prior and learned position features, classical logit correction | 128 | 753,356 | Cosine .003→.0003 |

The capacity control differs by only 624 parameters (0.083%). The short control
retains the parameter shapes but older-position weights receive no signal because
the older inputs are padded. Fixed bases still allow contextual spectra to train:
this control does **not** freeze the entire Hamiltonian. The MLP replaces the
quantum dynamics with a near-matched classical map and starts at the same prior.
Keep the stronger archived cosine adaptive MLP (2.407749 NLL at seed 11) visible;
beating only the new feature-matched control cannot establish classical superiority.
Also retain archived seed-11 prior (2.667615) with its own source and weight hashes.

All five fits use seed 11, the unchanged 524,288 training targets, 30 epochs,
batch 512 (30,720 optimizer updates), float64/complex128, and the same Adam/clipping
implementation. Train on NLL and select minimum validation NLL, breaking ties by
sampling success; include epoch zero and all 30 epochs. No early stopping or
changes based on emerging scores. Identical 4,096 validation targets and 8,192 test
targets, training-only byte BPE/counts, article partitions and UNK policy remain.
No held-out test scoring. The short dataset masks positions before the last 16;
it changes neither target selection nor statistics.

Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
All 60 official validation articles have already been used for selection. This is
adaptive exploration, not independent confirmation; bootstrap intervals condition
on these fits and do not correct repeated model selection or seed uncertainty.

## Decision and reporting

Retain the new recipe as a candidate only if its selected aggregate NLL beats both
the archived prior and the fresh capacity control. Report all results even if this
fails. Attribute a useful older-context gain only if full minus refitted-short NLL
has a descriptive article interval entirely below zero and non-space NLL also
improves. A gain over fixed bases supports learned-basis utility in this family;
it does not establish quantum advantage or an entanglement-specific contribution.

Report NLL, sampling success S, argmax accuracy, sampled-target bits per byte,
space/non-space buckets, selected epochs, fit times, parameter counts, and paired
article intervals. Preserve count/modal baselines. Generate three fixed prompts
with seeds 20261030–20261032, 64 direct draws each, for full-context fits and count
baselines; retain every invalid/control outcome. Do not generate the short refit
through the generic full-window generation path. Existing reference samples are
not resampled. All selected weights must reproduce saved predictions before reports
are accepted. Learned lag weights are diagnostic, not proof that older context helps.

## Execution and validation

Generated scripts, source snapshots, references and input manifest live in
`output/prior-position/`. Use isolated DGX checkout
`~/quantum-semantics-prior-position-20261105`. One full A100 allocation and four CPU
support threads; estimate 2–4 hours, with a six-hour limit. This estimate is unmeasured
for the new pooling implementation. Require the full CUDA test suite and actual
full-size CUDA forward, gradient and optimizer checks for all five configurations
before fitting. There is no CPU training fallback.

Local validation passes 91 tests; 39 CUDA checks await DGX execution. Independent
scalar pooling and dense Hamiltonian comparisons check probabilities and gradients,
zero-logit compatibility, padding, learning away from identity, fixed bases,
checkpoint serialization and locked training. Ruff and strict Pyright pass.

Submit once with the long-jobs controller; network-observation failure does not
authorize resubmission. After verified GPU startup, pause goal work as requested
while the experiment and completion delivery continue. No predictive result is yet
available.
