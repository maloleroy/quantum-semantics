# Locked content-model context interventions

Declared 9 November 2026, after seed replication and before this diagnostic.
The full 128-token model beats position baselines, but its independently refitted
16-token counterpart has lower NLL at all three seeds. Diagnose whether the full
model uses older context beneficially at inference before choosing another change.

Use the already selected full-content weights at seeds 11, 22 and 33. Keep the
existing 4,096 validation targets, training-only statistics/tokenizer, article IDs,
and weights fixed. No fitting, checkpoint selection, test scoring or new generation.
This is explicitly post-hoc reuse of validation, not independent confirmation.

Evaluate four input conditions with the last 16 context positions unchanged:

- **Intact:** reproduce the saved validation arrays before accepting results.
- **Mask older:** set the preceding 112 positions to padding.
- **Reverse older:** reverse only valid older token positions, retaining each row's
  older-token multiset and exact padding mask.
- **Other article:** replace valid older positions with tokens from a donor in a
  different validation article having the same number of valid older tokens.
  Choose donors using RNG seed 20261109 without consulting target labels. Leave
  rows with no valid older tokens or no eligible donor unchanged, and report coverage.

Donors come only from validation contexts; this deliberately breaks the causal
context relation and is a corruption diagnostic, never a deployable predictor or
an improved language-model score. All interventions can shift the input
distribution away from training. Loss changes measure sensitivity/co-adaptation;
they do not replace independent refits or identify a causal generalization mechanism.

Report per-seed and three-seed mean NLL, sampling success and argmax, target buckets,
and descriptive article-bootstrap intervals for intervention minus intact metrics.
Seed averages are averages of model scores, not ensemble probabilities. Archive
per-example score arrays and exact intervention contexts/donor indices as NPZ;
use CSV for reports and small JSON for execution metadata. Preserve checkpoint and
source hashes. No beneficial-effect threshold or new-model selection is implied.

Run on the laptop with four CPU threads, batch 512, and unchanged float64/complex128.
Twelve 4,096-row forward-only evaluations are expected to finish within ten minutes.
Generated tools and raw outputs live in `output/content-diagnostics/`; strict
Pyright and Ruff checks must pass before execution. No Slurm job or notification
watcher is required for this bounded local diagnostic.
