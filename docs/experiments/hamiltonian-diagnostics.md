# Locked Hamiltonian and factor-scale diagnostics

13 November 2026. No fitting, optimizer updates, checkpoint selection, new sampling
or held-out-test scoring. The previous depth screen did not distinguish deeper
from wider quantum dynamics in NLL; this diagnostic looks for a concrete
parameterization issue before another model-capacity change.

## Scope and verification

Use selected seed-11 90-epoch checkpoints: smaller three-stage/rank-128 quantum,
four-stage/rank-128 quantum, wider three-stage/rank-164 quantum, and matched
prior-feature MLP. Replay all **4,096 existing validation contexts** and reproduce
archived per-target metrics to 1e-10. Check source, dataset, selection and weight
hashes; saved weights remain unchanged. The full NPZ is the unchanged expanded
pool with hash `37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`.
Only validation inputs/targets and training-derived count tables enter computation.

Quantum replay caches each learned QR basis, checks full first-batch probabilities
against the production forward pass to 1e-12, and checks normalization, active
mass and auxiliary purity bounds. Compute each stage's probability mass in its
active Hamiltonian subspace, state infidelity across evolution, auxiliary-sector
mass/purity, and the fraction of unwrapped energies outside ±pi. Also measure raw
factor singular values and the covariance of context-dependent spectral shifts.

This laptop computation took **8.04 seconds** with four threads and batch 256,
well within the declared 540-second guard. Generated tools/protocol and NPZ arrays
are in `output/hamiltonian-diagnostics/`. An independent audit recomputes all 50
telemetry distributions and four covariance participation ranks from saved arrays,
checks physical bounds and archive locks, and checks ten factor-scaling probes.
Ruff and strict Pyright pass for all three diagnostic tools. No source-model or
training changes were needed.

## Dynamics are being used

Mean active-subspace masses across stages are:

- Smaller quantum: **.768, .663, .557**.
- Four-stage quantum: **.694, .629, .535, .478**.
- Wider quantum: **.796, .731, .603**.

Stage mean infidelities range **.156–.250**. These are not nearly idle evolutions.
Final auxiliary-one mass averages **.104, .115 and .113** respectively, with
reduced-auxiliary purities **.839, .825 and .827**. The pure joint simulator states
therefore exhibit entanglement; this does not establish its predictive necessity.
Likewise, active mass and state changes measure use, not task usefulness or a
capacity bound. Deeper stages act on less active-subspace mass on average, but
this checkpoint observation alone does not identify a cause of the classical gap.

Contextual-shift covariance participation ranks, defined as
`trace(C)^2 / sum(C*C)`, are **41.76 / 128**, **45.66 / 128**, **56.75 / 164** and
**78.89 / 128** for smaller, deeper, wider and matched classical models. All
controllers vary across inputs. These ranks depend on input distribution and
coordinate scaling; the quantum/classical difference does not establish missing
information or a representational bottleneck. Raw QR factor condition numbers
range **2.14–3.02**, providing no evidence of near rank deficiency here. Only
about .12–.23% of stage/example energy coordinates lie outside ±pi; that statistic
is descriptive, since periodic phases make large magnitudes neither sufficient
nor necessary evidence of an optimization problem.

## Redundant factor scale has grown substantially

Each Hamiltonian uses `Q = qr(F)` and evolution through `Q diag(E) Q†`. Positive
scaling of individual columns of raw complex factors F preserves the represented
unitary (up to harmless QR column phases). Raw scale remains a choice of optimizer
coordinates, not an additional predictive degree of freedom.

Reconstruct the original initialized factors using each locked configuration/seed
and compare them with the selected checkpoint:

| Recipe | Initial mean column norms | Selected mean column norms | Median per-column growth across stages |
|---|---:|---:|---:|
| Smaller quantum | 1.410–1.415 | 24.24–24.58 | 17.17–17.48× |
| Four-stage quantum | 1.410–1.417 | 27.26–27.66 | 19.36–19.59× |
| Wider quantum | 1.411–1.417 | 28.81–29.18 | 20.36–20.63× |

A follow-up **2.46-second** check uses the first 32 validation rows. It rescales
only the in-memory raw factor columns to norm sqrt(2), computes forward outputs
and NLL gradients before/after, and performs **zero optimizer updates**. The
largest probability change is **7.78e-16**. Gradients obey the inverse coordinate
scaling relation to **1.17e-15** absolute error. Saved weights and selection locks
are unchanged. This demonstrates a concrete scaling symmetry and coordinate drift,
not a score improvement or failed physical dynamics.

The next controlled optimization test should constrain raw factor-column norms
while preserving the same quantum model class and capacity. Its exact timing,
normalization target, Adam-moment handling, clipping and rate schedules must be
declared before fitting. Default training must remain unchanged, and a meaningful
test must compare fresh fits at equal budgets against the locked unnormalized
reference. Adam's adaptive moments interact with rescaling, so these diagnostics
alone do not show that gradient scale causes slow learning or that normalization
will help. No new DGX experiment was launched in this diagnostic step.

Compact evidence: [stage telemetry](hamiltonian-diagnostics-telemetry.csv),
[controller covariance](hamiltonian-diagnostics-controllers.csv),
[factor conditioning](hamiltonian-diagnostics-factors.csv),
[scaling and gradient checks](hamiltonian-diagnostics-gauge.csv), and
[reproduced metrics](hamiltonian-diagnostics-validation.csv).
