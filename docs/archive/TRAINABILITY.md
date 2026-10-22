# Barren plateaus: investigation and experiments

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


Research update: **2026-10-09**.

## Name and evidence

The term is **barren plateau**. In the setting studied by
[McClean et al. (2018)](https://arxiv.org/abs/1803.11173), sufficiently random
parameterized circuits have gradients concentrated near zero as register size
increases. Gradient variance over initializations can decrease exponentially
with qubit count under the paper's assumptions. A large optimization space,
more qubits, a flat accuracy curve or one small gradient is insufficient to
establish this mechanism.

[Cerezo et al. (2021)](https://arxiv.org/abs/2001.00550) distinguish cost locality:
for their alternating local-2-design ansatz, global observables can produce
exponentially small gradients at shallow depth, while local costs have at worst
polynomially vanishing gradients at logarithmic depth. These assumptions are not
proved for our encoder and nonlinear word loss. Single-qubit readout does not
automatically put the composed NLP objective under that local-cost guarantee.

[Grant et al. (2019)](https://quantum-journal.org/papers/q-2019-12-09-214/)
initialize shallow blocks to evaluate to identity, limiting effective initial
depth. Our Gaussian standard deviation 0.05 makes trainable rotations small;
it is not their inverse-block construction. We compare it against standard
deviation 1.0 without claiming a guaranteed mitigation.

[Wang et al.](https://arxiv.org/abs/2007.14384) identify a separate noise-induced
mechanism under specified local Pauli noise and depth scaling. Our exact
statevector experiments contain neither hardware noise nor shot sampling.
[Arrasmith et al. (2021)](https://quantum-journal.org/papers/q-2021-10-05-558/)
show that gradient-free optimization also suffers from cost concentration;
switching Adam to SPSA is not a general solution.

## Repository implications

Paper QCSE requires `ceil(log2(vocabulary_size))` qubits: **14** at the current
10,864-word vocabulary. Reducing it to four changes its vocabulary or encoding.
The learned hybrid has a latent register independent of vocabulary size and
provides the register-size comparison without changing the word task.

A distinct, exact cause of zero gradients is already present: final diagonal
RZ/CRZ/RZZ gates cannot change computational-basis probabilities. Final RZ before
CNOT also cannot, because CNOT permutes that basis. These zeros are structural,
not a statistical barren plateau. RZ before RXX can matter and remains eligible.
Earlier diagonal gates remain eligible. Hybrid XYZ readout retains phase
sensitivity, so its mask excludes none of these parameters. Other state
symmetries and causal cones can still yield zeros.

The old logger recorded the final optimizer batch's overall norm; large embedding
gradients could hide weak circuit gradients. Training now additionally records
online mean loss and circuit-only layer RMS, maximum absolute gradient and
fraction at absolute gradient ≤1e-10, averaged over all optimizer batches
**before clipping**. Known final diagonal zeros are excluded for QCSE's Z
readout. Counts and the threshold are saved. The threshold is a diagnostic
convention, not a plateau test. Logging needs no additional forward/backward pass.

## Optional NLP matrix

```bash
.venv/bin/python scripts/robust_benchmark.py list --trainability
.venv/bin/python scripts/robust_benchmark.py prepare --trainability \
  --output outputs/robust-trainability
```

The CRZ-chain controls complete this factorial comparison:

| Factor | Values |
| --- | --- |
| Hybrid latent qubits | 2, 4, 8 |
| Circuit layers | 2, 8 |
| Gaussian initialization standard deviation | 0.05, 1.0 |

Existing four-qubit small-initialization chain/depth arms supply two references;
**10 new configurations** supply the remainder. Four additional controls now
match identity learning rates and latent sizes (2/8 qubits). All use the same vocabulary,
contexts, word objective, grouped splits, draw/update budgets and validation-only
selection. Encoder/decoder capacity varies with register size, so performance
differences cannot be attributed exclusively to trainability.

The follow-up adds 14 further width/LR/q8/initialization controls. Standard:
**1,230 CV fits / 33 family refits**. Current exhaustive opt-in: **2,070 CV fits /
33 family refits**, with three arrays of **690** tasks capped at one job each
under the four-instance ceiling. Prefer the focused
[follow-up batch](BENCHMARK_FOLLOWUP.md), not an exhaustive rerun. Refits cap at four.
Prepare a fresh study: source-hash enforcement rejects changed source against
historical manifests. Inspect every qubit/depth/init candidate in `selection.json`
and its gradient traces, not just the winning hybrid.

The allocation preflight includes all **71** configurations. The profiler now
uses actual manifest arms; compute projection supports either matrix. Old CPU
estimates describe historical source and the standard matrix. Measure fresh costs
before launching the larger batch. Fresh full-matrix CPU timings and conditional
DGX budgets are now recorded in [BENCHMARK_COMPUTE.md](BENCHMARK_COMPUTE.md);
the completed six-arm CUDA timings are now audited in
[the DGX review](../../results/robust-timed-cv/README.md). Unseen variants still need
allocated-device calibration. Prefer [the staged screen](BENCHMARK_SCREEN.md)
over repeating this entire optional grid across fifteen fits.

## Separate mechanism probe

```bash
# One CPU thread; q=2/4/6/8, depth=2/8, std=0.05/1, 32 initializations.
.venv/bin/python scripts/probe_barren_plateaus.py

# Inside an actual CUDA allocation, extend the scaling study deliberately:
.venv/bin/python scripts/probe_barren_plateaus.py --device cuda \
  --qubits 2 4 6 8 10 12 14 --layers 2 8 32 --seeds 128
```

The probe uses fixed product inputs, taking the same prefix of a saved 16-qubit
angle array for every size. Initialization seeds vary the circuit only. Paired
costs are `(1 - <Z_0>)/2` and `1 - P(all zeros)` on identical circuits and inputs.
Float64 statevector arithmetic reduces numerical roundoff and uses no shots;
it does not reproduce an NLP loss or prove 2-design assumptions.

Unique folders retain runtime/source hashes and a verified source snapshot,
angles, seeds, losses and every
gradient coordinate. Summaries give unbiased initialization variance of the same
first RX parameter, mean eligible-coordinate variance, RMS and near-zero
fraction. Known dead coordinates remain in raw arrays but are excluded from
summaries; other causal-cone zeros remain.

Next: inspect log variance versus qubits separately by depth/init/cost, quantify
uncertainty using retained initializations and compare with NLP learning traces.
A short CPU run validates execution; it cannot prove asymptotic exponential
scaling, diagnose production QCSE or establish a mitigation. Larger scaling,
NLP training and allocated-device runtime/memory validation remain compute work.

## Local observations and verification

The initial one-thread CPU probe completed 512 circuit initializations, with
32 seeds per qubit/depth/init cell and paired costs, in 4.96 seconds. For depth
8 and initialization standard deviation 1.0, first-RX gradient variances were:

| Qubits | Local cost | Global cost |
| --- | ---: | ---: |
| 2 | 5.63e-3 | 2.30e-3 |
| 4 | 9.79e-3 | 6.45e-4 |
| 6 | 6.24e-3 | 2.36e-5 |
| 8 | 6.82e-3 | 8.63e-6 |

The global variance decreased about 266-fold across these endpoints while the
local variance stayed similar. This finite-size, fixed-input observation supports
keeping cost locality separate in the investigation; it does not establish
exponential scaling, the paper's ensemble assumptions or a QCSE barren plateau.
Raw arrays are retained at
`outputs/trainability-local/gradient-probe-20261009T005516Z-9c8g7szj/`.

A current 14-qubit, 10,864-word QCSE training microbatch also completed: eligible
layer-gradient RMS was 0.0188/0.0356, with 27 known dead parameters excluded and
55 remaining eligible. Thus even this large-register instance does not exhibit
universally tiny gradients. One microbatch cannot characterize an initialization
distribution or training trajectory. Evidence is retained in
`outputs/trainability-local/full-vocabulary-gradient-check.json`.

Verification: **127 passed / 46 CUDA/MPS skips**, 38.32 seconds; all **53** CPU
adapter preflights passed. Tests include diagonal-mask algebra, RXX sensitivity,
variable-qubit word heads, deterministic paired probes, saved variance
reconstruction, full new-arm CV/selection/refit/report execution and existing
resume/microbatch checks. Ruff, Pyright, shell syntax and diff checks pass.
After factoring shared source-snapshot saving, the focused final rerun passed
all **38** benchmark/trainability tests in **10.42 seconds**.
CUDA scaling and production NLP fits have not run.

The committed-source probe was rerun at
`outputs/trainability-verified/gradient-probe-20261009T010147Z-my46q9e0/`.
Its source/lock snapshot hashes were verified and every raw array matches the
initial run bit-for-bit. The full optional plan is prepared at
`outputs/robust-trainability-plan/qcse-bert-20261009T010146Z-vzr8545d/`.
It contains 1,530 unique tasks, the full pool/vocabulary and byte-identical data
artifacts to the independently audited standard plan. No scheduler job was
submitted. Prepare again in the frozen DGX checkout to capture its provenance.
Manifest fingerprint:
`676beeec9b250267f5b95276191796c684f7fba9d6e6495a2c4a9dd35746efbd`.
The submitter dry run verified two `0-764%5` arrays with offsets 0/765.

## Conversation handling

Exports stay under ignored `outputs/`. Explicit filename rules also protect
`conversation-export*`, `conversation-transcript*` and `CONVERSATION_EXPORT*`
outside that directory. Refreshed remote refs and fetched filename history contain
no tracked conversation-export/transcript files; no remote deletion or history
rewrite was necessary. Research notes and code are versioned; transcripts stay local.
