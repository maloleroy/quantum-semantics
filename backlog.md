# Quantum semantics: current work

Updated 2026-10-26.

The project now centers on **Hamiltonian encoders for causal word prediction**.
Start with the [Hamiltonian report](experiments/hamiltonian_study/REPORT.md),
[matched encoder/ansatz comparison](experiments/hamiltonian_study/CAUSAL_COMPARISON.md),
and [retained benchmark findings](docs/findings.md).

The robust branch was merged at `36dd09a`; its complete snapshot is `fb71a4a`.
Its historical report exports, duplicated notes and one-off cohort tools have
been retired from the working tree. Their conclusions and exact Git recovery
commands are in the findings document. The Hamiltonian study, reusable model
controls and working simulation/training pipelines remain available.

## What we know

- Accuracy jumps/flat intervals do not by themselves diagnose an optimization
  plateau. Actual NLP gradients remained active; the cause of the historical
  plateau has not been established.
- Independent-bit readout restricts the word model, final diagonal gates can be
  unobservable, and 100 sampled epochs represented about two expected passes.
- MLP led the latest robust comparison. Small circuit improvements depended on
  the endpoint; there is no demonstrated quantum advantage.
- The graph model addresses vocabulary relabeling symmetry. Its classical graph
  and calibrated bigram controls remain essential: quantum evolution has not
  established an additional predictive benefit.
- The new [contextual Hamiltonian POC](experiments/hamiltonian_study/CONTEXTUAL_POC.md)
  learns on real Tatoeba data: mean validation NLL 2.9493→2.8050 and top-1
  32.91%→35.56% over three seeds. The matched classical model also improves;
  last-word-only training keeps epoch zero. This supports context correction,
  with no fresh held-out-test or quantum-advantage claim.
- Matched tuning completed 30 fits in 4m11s on all 16 logical CPUs. Both models
  prefer LR 0.01 / decay 0.01 within the grid. Fresh-seed mean NLL favors
  classical (2.7816 vs 2.7940); accuracy is nearly tied (36.03% vs 36.17%).
  Selected epochs are 13–17 classical / 9–12 Hamiltonian. The upper grid
  boundaries remain open; these are best tested settings.
- The [fixed-encoding/Born experiment](experiments/hamiltonian_study/BORN_EVIDENCE.md)
  establishes a gain from Hamiltonian-only training: five fresh seeds reach 100%
  on unseen symbolic compositions and 91.96% on official held-out UCI digits.
  No learned classical encoder/controller/head is present. Digit NLL 1.0475 is
  below the 1.3301 lower bound for every completely dephased model with this
  readout. A similarly sized classical MLP reaches 95.43%, so a useful coherence
  contribution is established within this architecture, with no general quantum
  advantage or language result. All 80 fits finished in 27.664s on eight CPU workers.
- The [natural-language follow-up](experiments/hamiltonian_study/LANGUAGE_BORN.md)
  confirms Hamiltonian-only learning on a fresh Tatoeba split: test NLL
  6.5861→3.5974 and word accuracy 2.57%→24.21%, with eight qubits / 256 words.
  Retrained dephasing reaches 3.6683 / 19.51%; the coherent-minus-dephased
  sentence-group interval is [-0.0830, -0.0594] NLL. Known-word accuracy reaches
  22.49%, so the gain extends beyond UNK. Classical MLP (38.29%) and subset-only
  bigram (34.09%) remain stronger. All 24 fits plus scoring/audit took 4m20s.
- Operational metric correction: the preceding word accuracies are argmax scores.
  For one preparation/evolution/measurement, ideal success is **6.76%** for H
  (1.91% initially), 5.71% after dephasing and 24.77% for MLP sampling. Classical
  argmax remains a valid, stronger word-guessing baseline. New studies must specify
  actual readout/shot cost and select on the declared device-level endpoint.

## Next work

- [x] Establish a Hamiltonian-only training gain with fixed encoding and quantum
  readout, frozen initialization, separately retrained dephasing and classical
  controls. Completed on a symbolic task with an exact coherence separation and
  on real digit data; the post-hoc digit bound rules out insufficient dephased
  training as the explanation of the likelihood gap.
- [x] Test the design on vocabulary-space causal word prediction with fixed
  ordered context loading and direct Born readout. The new dense-H experiment
  establishes training/coherence gains on language with matched controls.
- [ ] Improve the language model against the matched MLP and subset bigram.
  H selected epoch 30/30/30 and the lowest tested LR; convergence and optimal
  settings remain open. Consider structured/input-dependent couplings and
  confirm on independent splits before investing in a large run. Current dense-H
  timings do not validate near-20-qubit scaling. The latent hybrid remains a
  separate experiment with unresolved component attribution.

- [x] Profile larger contextual shapes on CPU for the user's 24-physical-core /
  10-GB planning limit. The [scale notes](docs/optimizations.md) recommend a
  4–5-qubit, approximately 0.5–2.4M-parameter model and WikiText-2 / 1–3M tokens
  as a possible latent-hybrid scale only. Short profiles do not establish a
  vocabulary-space model ETA or justify scaling before contribution is isolated.
- [ ] Stream metrics and context batches, bound worker/bigram memory and add
  train-only subword preparation before million-token experiments. The current
  POC's accumulated vocabulary predictions exceed the proposed RAM limit.

- [ ] Check the directional model with a smaller LR and more eligible training
  targets: its fixed bigram uses 1.16M transitions but its 33-scalar correction
  was fitted on 2,048 targets. The short-time normalization check supports a
  restricted near-bigram hypothesis, not a demonstrated optimization plateau.
- [x] Prototype an ordered-context correction with matched classical and
  Hamiltonian feature maps, retaining the bigram baseline and avoiding a rare
  role-conditioned readout. Ran nine real-data and nine controlled-task fits.
- [x] Run equal-budget LR/regularization tuning through 60 epochs, then confirm
  the top two recipes per model with three fresh seeds. This removes the
  initial Hamiltonian NLL lead; both context models retain gains over bigram.
- [ ] Compare a frozen Hamiltonian controller and stronger classical feature
  maps, then prespecify fresh confirmation before attributing the gain to a
  particular component. The current classical model may benefit from more training.

- [ ] Identify the exact run/curve with jumps and flat accuracy. Compare fixed-set
  NLL, word ranks and bit transitions around it before diagnosing convergence.
- [ ] Inspect trained-state gradients with fixed head dimensions and optimizer-sized
  batches if a loss plateau remains; initialization probes alone are insufficient.
- [ ] Extend the Hamiltonian model with explicit causal order/directional statistics,
  retaining matched identity, calibrated bigram and classical graph controls.
- [ ] Select one primary metric and a small contrast list before fresh-seed or
  source/template-held-out confirmation.

The cleanup submitted no DGX jobs. The subsequent context POC ran on CPU under
its own [protocol and report](experiments/hamiltonian_study/CONTEXTUAL_POC.md).
The historical broad benchmark expansion plan is retired. Use the [study protocol](experiments/hamiltonian_study/PROTOCOL.md)
for Hamiltonian experiments and the [DGX guide](docs/dgx-a100.md) before cluster work.
[Validation](docs/validation.md) records the cleanup checks.
The retained suite passes 267 tests with 46 hardware skips; Ruff and CPU backend
parity pass. Repository-wide Pyright has zero errors, with nine existing warnings
for optional PDF-export imports. The 186 retired files remain recoverable in Git.

Dependency setup defaults to CPU PyTorch plus development tools. CUDA is opt-in
with `uv sync --locked --no-group cpu --group cuda`; GPU commands must keep
`uv run --no-sync`. The preceding setup passed 308 tests (46 hardware skips),
CPU preflight, Ruff and configured Pyright; CUDA was checked by dry run only.
