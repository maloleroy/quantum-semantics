# Quantum semantics: branch state

Updated 2026-10-22 · `codex/qcse-bert-robust-benchmark`

The branch tests whether QCSE/circuit features help word prediction beyond
classical models. **Left-only and bidirectional prediction are separate,
ongoing studies.** The latest received cohort is complete; no new DGX batch
was submitted during this cleanup.

## Read by question

| Question | Document |
| --- | --- |
| How do I use the code? | [README](README.md) |
| What do the models actually compute? | [Models and code map](docs/models.md) |
| What makes a comparison fair/reproducible? | [Protocol](docs/protocol.md) |
| What ran, what finished, what did it show? | [Runs and diagnoses](docs/runs.md) |
| Hamiltonians, NLP, plateaus, data: what does the literature support? | [Research and literature](docs/literature.md) |
| How do I operate the DGX jobs safely? | [DGX-A100](docs/dgx-a100.md) |
| What became faster, and what is only proposed? | [Optimizations and compute](docs/optimizations.md) |
| What should run next? | [Next experiments](docs/next-experiments.md) |
| What was checked? | [Validation](docs/validation.md) |

## Since branch creation

The branch starts at `5c748b4` (2026-10-08), from `2e037ca`. Earlier SPSA,
semantic-decoder and attention code was inherited; its results stay separate.

| Stage | Change and outcome |
| --- | --- |
| Oct 8–9 | Audited QCSE equations/generators; added task-matched QCSE/hybrid/MLP/BERT adapters, grouped CV, paired reports and trainability probes. |
| Oct 11 | Fixed `sbatch`-only launch, terminal-job queue handling and four-MIG ceiling; added detailed timing. The 90-fit timed cohort completed, but used shorter training than later studies. |
| Oct 11–16 | Fixed-budget 156-record screen: 100×20,000 draws. Deeper QCSE was costly; identity was competitive. Parameter-free records are not trained fits. |
| Oct 16–20 | Focused follow-up: 258 CV +30 refits across both native tasks. MLP-32 led; circuit top-1 gains were inconclusive. Frozen training source: `483bea6`. |
| Oct 20 | Corrected float 64 prediction reductions. Found a target-position confound; reviewed 9,216 initialization-gradient evaluations without establishing an NLP barren plateau. |
| Oct 21 | 32 common-target controls isolated context direction/training population. Added local/frozen/XX/ZZ bidirectional mechanisms. |
| Oct 22 | Eight missing causal controls completed: the core now has eight models ×two seeds ×two tasks. Microbatch pilot passed recorded one-step checks. Derived CRZ/ZZ coordinate equivalence and bounded chain entanglement. |
| Current cleanup | Structured docs; added previously local focused code/results and batched encoding with parity, checkpoint and resource-limit fixes. These changes are not new DGX performance measurements. |

## Findings to carry forward

- MLP-32 leads the current paired results, but has roughly twice the hybrid's
  parameters. MLP-16 is the close-capacity control still to train on this protocol.
- ZZ−identity top-5: +0.362pp causal, +0.347pp bidirectional. Only two seeds,
  reused validation, mixed causal top-1/NLL: exploratory evidence.
- Causal trainable CRZ beats frozen on NLL/top-5 and loses top-1. The
  bidirectional learning contrast remains mixed.
- Current CRZ/ZZ chains are reparameterizations with free local RZ fields.
  Their ranking does not establish an expressivity difference.
- No demonstrated quantum advantage, semantic generalization or NLP barren
  plateau. Low variance alone is insufficient; readout and exposure matter.

## Next, in order

- [ ] Export source/frequency/length/position and paired rank-transition statistics
  from existing DGX predictions; calibration/full MRR require checkpoint inference.
- [ ] Implement/profile 16 full-budget microbatch 64 checks against existing
  microbatch 8 anchors, both tasks. Training-only proxy: 1.46 MIG-hours;
  whole-fit ETA still needs evaluation/setup/save profiles.
- [ ] Integrate scratch causal GPT and classical angle features; add MLP-16.
  Ten fixed-recipe discovery fits, or a separately declared tuning stage.
- [ ] Test mapped CRZ/ZZ initial functions before another gate-family sweep.
- [ ] Implement focused readout/joint-Born, fixed-head plateau, sampler/mixture,
  convergence and graph-Hamiltonian modules; retain both directions.
- [ ] Confirm a short prespecified contrast list with fresh seeds and external/
  source/template-held-out evaluation. More bootstrap draws are not more fits.

The next-batch launcher is **not implemented**. Keep the frozen DGX checkout
and its manifests intact. All new launches must respect four concurrent
one-10 GB MIG instances. Earlier notes and exact protocols remain in the
[archive](docs/archive/README.md); numerical reports remain under `results/`.

Latest local check: 245 passed, 46 hardware skips; Ruff and project/changed-script
Pyright pass. [Validation details](docs/validation.md).
