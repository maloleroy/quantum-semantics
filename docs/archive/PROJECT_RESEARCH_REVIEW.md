# Project evidence, literature and remaining work — 2026-10-22

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


**New evidence:** the one-MIG optimization pilot and eight causal mechanism
fits completed and passed the [compact-export audit](../../results/robust-causal-controls/README.md).
The core eight-model comparison is now complete in both tasks at two seeds.
The [next-experiment specification](BENCHMARK_POST_CAUSAL_PLAN.md) supersedes
the earlier launch priorities. It includes the newly derived CRZ/ZZ coordinate
equivalence and bounded-entanglement simulation opportunity.

The project needs **two complete studies**, causal left-only prediction and
bidirectional masked prediction. Neither can stand in for the other. The new
[32-fit audit](../../results/robust-target-controls/README.md) resolves a context/target
confound; its eight new causal controls complete the initial mechanism matrix,
but not either full research study. The [dual-track protocol](BENCHMARK_DUAL_TRACK_PLAN.md) specifies the
experiments, implementation gaps, robustness and compute decisions.

This review reads existing project reports, progress/architecture notes and
available run-status metadata, and independently audits the new compact export.
Older raw binaries were not all rerun/reverified in this review. Prior modern
audits below retain their documented scope; report reuse and extracted copies
are not additional experiments. Uncommitted focused-experiment work is listed
as evidence without changing or publishing those files.

## Evidence ledger: experiments are not interchangeable

| Family / evidence | Status and finding | Interpretation boundary |
| --- | --- | --- |
| Original production cluster sweep, `results/cluster-sweep/` | 44/45 configurations had complete historical exports. Selected exact-bit word accuracy about 1.07% test. | Ten-epoch pipeline-era experiment; one incomplete configuration, varying batch/update budgets. Do not merge with word-softmax studies. |
| Attention CV / circuit ablation / hyperparameter reports | 5,000-sentence cap; learned input encoding improves over fixed QCSE; removing circuit can improve retrieval; tuned classical LR 0.01 reaches 17.44% cosine top-1. | Reports reuse fits.25×5,000 draws; cosine retrieval differs from CE ranking; no-circuit was not equally retuned in every report. |
| Semantic prototype ablation/pathway/matrix/grid/hyper studies | Mostly 1,000-sentence caps and single seed 42,25–30 epochs; identity/frozen competitive. Source-specific capped fits are available. | Useful debugging and representation hypotheses, not full-corpus mixture interventions or a robust quantum advantage. |
| `semantic-local-30` and pipeline-validation notes | Historical repeated local protocol and tiny pipeline checks completed as documented. | Do not count smoke/validation fits as scientific confirmation; no repetition of CPU-heavy local sweeps. |
| Focused local matrix, uncommitted reports | Revised 30-fit matrix reported complete; encoding/attention and 16 causal fits used distinct exposure caps. Causal exact top-1 mostly zero; toy attention benefit encoder-dependent. | Historical 14-fit attention report uses a133-example synthetic test and older budgets. Cannot pool with revised causal or DGX benchmarks. |
| Later local causal best-setting runs | Eight 1,024-draw fits marked complete with test files; eight 4,096-draw fits marked interrupted with no tests. `full-cv-50` also interrupted. | Record incomplete runs; do not silently infer final quality or restart obsolete low-exposure jobs. |
| `outputs/train` / `train_64l_gpt` and root images | Legacy fixed-encoding QCSE context directions, with 64 ansatz layers in the available configurations. | “BERT/GPT” in filenames is not evidence of a 64-layer Transformer comparison. They are not the modern BERT baseline. |
| [Timed CV](../../results/robust-timed-cv/README.md),90CV records | Six models ×five folds ×three seeds; early-stopped 50×5,000 maximum, sampled final validation. Hybrid CRZ/id close; paper QCSE near frequency. | Shorter/different evaluation protocol; no pooling of accuracy with 100×20,000 full-validation runs. |
| [Main screen](../../results/robust-screen-main/README.md),156CV/control records |52 arms ×three folds ×one seed.100×20,000 for trainable arms; depth 8 expensive without clear gain; ID permutation sensitivity; analytic identity parity. | Single training seed; parameter-free controls are not trained fits. Width/depth change head capacity/RNG; screen winners need equal tuning. |
| [Followup](../../results/robust-followup/README.md),258CV+30refits | Both native tasks; three learning rates, fresh model comparisons, common runtime. MLP beats BERT in all 15 paired CV cells and three test seeds per task; selected LR 0.003 also wins under top-5. | Native CBOW includes sentence-initial targets. Source 483bea6 and full data protocol differ from earlier phases. |
| [Independent prediction review](../../results/robust-followup-prediction-review/README.md) | Reaggregated float 64;10,000 paired bootstraps; original top-5 CBOW CRZ−id+0.2705pp, common-position+0.2787pp; top-1 uncertain. | Post-hoc secondary result; matching evaluation positions did not yet match training pools. The apparent QCSE context gain was mostly target composition. |
| [Replicated gradients](../../results/gradient-replicated-10324/README.md) |288 conditions /9,216 gradient evaluations, both objectives; raw statistics and sampled input reconstruction checked. No initialization collapse established through q12. | Diagnostic evaluations, not training fits. Eight targets per batch, crossed heads/batches and changing head shape; no later-training scaling inference. |
| [Target controls](../../results/robust-target-controls/README.md),32 fits | Common positive training/evaluation targets; seeds 47/48. MLP best in all three recipes. Causal CRZ top-5 small positive in both seeds; bidirectional mixed. | Two seeds, reused fold 0, fixed LR, no new test; four causal mechanism arms were missing at this stage. |
| [Causal completion](../../results/robust-causal-controls/README.md),8new fits | Local/frozen/XX/ZZ complete the two-task eight-model matrix. Causal CRZ improves top-5/NLL versus frozen, with lower top-1. ZZ top-5 improves versus identity in both seeds of both tasks. | Both tracks remain exploratory, two initializations, reused fold; CRZ/ZZ have equivalent function classes. Older eight L2pad fits are supplementary, not another core task. |
| Same export's microbatch pilot, 64 conditions | Both contexts/four anchors, full/partial batches, recorded one-update parity passes; micro 64 gives 4.9–7.7× training-step speedup. | Diagnostic conditions, not training fits or whole-fit speedups. Evaluation/long-run dropout trajectories need checking before adoption. |

The active robust corpus is 202,172 deduplicated sentences, vocabulary 10,864.
Its frozen vocabulary was built from all source rows before splitting; this
is a disclosed transductive convention, not a train-only OOV benchmark.
The verified common-target training fold has 1,014,085 eligible targets; each
fit takes 2M replacement draws, not 100 complete passes over the corpus. Uniform
replacement exposure also does not guarantee every eligible target is seen.

## Conclusions supported now

1. **Classical baselines are strong in both directions.** MLP remains the best
   observed modern followup/reference in both tasks and the new common-target
   experiment, with much lower simulation cost. Earlier screen BERT wins do
   not contradict this: comparison capacity, LR and evaluation differ.
2. **Quantum circuit gains depend on endpoint and task.** In the new causal
   track CRZ−identity top-5 is+0.177pp, top-1−0.040pp. Bidirectionally the top-5
   gain is+0.071pp with mixed seed signs, while NLL improves. Keep both the
   small positive and null/negative findings; do not turn NLL into a top-1 win.
3. **Learning the circuit has task/endpoint-dependent effects.** Causal CRZ
   beats frozen CRZ on NLL (−0.00800) and top-5 (+0.2199pp), both seeds;
   top-1 falls 0.2011pp in both. Bidirectionally the contrast remains mixed.
   This replaces the earlier missing-causal-control conclusion. Frozen circuit
   features remain useful, but they are not evidence that learning is always
   irrelevant; encoder/head still learn in the frozen-circuit arm.
4. **Observed q8 training has active gradients.** Depth/qubit degradation can
   reflect optimization, encoding, readout, exposure, capacity or overfitting.
   No NLP barren plateau has been established, and its absence is not proved.
5. **100 epochs is a fixed comparison budget, not proven convergence.** Late
   monitored loss/top-5 still improve in both tracks. Extend only a small
   prospective subset with intact optimizer/RNG and explicit new endpoints.
6. **Balanced source learning has barely been studied under the robust protocol.**
   Phrases contribute only about 0.4% of training target mass; the old capped
   source matrix changes test populations. New fixed-evaluation sampling
   interventions are needed, separately in each direction.
7. **No present result establishes semantic understanding or quantum speedup.**
   The controlled task is short-context word prediction on curated sentences,
   largely cleaned-source data. A downstream/external generalization claim
   requires a new evaluation design, not more resamples of this validation set.
8. **CRZ versus ZZ currently tests coordinates/initialization, not greater
   expressivity.** Free local RZ fields absorb the CRZ's local component, giving
   an invertible per-layer conversion to ZZ. Float 64 state/gradient checks pass.
   Default Gaussian initializations and coordinate-wise Adam do not map
   equivalently. Additionally, the product-input depth 2 chain's Schmidt rank
   is at mostfour across every cut, allowing exact tensor-network simulation
   polynomial in q at fixed depth. That backend is not implemented/profiled.
9. **Exposure and weighting deserve explicit analysis.** Frozen-sampler replay
   predicts 86.08–86.11% unique targets seen over 2M draws; the actual streams
   were not exported. Causal ZZ's token-NLL effect is−0.00228 but sentence-macro
   effect+0.00135. Weighting changes the conclusion; token/source/rank confidence
   exports are needed to explain it, not a post-hoc story about frequency.

## Literature checked and implications for this project

Primary pages checked on 2026-10-21 and extended on 2026-10-22; this is a targeted review of the project's
questions, not a systematic exhaustive review of all QNLP research.

| Primary source | Finding and actionable implication |
| --- | --- |
| [QCSE v2](https://arxiv.org/html/2509.05729v2), V-A/AppendixA | Five numerical context encodings are discussed. Its evaluations use tiny corpora, distinct quantum/classical objectives and a half-bits similarity threshold; these numbers are not exact-word top-1. Reproduce its metric separately and use a shared valid-word endpoint for our comparisons. Its numerical context matrix is not a learned graph Hamiltonian. |
| [Bengio et al. 2003](https://www.jmlr.org/papers/v3/bengio 03a.html) | A learned word lookup with a feedforward conditional predictor is a substantive language-model baseline. The strong MLP is a finding to explain, not a dispensable sanity check. |
| [GPT generative pretraining](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), section 4; [BERT](https://aclanthology.org/N19-1423/) | GPT uses a masked-attention decoder; BERT jointly uses both directions. Our causal BERT-with-left-input model does not instantiate GPT. Add a true causal decoder; keep scratch word-level controlled prediction separate from their pretrained/tokenization/downstream recipes. |
| [Chen & Goodman 1996](https://aclanthology.org/P96-1041/) | Count-model smoothing is consequential. Add normalized, validation-tuned count/backoff baselines per task; naive unsmoothed n-gram failure would be an unfair classical reference. |
| [McClean et al. 2018](https://arxiv.org/abs/1803.11173) | Barren plateaus concern gradient concentration/scaling under stated circuit ensembles. Poor accuracy or a flat learning curve alone does not identify one. Measure gradients and their mean/variance/RMS as qubits/depth change. |
| [Cerezo et al. 2021](https://www.nature.com/articles/s41467-021-21728-w) | Locality/depth and ensemble assumptions affect trainability. The theorem is not a blanket guarantee for a nonlinear word-softmax loss after local measurements; probe the actual NLP objective in both tracks. |
| [Grant et al. 2019](https://quantum-journal.org/papers/q-2019-12-09-214/) | Identity-block initialization can improve initial trainability. Our small Gaussian initialization is not automatically the same construction; contrast scales and meaningful gradients rather than treating low initializer variance as vanishing mean gradient. |
| [Schuld, Sweke & Meyer 2021](https://arxiv.org/abs/2008.08605) | Encoding gates constrain accessible Fourier frequencies. Vary encoding/readout and compare a simple classical feature map before adding circuit depth indiscriminately. |
| [Schreiber, Eisert & Meyer 2023](https://arxiv.org/abs/2206.11740); [Cerezo et al. 2025 perspective](https://www.nature.com/articles/s41467-025-63099-6) | Classical surrogates and some trainable circuit structures challenge advantage claims. These are not a theorem dequantizing every circuit. Our identity control is already exactly analytic; add a feature-matched nonlinear classical model to each task. |
| [Farhi & Gutmann 1998](https://arxiv.org/abs/quant-ph/9706062) | Continuous-time unitary graph evolution is a valid separate construction. It does not show that a co-occurrence graph will improve NLP; compare identity/heat/classical graph filtering and profile sparse evolution. |
| [DoReMi 2023](https://arxiv.org/abs/2305.10429) | Data-mixture weighting can be an optimization/design variable. Its large-model findings do not predict our optimal mixture. Start with declared modest source-weight interventions and fixed evaluation, not an automatic dynamic weighting system. |
| [Lee et al. 2022](https://aclanthology.org/2022.acl-long.577/) | Near duplication can affect evaluation beyond exact duplicate removal. Existing sentence grouping handles exact matches, not all templates/paraphrases. Reserve source/template-held-out analysis and audit near duplicates before broader claims. |
| [Bouthillier et al. 2021](https://arxiv.org/abs/2103.03098); [Dror et al. 2018](https://aclanthology.org/P18-1128/) | Training/selection/data variability and the evaluation design matter. Paired sentence resampling is useful but does not replace independent training seeds or a new population. Prespecify a few contrasts and retain both tracks rather than bootstrapping hundreds of selected comparisons into apparent certainty. |
| [Larocca et al. review, 2025](https://arxiv.org/html/2405.00781v2) | Probabilistic/deterministic concentration depend on inputs, ansatz, measurements and loss. Small gradient variance alone can be misleading when the mean remains useful; inspect loss and parameter-level concentration and distinguish structural zeros. Our active q8 histories neither diagnose nor exclude qubit-scaling barren plateaus. |
| [Bowles, Ahmed & Schuld 2024](https://arxiv.org/abs/2403.07059) | Their 12-model/160-dataset binary-classification benchmark finds strong classical references and sensitivity to experimental design. It motivates credible baselines and negative controls here; it does not determine an NLP ranking or show all quantum models fail. |
| [Del Castillo, Zhao & Pei 2025](https://arxiv.org/pdf/2502.20744) | The ansatz comparison studies text classification with hyperparameter tuning. It supports controlled expressivity/optimization comparisons, but its task/metrics cannot be imported as word-prediction evidence. Our CRZ/ZZ equivalence is a separate derivation for our implementation. |
| [Quixer, Khatri et al. 2024](https://arxiv.org/html/2406.04305v1), sections 4/6 | This is an actual subsequent-word quantum Transformer study on Penn Treebank, using matched context/stride and tuned classical references; its Transformer reference is strongest in the reported table. LCU/QSVT and simulation/hardware overhead differ from our hybrid. It motivates a later standard causal-LM dataset, not copying its A100 duration to our MIG or replacing our model wholesale. |
| [Kerenidis 2026, v2](https://arxiv.org/html/2607.24014v2), IV.1/Proposition 8 | For its specified passive-FLO architectures/block-product inputs, fixed-body readouts admit polynomial classical evaluation via reduced density matrices, despite possible sampling hardness. It is not a theorem for our CRZ circuit. Its lesson is to analyze the actual prediction observables, rather than infer hardness from Hilbert-space size. Version 2 is important; older summary claims omit this readout qualification. |
| [Vidal 2003](https://arxiv.org/abs/quant-ph/0301063) | Restricted entanglement enables efficient classical simulation. Our product-input chain has a direct rank ≤ 2^depth bound, providing a concrete exact-MPS candidate. The bound says nothing about measured q8 wall time, the paper encoder or general deep/dense circuits. |

## Falsifiable hypotheses and the next discriminating work

| Hypothesis | Discriminating test in **both** tasks | What would weaken it |
| --- | --- | --- |
| Circuit features add mainly fixed nonlinear structure | Identity/local/frozen/trainable and feature-matched classical replacement | A reproducible learned-circuit gain over frozen and classical controls at equal tuning/exposure |
| The paper-style decoder is an important bottleneck | Fixed-code versus learned word head on the same features, with classical encoder controls | Learned heads fail to improve full-word quality after equal tuning; frequency/encoding accounts for the result |
| q growth causes optimization trouble rather than insufficient features | Fixed surrounding head, controlled q projection, actual signed gradients/update ratios and later-state probes | Loss worsens while useful gradients/updates remain healthy and simpler encodings resolve it |
| The learned models are exposure-limited at 100 epochs | Prospective 100/200 full-validation checkpoints on a small fixed subset | Later scores stagnate or worsen despite improving training loss |
| Uniform-token sampling suppresses phrase-domain learning | Uniform-sentence /10% phrase policy with fixed targets/source strata | No robust phrase improvement, or large overall degradation for negligible phrase benefit |
| A graph prior supplies useful language structure | Train-only graph with unitary/heat/identity/classical filters and source-held-out evaluation | Benefits disappear with matched classical filtering or fail to generalize |
| Default ZZ wins because its physical initialization/optimizer coordinates differ | Map CRZ to the same initial ZZ function, retain encoder/head, then compare coordinate-wise Adam in both tasks | Same mapped trajectories but persistent default difference favors initialization; any remaining claim requires a separately controlled explanation |

The fixed-code interpretation is an algebraic result about our actual
independent-bit adapter, not a claim that all QCSE/QNLP models use that decoder.
The new tiny tests compare its word-softmax probabilities and gradients against
the reduced code head, including invalid bitstrings. They also distinguish
joint measurement probabilities from marginals. No corpus training is needed.

## Compute/research work remaining

**Ready evidence:** completed 32+8-fit audits, paired 10,000-draw intervals, all
seed results, rank counts, history/compute summaries and source-composition
counts. The original 32 fit wrappers used 33.616 MIG-hours over 10h47m; the new
eight used 15.357 MIG-hours over 3h53m, peak four. The latter observed cost/span
were about 6–7% below forecast. Both exclude profile/queue/
packaging overhead and are not full Slurm accounting. One-MIG optimization
pilot is complete; micro 64 has not yet been adopted.

**Next implement/profile:** sixteen full-trajectory micro 64 validation fits,
fresh evaluation/whole-fit profiles and reproducible execution-recipe guards.
Training-only estimate≈1.46 MIG-hours; whole-batch ETA remains unmeasured.
GPT/classical-feature CPU prototypes exist (373280/186824 parameters); existing
MLP-16 has 187312 versus hybrid 186190 and is needed to control total capacity.
For the new prototypes,
factory/resume/CUDA integration remains. Then focused model/coordinate controls,
not another generic ansatz grid. Decoder/joint-Born heads, new samplers,
longer-budget continuation, fixed-head gradient probes, exact-MPS evaluation
and sparse graph evolution each need tests/profiles; no aggregate ETA exists.

**Next export from existing DGX outputs:** raw validation predictions or
reproducible source/frequency/position cluster sums, training-count buckets and
paired rank 1/rank 2–5/miss transitions; checkpoint/optimizer states for later
gradient and continuation designs; Slurm accounting for full allocation cost.
The saved token NPZ has top-five IDs/per-token metrics, not full logits.
Confidence/calibration/entropy and full MRR need fresh checkpoint inference,
not more training. Compact top-k/NLL sums alone do not identify them or
source-conditioned accuracy.

**Confirmation:** use three fresh seeds for a small fixed selection in each
track; add grouped splits only for a specific split-sensitivity question.
Changing a split of these already-used corpora is not an independent new test.
External data and downstream/native GPT/BERT studies remain research work.
Keep negative results, source/runtime hashes and all predictions. The shared
production limit staysfour 10 GB MIG instances; no new jobs have been submitted
as part of this local audit.
