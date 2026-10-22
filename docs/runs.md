# Runs, results and diagnoses

Latest received evidence: the eight-fit causal completion exported on 2026-10-22.
Its JSON/CSV summaries and [independent audit](../results/robust-causal-controls/README.md)
are committed. No new training was launched in the documentation cleanup.
Completion here means received artifacts passed their stated checks, not a live
query of today's DGX queue.

## Branch experiment ledger

| Cohort | Status / budget | Finding and boundary |
| --- | --- | --- |
| Initial `robust-first` | One pilot fit was reported complete before replacement by the timed study. | Not 90 completed independent fits. Completed Slurm rows initially blocked submissions; queue guard was fixed. |
| [Timed study](../results/robust-timed-cv/README.md) |90 CV fits, six models ×five folds ×three seeds;50×5000 maximum with early stopping and sampled final validation. Refits absent from received export. | Hybrid/identity close; exact fixed-word QCSE near frequency. A shorter development protocol, not the later 100×20k comparison. |
| [Main screen](../results/robust-screen-main/README.md) |156 records =52 arms ×three folds ×one seed;100×20k trainable arms, full final validation. Five proposed refits were not included. | Depth 8 QCSE costly without clear gain; identity competitive. Includes parameter-free controls. BERT-32 beating MLP-16 did not establish a capacity-matched Transformer win. |
| [Follow-up](../results/robust-followup/README.md) |258 CV +30 refits; refinement/LR selection then five-fold, three-seed confirmation for both native tasks;100×20k. | MLP-32 beats BERT-32 across paired confirmation cells/test seeds. Circuit primary top-1 intervals include zero. Native CBOW includes first-word targets. |
| [Prediction review](../results/robust-followup-prediction-review/README.md) |Reanalysis of those predictions, not new fits; float 64 sums and 10k paired bootstraps. | Corrected reductions; secondary CBOW top-5 gain retained. Matching evaluation positions alone left a training-pool confound. |
| [Replicated gradients](../results/gradient-replicated-10324/README.md) |288 conditions /9216 circuit-gradient evaluations, both tasks; job 10324. | No initialization collapse established through q12. Eight targets/batch; changing head shape and no trained-state probes limit inference. These are not training fits. |
| [Target controls](../results/robust-target-controls/README.md) |32 fits: L4/L2pad four anchors ×two seeds each; L2R2 eight models ×two seeds. Common positive training/evaluation targets;100×20k. | Isolates target composition and context direction. Missing causal local/frozen/XX/ZZ arms motivated the completion. No new test selection/refits. |
| [Causal completion](../results/robust-causal-controls/README.md) |Eight new L4 local/frozen/XX/ZZ fits, seeds 47/48;100×20k. Array 10400, pilot 10399. | Core now 32 records: eight models ×two seeds ×two tasks. Eight older L2pad controls are supplementary;40 distinct trained fits across the two cohorts. |
| Same export's microbatch pilot |64 conditions: two tasks ×four anchors ×two batch sizes ×four microbatches; five timed steps each. | Recorded forward/gradient/one-Adam parity passes. Training steps 4.9–7.7× faster at micro 64; whole trajectories and full-fit speedups unmeasured. Scientific fits kept micro 8. |

Do not sum reanalyses, pilot conditions and reused fits into a new training count.
The robust frozen source is 483bea6; copied diagnostic/control extensions carry
their own hashes. New branch edits do not change the received runs.

## Current paired quality

Means over seeds 47/48 on the same 32348 validation sentence groups/252120
positive-position targets. Accuracy columns are percentages.

| Model | L4 NLL | L4 top-1 | L4 top-5 | L2R2 NLL | L2R2 top-1 | L2R2 top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-32 |3.24873|35.656|61.463|2.20636|54.893|77.452|
| BERT-32 |3.33044|34.084|59.546|2.30797|51.722|75.553|
| q8 CRZ |3.46637|33.316|58.119|2.46840|50.235|73.597|
| q8 identity |3.47083|33.357|57.942|2.47789|50.032|73.526|
| q8 ZZ |3.46854|33.259|58.304|2.45725|50.378|73.873|

All eight arms, seed rows, ranks and intervals are in the
[causal audit](../results/robust-causal-controls/README.md). MLP/BERT have roughly
twice the hybrid's parameters; the table is a fixed-recipe comparison, not an
equally tuned/capacity-matched architecture result.

## Diagnoses that changed the next experiment

1. **Endpoint trade-off:** causal CRZ−frozen is−0.00800 NLL,+0.2199pp top-5 and
   −0.2011pp top-1. All three signs agree across the two seeds; top-1 uncertainty
   still includes zero. “Learning the circuit never helps” is too broad.
2. **ZZ is a candidate, not a settled winner:** ZZ−identity top-5 is+0.3623pp
   causal,+0.3467pp bidirectional, both seed signs positive. The bidirectional
   seed/sentence nominal interval crosses zero. Causal NLL changes seed sign;
   its token mean−0.00228 becomes+0.00135 under sentence-macro weighting.
3. **Gate coordinates:** current CRZ/ZZ chains are exactly reparameterizable.
   Independent initialization and Adam coordinates differ; increased
   expressivity is not an explanation for their present ranking.
4. **Plateau remains unresolved:** q8 histories show active circuit gradients,
   zero clipping and improving sampled monitors between 50/100. Initialization
   probes through q12 do not settle trained-state or larger-q behavior.
5. **Exposure is not convergence:**2M replacement draws are about 1.97 expected
   corpus passes, with about 86% unique targets seen in seed replay.100 sampled
   epochs are not 100 full passes; monitor gains are not full-fold endpoints.
6. **Error analysis can precede training:** existing NPZ files contain top-five
   IDs and per-token metrics, enough for paired rank/source/frequency strata.
   Full MRR/confidence/calibration require checkpoint inference, not another fit.

Latest intervals use 10k paired seed/sentence draws, nominal 95%, with no
multiplicity/selection correction. The two seeds and reused validation fold
limit claims. No new external semantics test or quantum speedup is established.

## Inherited and local experiments

These explain the code's development; they do not enlarge the robust sample.

| Evidence | Status | What to retain |
| --- | --- | --- |
| [Production cluster sweep](../results/cluster-sweep/short-report.md) |44/45 historical configurations exported; one incomplete. Ten-epoch pipeline-era protocol, varying updates. | Exact-bit test accuracy around 1.07% for the selected report. Do not merge with word-softmax accuracy or restart the obsolete missing job. |
| [Attention CV/ablation](../results/attention-cv-25-ablation/report.md), [LR sweep](../results/attention-hyper-25/report.md) |25-epoch capped studies; reports reuse fits. | Learned inputs are important; circuit removal can help. Cosine retrieval and caps differ from robust CE evaluation. |
| Semantic ablation/matrix/grid |Mostly 1000-sentence caps, seed 42 and 25–30 epochs; [historical notes](archive/SEMANTIC_PROTOTYPE.md). | Frozen/identity competitive. Source-specific capped tests did not isolate mixture weights on a fixed population. |
| Revised focused local matrix |30 completion markers in the historical ledger; published [causal 16 report](../results/focused-causal-1024/detailed-report.md). |1024 draws/epoch, 512-target monitor, 2048 test examples. Exact-bit top-1 mostly zero; only one batch 16 repetition nonzero. Not the modern fixed-budget protocol. |
| [Older focused 14 snapshot](../results/focused-interim-14/detailed-report.md) |Separate encoding/attention reports from earlier 5000-draw runs. |Attention has a133-example synthetic test. Do not add this snapshot to the revised causal 16 as a homogeneous 30-fit benchmark. |
| Later local causal attempts |Eight 1024-draw best-setting fits complete; eight 4096-draw runs and `full-cv-50` interrupted, as recorded in the earlier inventory. |No final quality inferred for interrupted runs; no automatic restart in this cleanup. |
| Root `*gpt*`/`*bert*` images and old `outputs/train*` |Legacy QCSE context-direction labels, not evidence of Transformer training. |Unattributed loose images remain local; use recorded manifests/reports for claims. |

The [archived progress](archive/PROGRESS.md) and
[benchmark ledger](archive/BENCHMARK_PROGRESS.md) preserve the earlier measurements
and decision sequence. Historical binaries were not all rerun in this cleanup.
Current [validation](validation.md) distinguishes metadata/report review from
independent prediction checks and new backend tests.
