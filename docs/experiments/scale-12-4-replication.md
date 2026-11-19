# Replicate the 4,096-token comparison

Declared 18 November 2026. Extend the [seed-11 Hamiltonian and matched MLP
comparison](scale-12-4-mlp.md) to three seeds and add the [raw QCSE
control](qcse-wikitext.md) on the same WikiText-2 tokenization. The earlier
512-token studies remain the source for the report's context, coverage and
architecture ablations.

## Fixed protocol

| Model | Fresh seeds | Vocabulary and readout | Trainable parameters |
|---|---|---|---:|
| `adaptive` | 22, 33 | 12 vocabulary qubits and 4 auxiliary qubits, direct Born readout | 35,520,800 |
| `adaptive_mlp` | 22, 33 | Matched classical features and parameter budget, softmax | 35,463,168 |
| `qcse` | 11, 22, 33 | 12 vocabulary qubits, latest 16 tokens, 64 ansatz layers, product of bit marginals | 2,240 |

Use the original 4,096-token byte-BPE dataset and source snapshot. Verify both
archived seed-11 selection manifests, the software versions and the prepared
dataset SHA-256
`9b499cfde14ff1e83f3ffcf7fa819516703b1896dd60313361ca8a8427f38544`.
The 524,288 training rows, 4,096 validation targets and 8,192 held-out test
targets preserve the original tokenizer, training-only statistics and official
article splits. Within each seed the three families use the same shuffled rows.
Keep 30 epochs, batch 512, 30,720 updates, constant Adam .003, gradient clipping
at 5 and NLL loss. Select by minimum validation NLL over the 31 checkpoints,
breaking ties by higher sampling success and then earlier epoch. Reuse the
locked seed-11 neural fits in the analysis. Hold out test scoring for a separate
pass after all checkpoints are locked and the final comparison is specified.

Report each seed and three-seed means and SDs for NLL, bits per byte, expected
sampling success and most-likely-token accuracy. Include paired article-bootstrap
intervals on mean per-target differences, and whitespace and non-whitespace
breakdowns. For QCSE, expected sampling success refers to sampling from the
independent-bit distribution. Assess metric-specific consistency across seeds
before changing the main comparison. Article intervals are conditional on the
selected fits and reused validation articles. The 512- and 4,096-token experiments
also differ in parameter count, architecture and training data coverage, so any
cross-scale comparison is descriptive.

## DGX allocations

Train in a separate CUDA checkout, `~/quantum-semantics-scale-12-4-replication-20261118`.
Use `prod10` with `nvidia_a100_1g.10gb` for the Hamiltonian and MLP fits, as in the
original measured runs. Use `prod40` with `nvidia_a100_3g.40gb` for raw QCSE. At
12 qubits and batch 512, one complex128 state batch uses 32 MiB. The 64-layer
ansatz retains one state batch per qubit rotation for angle gradients, about
24 GiB before the remaining graph and working memory. A realistic CUDA probe
checks the allocation before training. Each array task uses four support CPUs
and writes to its own model/seed directory. The arrays have no dependencies.

Each task runs six full-size Adam updates on CUDA with resident training data
and a saved model copy, checking CUDA parameters, probabilities, gradients and
changed weights. It records memory and a projected duration, then discards the
probe and reinitializes the declared seed for training. The launcher is
`uv run python -m output.scale_12_4_replication.run` inside `dgx.sbatch`.

## Submission — 18 November 2026

| Slurm array | Model | Partition | Seeds | Time limit per fit |
|---|---|---|---|---|
| 11835 | Hamiltonian | `prod10` | 22, 33 | 12 h |
| 11836 | Matched MLP | `prod10` | 22, 33 | 4 h |
| 11837 | Raw QCSE | `prod40` | 11, 22, 33 | 24 h |

Both Hamiltonian seeds and MLP seed 22 passed their CUDA probes and began
training. Their probes reserved 6.82 GiB and 3.80 GiB respectively, with projected
fit durations of 5.96 h and 1.45 h. The account's four-job limit includes the
earlier 512-token job 11773. Raw QCSE and MLP seed 33 are queued. The pending
MLP task's priority was lowered to give QCSE the next available account slot
when a 40 GB slice is available. Raw QCSE's CUDA memory and runtime measurements
are pending. The laptop completion notifier is active.

## Completed neural comparison — 19 November 2026

The four new neural fits (arrays 11835 and 11836) completed with exit `0:0`.
The saved dataset and weight hashes match their selection manifests. All six
fit records and validation arrays, including the archived seed-11 pair, agree
with the declared data, settings and checkpoint selection. Each history
contains 31 validation checks and the full 30,720-update exposure. The
Hamiltonian fits select epochs 19, 18 and 18, while the MLP fits select epochs
4, 3 and 4. The held-out test remains unscored.

| Predictor | Validation NLL | Bits per byte | Sampling success | Argmax accuracy |
|---|---:|---:|---:|---:|
| Hamiltonian | 3.17909 ± 0.00785 | 2.11223 ± 0.00521 | 37.320% ± 0.181 | 49.325% ± 0.070 |
| Matched MLP | 3.15812 ± 0.01268 | 2.09829 ± 0.00843 | 41.986% ± 0.302 | 49.072% ± 0.184 |

The table gives means and sample SDs across seeds 11, 22 and 33. MLP sampling
success is higher at all three seeds. The mean MLP-minus-Hamiltonian difference
is **+4.666 percentage points [4.343, 4.963]**. The differences in aggregate
argmax (**−0.252 points [−0.919, +0.429]**) and NLL (**−0.02097
[−0.04909, +0.00890]**) remain unresolved. On the 2,383 non-whitespace targets,
MLP sampling is **7.841%** versus **4.731%**, and MLP argmax is **15.722%** versus
**14.520%**. Both differences favour MLP in the paired article intervals.
The Hamiltonian leads on whitespace argmax and the MLP leads on whitespace
sampling. The MLP's final-epoch validation NLL exceeds 9.1 at every seed.

Intervals are nominal 95% ranges from 2,000 paired article resamples. For the
three-seed comparison, each target's difference is averaged across seeds before
resampling, conditional on the selected checkpoints and reused validation
articles. See the [per-seed and bucket metrics](scale-12-4-replication-neural.csv)
and [paired intervals](scale-12-4-replication-neural-paired.csv).

Raw QCSE seeds 11 and 22 passed their CUDA probes on `prod40`, with peak
allocated memory 27.24 GiB and reserved memory 27.30 GiB. Their full-fit
projections were 12.89 h and 13.20 h. The raw QCSE comparison is pending.
