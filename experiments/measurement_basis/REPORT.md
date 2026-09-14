# Learned measurement bases for causal QCSE

## Result in one sentence

With causal context window **8** and **8 ansatz layers**, a learned arbitrary local
measurement axis improved held-out bitwise BCE and top-5 vocabulary retrieval over
standard Z measurement, but did **not** improve hard-threshold exact-word accuracy.
The simpler one-angle learned basis was effectively tied with Z in screening.

## What was implemented

The original QCSE circuit measures each qubit in the computational Z basis and
returns its marginal probability. Halil's report motivates X, Y, Z, or rotated
measurements because Z alone discards phase-dependent information. Section 6.2 of
the report does not explicitly require that the rotation angles themselves be
learned, so this study tests that stronger, operational interpretation.

Three readouts were compared:

| Name | Circuit immediately before Z measurement | Extra parameters | Reachable local axes |
|---|---|---:|---|
| `z` | Identity | 0 | Z only |
| `learned_y` | RY on each qubit | 10 | X-Z plane |
| `learned_xyz` | RX then RY on each qubit | 20 | Any Bloch-sphere axis |

Two angles are sufficient for an arbitrary measurement axis because an axis has
two degrees of freedom. Adding a third Euler angle would introduce an unobservable
or redundant direction. Every learned rotation was initialized to zero, so all
three variants had the **same ansatz weights and exactly the same predictions at
epoch zero** for a given seed.

The production model and CLI now support all three choices through
`--measurement-basis z|learned_y|learned_xyz`. Checkpoints retain the selected
basis and older checkpoints default to Z.

## Protocol

- Task: causal/GPT-like next-word prediction only.
- Corpus: 1,000 sentences, 7,235 tokens, 729-word vocabulary, 10 qubits.
- Architecture: original exponential-sinusoidal context encoding, forward
  entangling direction, context window 8, ansatz depth 8.
- Parameters: 232 ansatz parameters for every model; 242 total for `learned_y`
  and 252 total for `learned_xyz`.
- Split: sentence-text-grouped; 3,990 training, 999 validation, and 1,246 held-out
  test targets. Identical sentences cannot cross partitions.
- Optimization: 30 epochs, batch size 32, learning rate 0.01, L2 0.001, three
  seeds (11, 22, 33). The ansatz uses the repository's SPSA estimate. The small
  readout layer uses its analytic BCE gradient, averaged at the paired SPSA
  evaluations, so its 10-20 angles are not hidden by a noisy high-dimensional
  gradient estimate.
- Selection: best checkpoint by validation BCE, checked every five epochs. Test
  data were evaluated only for the preselected Z and `learned_xyz` confirmation.
- Backend: exact complex statevectors; no sampling noise or hardware noise.

## Screening

The screen used 640 sampled training and 320 validation targets. Values are mean
± seed SD.

| Readout | Validation BCE ↓ | Selected epochs |
|---|---:|---|
| Z | 0.5767 ± 0.0027 | 30, 30, 20 |
| Learned RY | 0.5763 ± 0.0047 | 30, 20, 20 |
| Learned arbitrary axis | **0.5729 ± 0.0017** | 15, 30, 20 |

The one-angle basis improved mean BCE by only 0.00047 and had larger seed
variation, so it was not promoted. The arbitrary-axis basis improved all three
screening seeds and was confirmed on the full training partition.

## Held-out confirmation

Values are mean ± seed SD over the same 1,246 held-out targets.

| Readout | Parameters | BCE ↓ | Exact word ↑ | Top-5 ↑ | Vocabulary NLL ↓ |
|---|---:|---:|---:|---:|---:|
| Z | 232 | 0.5708 ± 0.0026 | **4.04%** | 10.46% | 5.665 |
| Learned arbitrary axis | 252 | **0.5679 ± 0.0017** | 3.69% | **12.07%** | **5.639** |

Paired differences (`learned_xyz - z`) averaged over seeds, with 95% bootstrap
intervals over held-out sentence-text groups:

| Metric | Difference | 95% interval | Interpretation |
|---|---:|---:|---|
| Bitwise BCE | **-0.00286** | [-0.00457, -0.00124] | Learned basis better |
| Vocabulary NLL | **-0.0260** | [-0.0422, -0.0104] | Learned basis better |
| Top-5 vocabulary accuracy | **+1.61 points** | [+0.68, +2.55] | Learned basis better |
| Exact thresholded word | -0.35 points | [-1.01, +0.32] | No reliable difference |

The BCE improvement occurred in every seed (-0.00148, -0.00200, and -0.00509).
Top-5 improved in two seeds and declined slightly in one; exact-word changes had
mixed signs. The bootstrap intervals quantify held-out sentence variation
conditional on these three fitted seeds and this single data split; they are not
confidence intervals over every possible retraining and corpus.

![Held-out comparison](results/comparison.png)

## Why it probably helped

The learned axes tilted **32.5° ± 1.8°** from Z, so the optimizer did not merely
leave the added gates at identity. Resetting the learned rotations to Z while
holding that model's ansatz fixed worsened test BCE by 0.0602 ± 0.0172. This is
evidence of strong readout/ansatz co-adaptation, not an estimate of the standalone
cross-model gain.

There is also a circuit-specific mechanism. Under Z measurement, the final
ansatz-layer RZ and CRZ gates only change phase and are exactly invisible to all
output marginals. The numerical ablation changes baseline probabilities by at
most floating-point roundoff. With the learned arbitrary-axis readout, zeroing
those final diagonal gates worsened test BCE by 0.0423 ± 0.0274. Rotated readout
therefore makes phase-sensitive capacity already present in QCSE observable and
trainable. This matches the qualitative motivation in Halil's sections 6.2-6.4.

The hard exact-word metric tells a different story because training minimizes
continuous bitwise BCE, while exact decoding thresholds every bit at 0.5 and
requires all ten arbitrary ID bits to agree. Better-calibrated probabilities can
improve BCE and vocabulary ranking without moving every bit across its threshold.
For language-model use, normalized vocabulary ranking is more informative than
turning the ten marginals directly into a raw binary ID.

## Recommendation

Use `learned_xyz` when QCSE outputs are scored probabilistically or used for
top-k retrieval. Keep Z as the conservative default for backward compatibility
and for the current raw-threshold greedy decoder: this experiment does not show
an exact top-1 benefit. Do not prefer `learned_y`; its restricted plane left most
of the potential gain unrealized.

The next output-side improvement should be a valid-vocabulary decoder that ranks
all 729 words by normalized independent-bit likelihood, followed by a clean
comparison against the current threshold-to-ID decoder. That uses the probability
quality gained here without changing the quantum circuit again.

## Reproduction

Run from the `QCSE` directory. Completed jobs are cached, and each worker prints
only one line per completed seed, so there is no progress loop to watch.

```bash
# Three-way screen used for basis selection.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  -m experiments.measurement_basis.run \
  --epochs 30 --layers 8 --window 8 --train-limit 640 --val-limit 320 \
  --seeds 11 22 33 --workers 3 \
  --output experiments/measurement_basis/results/screen

# Full-data confirmation and held-out evaluation.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python \
  -m experiments.measurement_basis.run \
  --methods z learned_xyz --epochs 30 --layers 8 --window 8 \
  --train-limit 0 --val-limit 0 --seeds 11 22 33 --workers 3 --test \
  --output experiments/measurement_basis/results/confirmation

uv run python -m experiments.measurement_basis.summarize
uv run pytest -q
uv run ruff check .
uv run pyright
```

## Sources

- [QCSE paper](../../../References/2509.05729v2.pdf): original circuit and Z-basis
  marginal readout, especially equations 19-20.
- [Amazigh Halil's report](../../../References/Amazigh-Halil_Rapport_2A_CS_R_10-2-2026.pdf):
  basis enhancement and phase-information motivation in sections 6.2-6.4.
