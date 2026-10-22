# Interim focused sweep report — completed 14/30 fits

## Scope and protocol

Architecture diagrams are in [architecture-changes.md](architecture-changes.md).

This report covers only the 14 completed encoding/attention fits from `outputs/focused-sweep-50-local`: three encoding setups and four attention setups, each with repetitions seeded 42 and 43. The causal arm was not complete in this snapshot and is excluded.

The encoding fits used the full phrases + cleaned corpus; the attention fits used the synthetic repetitive sanity corpus. Both used sentence-grouped 64/16/20 splits, 5,000 replacement draws per training epoch, 2,048 fixed validation-monitor examples, validation top-1-first epoch selection, patience 8 with a 10-epoch minimum, and one held-out test score after selection. Encoding tests used 10,000 test examples; attention has 133 held-out examples.

The primary selection metric is validation cosine top-1, followed by top-5 and cross-entropy. Test metrics are descriptive only and were not used to select epochs or setups.

## Aggregate results

| Family | Setup | Validation top-1 | Test top-1 | Test top-5 | Test CE | Selected epoch |
|---|---|---:|---:|---:|---:|---:|
| Attention | Classical encoding + no quantum attention | 61.90% ± 2.69% | 60.90% ± 2.13% | 94.36% ± 1.59% | 0.756 ± 0.042 | 18.5 ± 7.8 |
| Attention | Classical encoding + quantum attention | 65.24% ± 0.67% | 63.53% ± 0.53% | 94.36% ± 1.59% | 0.754 ± 0.003 | 8.5 ± 2.1 |
| Attention | QCSE encoding + no quantum attention | 44.29% ± 4.71% | 33.83% ± 4.25% | 87.59% ± 6.91% | 1.251 ± 0.271 | 15.0 ± 9.9 |
| Attention | QCSE encoding + quantum attention | 63.81% ± 2.69% | 59.77% ± 3.72% | 95.11% ± 1.59% | 0.991 ± 0.146 | 4.0 ± 1.4 |
| Encoding | Classical encoding + zero ansatz | 19.70% ± 0.03% | 18.81% ± 0.03% | 39.16% ± 1.20% | 4.373 ± 0.186 | 11.0 ± 2.8 |
| Encoding | Classical encoding + trainable ansatz | 20.17% ± 1.10% | 19.67% ± 0.47% | 38.37% ± 1.63% | 4.361 ± 0.131 | 11.5 ± 2.1 |
| Encoding | No encoding + zero ansatz | 3.15% ± 0.31% | 2.79% ± 0.46% | 17.05% ± 1.27% | 5.598 ± 0.065 | 6.0 ± 1.4 |

## Encoding conclusions

- Classical encoding is the dominant factor. Relative to the no-encoding/no-ansatz control, classical encoding with a trainable ansatz improves held-out top-1 by about 16.9 percentage points and top-5 by about 21.3 points, while reducing cross-entropy by about 1.24.
- The trainable ansatz is a small secondary effect on top of classical encoding: it adds about 0.86 test-top-1 points and reduces test cross-entropy by about 0.012 relative to the zero-ansatz setup, but loses about 0.79 top-5 points. With only two repetitions, this is weak evidence rather than a decisive ansatz result.
- The no-encoding control is consistently poor in both repetitions: roughly 2.8% test top-1 versus roughly 19–20% for either classical-encoding setup. This is a large, stable setup effect, not a subtle optimizer drift.

## Attention conclusions

- Quantum attention helps the QCSE encoder strongly: approximately +25.9 test-top-1 points, +7.5 top-5 points, and −0.260 cross-entropy relative to QCSE without quantum attention.
- Quantum attention has a much smaller effect with classical encoding: approximately +2.6 test-top-1 points, no mean top-5 change, and −0.002 cross-entropy. This is an encoder × attention interaction: the quantum-attention gain is much larger for QCSE than for classical encoding.
- Classical encoding remains stronger than QCSE without quantum attention by approximately +27.1 top-1 points and −0.495 cross-entropy. With quantum attention enabled, the classical advantage narrows to about +3.8 top-1 points, while QCSE retains a small top-5 advantage.
- The attention test set has only 133 examples and is a synthetic repetitive sanity corpus. These large effects are useful for pipeline diagnosis and ablation direction, but they should not be treated as corpus-level generalization evidence.

## Drift and reliability

- Repetition drift is modest for the encoding setups (test top-1 differences below one percentage point) but larger for attention, especially QCSE without quantum attention. The attention seed-43 run generally selects later epochs and changes the result more than the encoding repetitions.
- Selected epochs vary from early stopping at epoch 3–24 in the attention arm and 5–13 in encoding, with stopped epochs extending to 32. This validates the need for validation-based epoch selection rather than a fixed test-time epoch.
- The primary claims here are directional. Two repetitions are enough to expose large setup effects, not enough to establish tight uncertainty intervals; the new 1,024-sample run is being monitored separately and should supersede this interim snapshot for the final report.

## Figures

- `encoding-test-metrics.png`: test top-1/top-5/cross-entropy by encoding setup, with repetition points and mean ± SD.
- `attention-test-metrics.png`: corresponding attention comparison.
- `encoding-cross-entropy.png` and `attention-cross-entropy.png`: narrower loss-only figures with larger labels.
- `encoding-validation-curves.png` and `attention-validation-curves.png`: validation drift across epochs.
- `paired-contrasts.png`: paired parameter/setup effects.
- `selection-epoch-drift.png`: selected versus stopped epochs.
- `seed-drift.png`: repetition-to-repetition held-out top-1 drift.
