# Causal focused sweep report — 16 completed fits

## Protocol and scope

This report covers all eight current causal configurations, each with two repetitions (seeds 42 and 43). It uses the full phrases + cleaned corpus, sentence-grouped 64/16/20 splits, 1,024 replacement draws per training epoch, 512 fixed validation-monitor examples, 2,048 held-out test examples, 50 maximum epochs, and patience 8 with a 10-epoch minimum.

The requested primary metric is exact word top-1: all 14 decoded bits must match the target word. The test score was never used for selection. Paper-similarity accuracy (at least half of bits correct) and BCE are shown as secondary diagnostics only.

## Aggregate results

| Setup | α | Window | Layers | Batch | Validation exact top-1 | Test exact top-1 | Test paper similarity | Test BCE | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (α=1, w=4, L=8, B=64) | 1.0 | 4 | 8 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 51.95% ± 35.98% | 0.725 ± 0.051 | 14.0 ± 5.7 |
| α=0.1 | 0.1 | 4 | 8 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 57.45% ± 44.50% | 0.719 ± 0.054 | 20.5 ± 0.7 |
| α=3 | 3.0 | 4 | 8 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 51.83% ± 36.01% | 0.725 ± 0.047 | 12.5 ± 9.2 |
| L=2 | 1.0 | 4 | 2 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 74.49% ± 17.51% | 0.699 ± 0.001 | 10.0 ± 0.0 |
| w=2 | 1.0 | 2 | 8 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 50.61% ± 38.01% | 0.733 ± 0.051 | 7.0 ± 9.9 |
| w=8 | 1.0 | 8 | 8 | 64 | 0.00% ± 0.00% | 0.00% ± 0.00% | 57.13% ± 44.95% | 0.730 ± 0.072 | 14.0 ± 11.3 |
| B=16 | 1.0 | 4 | 8 | 16 | 0.10% ± 0.14% | 0.17% ± 0.24% | 89.87% ± 12.26% | 0.615 ± 0.071 | 33.5 ± 10.6 |
| B=256 | 1.0 | 4 | 8 | 256 | 0.00% ± 0.00% | 0.00% ± 0.00% | 51.61% ± 36.39% | 0.741 ± 0.066 | 9.5 ± 0.7 |

## Primary top-1 conclusion

Exact word top-1 is saturated at zero for the baseline, both alpha changes, both window changes, the 2-layer setup, and both batch-256 repetitions. Only batch-16 repetition 0 is nonzero (0.34% test top-1); its repetition mean is 0.17%. Therefore the 512-example validation cap does not provide enough resolution for exact-word top-1 to rank most causal settings. This is a metric-resolution limitation, not evidence that all configurations are equivalent.

The implementation still selects by exact top-1 first, then paper similarity and BCE as deterministic tie-breakers. Because exact top-1 is usually tied at zero, most selected epochs are effectively decided by those secondary tie-breakers; this is recorded as a limitation rather than relabeled as primary-top-1 evidence.

## Parameter and setup drift

- Alpha: α=0.1 is modestly above baseline on paper similarity (+5.5 percentage points paired mean), while α=3 is essentially neutral (−0.1 points). The α=0.1 gain is seed-dependent; it is not a decisive optimum.
- Window: window 8 is mildly above baseline (+5.2 points), while window 2 is slightly below (−1.3 points). Both effects are small relative to repetition drift.
- Layers: the 2-layer model is the clearest depth improvement in this sweep (+22.5 paper-similarity points versus 8 layers, with both repetitions improving), although exact word top-1 remains zero. Its test BCE is also slightly lower as a diagnostic, but BCE is not the selection target.
- Batch size: batch 16 is the strongest setup by secondary quality (+37.9 paper-similarity points versus batch 64; −0.1095 BCE), and it is the only arm with a nonzero exact-word result. Batch 256 is not a quality win: it is slightly below batch 64 on paper similarity (−0.3 points) and worse on BCE (+0.016). The larger batch performs only 4 Adam updates per 1,024-draw epoch, versus 16 for batch 64 and 64 for batch 16.

## Repetition drift and validity

The seed effect is very large for most 8-layer settings: seed 42 typically reaches roughly 0.77–0.99 paper similarity, whereas seed 43 is often roughly 0.23–0.27. Batch 16 is more robust but still differs between repetitions. Two repetitions are sufficient to expose this instability, not to provide tight uncertainty intervals.

All fits completed with finite histories and fixed sentence-grouped splits. Training draws may repeat examples by design, but the held-out sentence groups remain separated from training and validation. Test metrics were computed once after validation-based epoch selection.

## Interpretation

The defensible causal conclusion from this run is that smaller batches and, secondarily, fewer layers are more promising under the current SPSA/Adam implementation. Alpha and window effects are weak and seed-sensitive. Batch 256 is faster per fixed epoch but does not compensate for its much smaller optimizer-update budget in quality. A final claim about exact word top-1 requires a larger validation monitor or a denser retrieval metric; the current capped run should not be used to declare a top-1 winner beyond the observed batch-16 signal.

## Figures

- `causal-test-metrics.png`: test primary/secondary metrics with repetition points and mean ± SD.
- `causal-validation-curves.png`: validation drift across training epochs.
- `causal-parameter-effects.png`: matched parameter changes versus baseline.
- `causal-seed-drift.png`: repetition drift by setup.
- `causal-selected-epochs.png`: selected and stopped epochs.
- `causal-batch-tradeoff.png`: batch quality versus optimizer updates per epoch.
