---
name: qcse-experiments
description: Run bounded reproducible QCSE ordering and local vocabulary-embedding experiments in this repository.
---
Read ../../PROGRESS.md and the runner's saved protocol before extending experiments.
Freeze data, splits, target IDs, initialization and optimization budget for paired
ordering comparisons. Keep all permutation seeds; never select using test scores.
Report actual upload-layer counts: window 4 on 10 qubits has only one layer.
Pair layouts move RX/RZ pairs; word-ID maps change numerical context features;
context-position permutations change linguistic positions. Name these separately.
Only cache encodings if layouts are fixed across training. Layer-specific frozen maps
are cacheable; epoch/batch resampling would require recomputation and inference policy.
Fit projections and predictors on training contexts only. Pretrained vocabulary vectors
are external knowledge; distinguish order-only use from direct numeric input. Compare
semantic layouts with shuffled-vector controls and direct features with classical heads.
Record raw outputs, hashes, all seeds, wall time and a report per experiment. Interpret
short pilots as sensitivity evidence, not converged performance or quantum advantage.
