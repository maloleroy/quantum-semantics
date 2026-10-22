# Hamiltonian encoder research protocol

Date: 2026-09-16. Starting commit: 78e0ce7. Exploratory research, not a registered study.

## Questions fixed before held-out evaluation

1. Does a corpus graph remove dependence on arbitrary vocabulary labels?
2. Does alternating graph/context evolution improve causal prediction over static walks?
3. Does it improve over classical predictions built from the same training counts?
4. Which parts preserve genuine order, distinguish contexts, and survive graph ablation?

## Planned controls

- Original exponential QCSE encoding, deep ansatz, original bit loss and categorical loss.
- Unigram, smoothed bigram, recency transition pooling, learned classical token embedding.
- Static graph walk, alternating context/graph walk, count and PPMI graphs.
- Directed graph through a Hermitian source/target dilation, with success probability reported.
- Predictive-amplitude initialization plus quantum evolution, tested against its identity control.
- Graph removal, graph relabeling relative to words, context-order reversal, and depth sweep.

## Evaluation contract

Split by normalized sentence text before fitting vocabulary or corpus statistics. All exact
duplicates stay together. Vocabulary is fitted on training sentences with an explicit UNK;
report retained-token coverage, UNK rate and known-target metrics. Causal windows contain
preceding tokens only. Graphs, priors, transition tables use training data only. Fixed seeded
subsamples may bound optimization/evaluation cost; save their identities and full data hashes.
Select hyperparameters/checkpoints with validation NLL, then lock selections before evaluating
test. Use at least three optimization seeds for main models. Use 16 alternating/ansatz layers
as the primary deep comparison and 1/4/8/16/32 for depth diagnostics. Report categorical NLL,
perplexity, top-1/top-5, seed variation and sentence-group paired bootstrap intervals.

## Interpretation

The experiment changes both encoder and readout relative to the paper; do not attribute a
whole-system win solely to encoding. Train-only graph statistics are classical computation
and storage even when they are not trainable embedding weights. Exact statevector simulation
is not quantum hardware, and these small one-particle models do not establish quantum speedup.
Shared unitaries preserve fidelity; extra position registers require position-conditioned
operations/readout to retain ordering in word-only predictions. Report unsuccessful variants.

Detailed budgets and any departures will be preserved in per-stage configurations and REPORT.md.

## Validation-only adaptation after first screen

The one-seed, 15-epoch screen used 2,048 optimization targets, 2,048 validation
and 512 vocabulary entries. Count graphs beat PPMI/nPMI. A predictive-amplitude
model initialized with total walk time near 9.6 destroyed much of its useful prior.
Add a near-identity predictive residual (beta around .02 per layer), and initialize
the directed dilation the same way. This is an explicit initialization change,
not evidence of quantum superiority. Add chronological token-specific context
potentials (`ordered`) as an order-preserving control. Main comparisons will use
30 epochs, 8,192 optimization targets, 4,096 validation, 8,192 held-out test targets,
16 layers and seeds 11/22/33. All corpus statistics still use the full training split
(1,162,743 causal transitions). A small-corpus replication uses all training targets
and a full training-derived vocabulary. Test values remain unread at this point.

## Additional fixed diagnostic scope

The final test comparison lists were committed before test evaluation in `5c6756d`.
No subsequent model selection depends on these scores. Add a validation-only static
walk time/smoothing grid, a full-training classical control (10 epochs, batch 1,024,
same held-out examples), and an exploratory external SimLex-999 probe. SimLex is not
used for tuning, choosing models or refitting vocabulary. It only probes the overlap
of its word pairs with the already fixed training vocabulary; full-benchmark claims
are disallowed. Main quantum tests use exact gradients, not the earlier SPSA budget.

The predictive residual's main-validation improvement was approximately 0.002 NLL.
Before reading the main test scores, specify two additional classical controls:
a learned unigram mixture of the same bigram prior (exactly the no-evolution model),
and that mixture plus a learned temperature. Both use the same 8,192 examples,
30 epochs, batch size 128, seeds and validation selection. Their independent
selection lock is in `results/calibration/selection_lock.json`. This prevents
attributing a small classical-calibration effect to quantum interference.

A third original-encoding control, `qcse_born`, uses joint Born probabilities instead
of reconstructed independent-bit probabilities. It holds the measurement rule closer
to the Hamiltonian model, while retaining the original encoder and ansatz. Its separate
comparison was fixed before training/evaluation; it uses 512 classes so no invalid-basis
postselection is needed. It does not isolate every encoder/ansatz/statistics difference.
