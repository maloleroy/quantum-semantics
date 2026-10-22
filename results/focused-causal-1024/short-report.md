# Causal focused sweep — short report

All 16 causal fits completed with 1,024 training draws/epoch, 512 validation examples, and 2,048 final test examples.

- Exact word top-1 is zero almost everywhere, so the capped validation set cannot reliably rank causal parameters.
- Batch 16 is the strongest arm on the secondary paper-similarity metric and the only arm with a nonzero exact-word result.
- Two layers improve paper similarity over eight layers by about 22.5 percentage points.
- Alpha and window changes are small and seed-sensitive.
- Batch 256 is faster per epoch but is not a quality win: it has fewer optimizer updates and slightly worse secondary results than batch 64.

Use the detailed report and plots for the seed drift and the exact paired contrasts. These results are valid pipeline evidence, but not a strong exact-top-1 ranking because of metric saturation.
