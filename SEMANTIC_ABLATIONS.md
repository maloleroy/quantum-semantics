# Semantic decoder ablations

The 25-epoch local comparison uses the same `phrases` corpus, seed, grouped
64/16/20 split, four-qubit model, 5,000 replacement draws per epoch and learning
rate 0.003. Each run started from the same initialization. The first three rows
train all available parameters; the last row trains only the linear decoder and
output bias while leaving the embedding, encoder and circuit fixed.

| Setup | Best epoch | Best validation CE | Best validation cosine top-1 | Final test cosine top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trainable ansatz / all | 10 | 4.754 | 13.4% | 14.7% |
| frozen ansatz / all | 10 | **4.703** | **13.6%** | **14.7%** |
| zero ansatz / all | 10 | 4.694 | 13.3% | 12.9% |
| trainable ansatz / decoder only | 25 | 5.669 | 2.0% | 2.1% |

The normal training objective remains the tied dot product
`hidden · E[word] + bias`. The report now uses cosine top-1 and top-5 as the only
retrieval metric: it L2-normalizes `hidden` and each vocabulary embedding and
does not use the output bias. Dot scores are not reported as performance metrics.

The four setups mean:

- **Trainable ansatz / all:** embeddings, input encoder, circuit rotations,
  decoder and output bias all receive gradients.
- **Frozen ansatz / all:** circuit rotations are initialized identically but held
  fixed; embeddings, encoder and decoder still learn.
- **Zero ansatz / all:** all circuit rotations are fixed at zero; the rest still
  learns. This tests whether the circuit transformation itself contributes.
- **Trainable ansatz / decoder only:** only the linear decoder and output bias
  learn; embeddings, encoder and circuit remain fixed at initialization.

The decoder-only control does not recover the task: its held-out cosine top-1 is
2.1%, while the complete model reaches 14.7%. This shows
that the small decoder layer is not producing the result by itself; useful signal
comes from jointly learned input embeddings and encoder/circuit parameters.
However, the frozen and zero ansatz controls match or slightly exceed the
trainable ansatz in this short run. The ansatz is therefore not demonstrated as
useful here. The current result supports treating it as an ablation target rather
than claiming a quantum contribution.

Validation cross-entropy improves through about epoch 10 and then worsens for all
full-model variants, while training loss continues to fall. The 25-epoch result
does not improve the earlier 10-epoch checkpoint on probability calibration.

The plot is [validation-ablation.png](outputs/semantic-report-25/validation-ablation.png).
Raw histories and test metrics are in the ignored run directories under
`outputs/semantic-ablation-25/`; the aggregate is
`outputs/semantic-report-25/ablation-summary.json`.
