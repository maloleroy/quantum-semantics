# Semantic decoder ablations

The 20-epoch local comparison uses the same `phrases` corpus, seed, grouped
64/16/20 split, four-qubit model, 5,000 replacement draws per epoch and learning
rate 0.003. Each run started from the same initialization. The first three rows
train all available parameters; the last row trains only the linear decoder and
output bias while leaving the embedding, encoder and circuit fixed.

| Ansatz / optimizer | Best epoch | Best validation CE | Best dot top-1 | Best cosine top-1 | Final test dot top-1 | Final test cosine top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trainable / all | 10 | 4.754 | 18.3% | 13.4% | 18.9% | 13.5% |
| frozen / all | 10 | **4.703** | **19.1%** | 13.6% | **18.9%** | 13.4% |
| zero / all | 10 | 4.694 | 18.3% | 13.3% | 17.5% | 12.4% |
| trainable / decoder only | 20 | 5.769 | 8.4% | 2.1% | 9.5% | 2.2% |

The normal training objective is the tied dot product
`hidden · E[word] + bias`; this is the score used for cross-entropy and the
reported `dot` accuracy. The model now also reports cosine retrieval, computed
after L2-normalizing `hidden` and each vocabulary embedding, without the output
bias. Cosine is therefore an evaluation comparison, not the training objective.

The decoder-only control does not recover the task: its held-out dot accuracy is
the 9.5% frequency baseline, while the complete model reaches 18.9%. This shows
that the small decoder layer is not producing the result by itself; useful signal
comes from jointly learned input embeddings and encoder/circuit parameters.
However, the frozen and zero ansatz controls match or slightly exceed the
trainable ansatz in this short run. The ansatz is therefore not demonstrated as
useful here. The current result supports treating it as an ablation target rather
than claiming a quantum contribution.

Validation cross-entropy improves through about epoch 10 and then worsens for all
full-model variants, while training loss continues to fall. The 20-epoch result
does not improve the earlier 10-epoch checkpoint on probability calibration.

The plot is [validation-ablation.png](outputs/semantic-report/validation-ablation.png).
Raw histories and test metrics are in the ignored run directories under
`outputs/semantic-ablation/`; the aggregate is
`outputs/semantic-report/ablation-summary.json`.
