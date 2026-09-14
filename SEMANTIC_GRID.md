# Semantic decoder hyperparameter checks

These five additional 25-epoch runs vary one setting at a time around the
baseline. The metric in this table is cosine retrieval; top-5 is included.

| Setup | Best epoch | Best validation CE | Best validation cosine top-1 / top-5 | Test cosine top-1 / top-5 |
| --- | ---: | ---: | ---: | ---: |
| baseline: LR 0.003, alpha 0.05, 2 layers | 10 | 4.754 | 13.4% / 33.6% | 14.7% / 34.0% |
| LR 0.001 | 25 | 4.821 | 11.7% / 32.3% | 11.6% / 31.6% |
| LR 0.01 | 5 | 4.813 | **15.3% / 36.3%** | **14.6% / 36.3%** |
| alpha 0.01 | 11 | 4.766 | 15.0% / 34.2% | 12.9% / 32.7% |
| alpha 0.2 | 10 | 4.757 | 14.2% / 34.6% | 10.9% / 32.7% |
| 4 layers | 10 | **4.725** | 13.3% / 35.4% | 13.5% / 33.1% |

Here `alpha` is the initial standard scale of the trainable ansatz angles;
it is a semantic-prototype hyperparameter, separate from the legacy QCSE
context-decay alpha. The ansatz has `3 × qubits − 1` parameters per layer.

The higher learning rate is the only tested change that improves held-out cosine
top-5 over the baseline. The lower learning rate is still improving at epoch 25,
while the higher rate reaches its best validation CE at epoch 5 and then begins
to overfit. More layers lower validation CE in this run but do not improve test
cosine accuracy. No setting establishes a reliable benefit from the trainable
ansatz; these are single-seed local checks, not a broad statistical sweep.

## Data and training protocol

All runs selected `phrases.csv` as the training sentence source. The vocabulary
was still built from the complete active pair, `phrases.csv` plus
`cleaned_sentences.csv`, so it contained 10,864 words. Tatoeba was not used.
Default dedupe cleaning retained 1,000 phrase sentences. There was no sentence
cap beyond that source selection.

Examples were causal next-token windows of size four. Sentences were grouped
before splitting: 80% development and 20% test, then 80% of development for
training and 20% for validation, approximately 64/16/20. The same seed (42),
5,000 replacement token draws per epoch, batch size 64, embedding dimension 16,
four qubits, exact statevector backend and CPU execution were used throughout.
Each run trained for 25 epochs and evaluated the held-out test once at the end.

Raw runs are under `outputs/semantic-grid-25/`; the generated cosine-only plot
is [validation-ablation.png](outputs/semantic-report-25/validation-ablation.png).

## Encoding and decoding controls

| Pathway | Best validation CE | Best validation cosine top-1 / top-5 | Test cosine top-1 / top-5 |
| --- | ---: | ---: | ---: |
| Direct mean-embedding baseline, no encoding/decoding | 5.720 | 2.2% / 12.5% | 0.6% / 11.6% |
| Frozen encoding and decoding, trainable embeddings/ansatz | 4.863 | 7.4% / 26.9% | 7.9% / 23.9% |

The direct baseline averages the non-padding context embeddings and feeds that
vector directly to the tied vocabulary softmax. It has no encoder projection,
quantum state, circuit or decoder. The frozen pathway retains the quantum
encoding and circuit, but freezes the input projection and classical decoder;
the embedding table and ansatz remain trainable. Both controls are substantially
below the full trainable pathway (14.7% test cosine top-1 / 34.0% top-5), showing
that the learned encoder/decoder pathway matters for this setup.
