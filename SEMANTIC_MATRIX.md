# Dataset and former-setup matrix

I ran 25 epochs for each of the five model setups across three local data
selections. The comparison uses cosine top-1/top-5 only.

| Data selection | Full trainable | No circuit | No encoding/decoding | Frozen circuit | Frozen encoder/decoder | Baseline |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| `phrases.csv` only | 14.7 / 34.0% | 12.9 / 32.1% | 0.6 / 11.6% | 14.7 / 35.2% | 7.9 / 23.9% | 9.5 / 16.9% |
| `cleaned_sentences.csv` only | 14.5 / 31.5% | 14.6 / 32.5% | 2.6 / 12.3% | 15.7 / 31.5% | 7.3 / 27.7% | 7.2 / 20.1% |
| Both files | 16.8 / 35.2% | 17.4 / 34.6% | 2.7 / 13.2% | 16.8 / 34.9% | 8.4 / 25.8% | 7.6 / 21.2% |

Each cell is `test cosine top-1 / top-5`. The Baseline is a frequency-only
predictor calculated from the training split; it does not train a model.

The five model setups are:

- **Full trainable:** trainable embeddings, encoder, circuit ansatz, decoder and
  output bias.
- **No circuit:** ansatz angles fixed to zero. Since RX(0) and controlled-RZ(0)
  are identity gates, this is equivalent to omitting the ansatz. The input RY/RZ
  encoding remains.
- **No encoding/decoding:** direct mean context embedding to the tied softmax;
  no quantum state, encoder projection or decoder.
- **Frozen circuit:** initial nonzero ansatz angles are fixed; surrounding
  embeddings, encoder and decoder train.
- **Frozen encoder/decoder:** quantum path remains, but the input projection and
  classical decoder are fixed; embeddings and ansatz train.

The local data protocol uses the selected source for sentences, but the complete
active pair still supplies the vocabulary (10,864 words). Tatoeba is excluded.
For speed, each selection is capped at 1,000 curated sentences: the 1,000 phrase
rows, a seeded 1,000-sentence sample from cleaned data, or a seeded 1,000-sentence
sample from both. Cleaning is dedupe, examples are causal windows of four, and
sentence groups are split approximately 64/16/20 for train/validation/test.
Each epoch draws 5,000 token examples with replacement, batch size is 64, the
seed is 42, and execution is CPU exact-statevector.

The main plots are:

- [test-cosine-bars.png](outputs/semantic-matrix-report/test-cosine-bars.png)
- [best-validation-cosine.png](outputs/semantic-matrix-report/best-validation-cosine.png)

The machine-readable table is
`outputs/semantic-matrix-report/matrix.json`; raw runs are under
`outputs/semantic-matrix-25/`.
