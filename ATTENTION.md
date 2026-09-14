# Quantum attention: QCSE and classical inputs

Branch: `qcse-classical-attention`.

This single-head experiment follows the bilinear attention idea in the supplied
*GPT on a Quantum Computer*, sections 3.1-3.2, Figure 5 and equations 12-19.
The paper assumes classical embeddings and positional encodings before quantum
attention. Here both input choices feed the same trainable quantum attention
and classical word readout:

| Input | State preparation |
| --- | --- |
| `qcse` | Original exponential QCSE context matrix, then H/RX/RZ/CNOT encoding, independently for each causal prefix. No learned input embedding. |
| `classical` | Learned word embeddings plus sinusoidal positions, normalized and amplitude encoded. |

The default register has four qubits for both encoders. QCSE retains its original
matrix/gate construction but uses this configurable register, rather than the
production binary decoder's vocabulary-sized register (14 qubits on the current
corpus). This experiment predicts vocabulary logits; it does not reuse that binary
decoder or its ansatz. `--context-alpha` controls the fixed QCSE decay;
`--alpha` controls the initial quantum attention parameter scale.

## Attention and readout

Three independent RY/RZ/CNOT chain circuits produce quantum queries, keys and
values. For query position `i`, the score is `Re(<q_i|k_j>)` for real tokens at
positions `j <= i`, and zero otherwise. Scores can be negative: attention has no
softmax. The output state is proportional to `sum_j score[i,j] |v_j>`.
Normalization represents conditioning on successful ancilla and output-position
postselection. A completely masked or zero-norm output gives zero features.

Taking the real overlap extends the paper's real-vector bilinear form to complex
QCSE states. Indices use the usual query-row convention. Causal masking applies
to the input prefixes and the attention matrix; padding contributes neither
states nor scores. Classical positions count real tokens from zero.

Training directly simulates the postselected statevector action with Torch
autograd. `QuantumAttentionModel.attention_circuit(context)` also constructs a
small Qiskit reference: quantum value gates and an SVD unitary dilation of the
masked score matrix divided by sequence length. Tests compare its postselected
amplitudes with the differentiable implementation. The reference prepares states
and computes overlaps/dilation classically; efficient hardware oracles, finite
shots and their sampling cost are outside this implementation.

Exact XYZ expectations feed two linear layers producing full-vocabulary logits.
Cross-entropy uses these logits; cosine top-1/top-5 compare the intermediate
representation with output weight rows. Both variants have the same attention
and head initialization for a given seed. The classical input lookup adds
parameters, so total model sizes are not matched.

## Run and resume

```bash
uv sync --locked
# Both encoders; all 1,000 phrases, full shared vocabulary, same seed/split.
bash scripts/run_attention_comparison.sh --datasets phrases --epochs 10 --evaluate-test
# Omit --datasets for the small repetitive sanity corpus.
# Both complete curated sources, without a permanent training-example cap:
bash scripts/run_attention_comparison.sh --datasets phrases cleaned --epochs 10 --device cuda --evaluate-test

# Resume one saved run to a total epoch target, restoring model/optimizer/RNG/splits:
uv run python scripts/semantic_experiment.py --resume outputs/attention/qcse/<run> --epochs 20 --evaluate-test
```

`ATTENTION_OUTPUT` changes the comparison output root; `ATTENTION_ENCODING=qcse`
or `classical` selects one encoder. Other model/training flags pass through to
`scripts/semantic_experiment.py`; use its `--help`. Individual runs also accept
`--model attention --encoding qcse|classical`. Existing semantic invocations and
checkpoints retain their previous model selection.

Defaults: window 4, four qubits, embedding/readout dimension 16, two circuit
layers, attention angle scale 0.05, Adam LR 0.003, batch 64, seed 42, and 5,000
replacement draws per epoch. The final partial batch is retained. Data uses one
sentence-grouped 64/16/20 split. Monitoring uses at most 512 training examples and
the full validation split. Test is scored only with `--evaluate-test`; request it
at the final intended stage. Each run saves unique output paths, atomic checkpoints,
vocabulary, data provenance, histories and decoded test predictions.

For two independent full-pool GPU jobs on the existing MIG allocation:

```bash
git switch qcse-classical-attention
uv sync --locked
mkdir -p logs
sbatch slurm-attention-prod10.sbatch
```

The array runs one encoder per job for ten epochs. `ATTENTION_EPOCHS` and
`ATTENTION_OUTPUT` override its epoch target and output root. It preserves Slurm's
device mask and runs both the existing backend preflight and a small differentiable
attention fit on CUDA before the full-pool fit. Full-pool evaluation/checkpointing
still scales with corpus size; local runs below establish CPU behavior only.

## Local results: 2026-09-14

Both configurations used all 1,000 phrases, the complete 10,864-word vocabulary,
seed 42 and the defaults above. Each trained for five epochs and resumed to ten
(50,000 training draws total). Splits contained 3,991/998/1,246 token examples;
vocabulary, contexts, splits and sampler states matched across encoders. Test was
scored once after epoch ten. No configuration tuning was performed.

| Encoder | Parameters | Initial validation CE | Epoch-10 validation CE | Test CE | Test cosine top-1 / top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QCSE | 184,944 | 9.331 | 5.520 | 5.677 | 4.49% / 16.61% |
| Classical | 358,784 | 9.318 | 4.995 | 5.150 | 8.03% / 27.61% |

Classical encoding performed better in this single-seed comparison. Both cosine
top-1 scores were below the training-frequency baseline (9.47%); its top-5 was
16.85%. These are final-epoch results, not a validation-selected checkpoint.

Evidence is in ignored `outputs/attention-validation/{qcse,classical}/` folders.
Tests cover original encoding and Q/K/V circuit parity, postselected block-encoding
parity for both encoders, gradients, causality, padding, checkpoint reload and exact
resume equivalence. Local wrapper runs and the CPU backend preflight also passed.
CUDA/Apple MPS and actual Slurm submission were not available for local validation.

## 25-epoch cross-validation results

The tracked report and figures are in [results/attention-cv-25](results/attention-cv-25).
This run used both datasets with balanced sampling, all 1,000 phrase rows and 4,000
cleaned rows, 5,000 replacement draws per epoch, 25 epochs per fit, and two repeated
20% validation splits inside the development partition. Each setup had two folds and
a fresh development refit before its single held-out test score.

| Setup | CV validation CE mean ± SD | Test CE | Test cosine top-1 / top-5 |
| --- | ---: | ---: | ---: |
| QCSE, one layer | 5.6910 ± 0.0311 | 5.7268 | 4.35% / 18.07% |
| Classical, one layer | **4.6028 ± 0.0235** | **4.6084** | **9.94% / 30.48%** |
| QCSE, two layers | 5.6932 ± 0.0578 | 5.7180 | 4.11% / 15.45% |
| Classical, two layers | 4.6085 ± 0.0326 | 4.6209 | 9.74% / 31.30% |

The learned classical encoder was superior on this protocol at both depths. The
one-layer classical setup had the best mean validation cross-entropy; the two-layer
classical setup had the highest test top-5. This is a CPU statevector experiment with
a 5,000-sentence cap, not evidence of hardware speed or quantum advantage. Omit
`--max-sentences` in `scripts/attention_cross_validation.py` on an available GPU to
use the complete curated pool.
